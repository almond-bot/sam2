import asyncio
from fastapi import FastAPI, File, Form, UploadFile
import json
import uvicorn
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import lz4.frame

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

app = FastAPI()

grounding_dino_processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
)
grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).to("cuda")

sam2 = build_sam2(
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoints/sam2.1_hiera_large.pt",
    device=torch.device("cuda"),
    apply_postprocessing=False,
)
sam2_mask_generator = SAM2AutomaticMaskGenerator(
    sam2,
    stability_score_thresh=0.8,
)

def grounding_dino_inference(img: np.ndarray, item: str) -> np.ndarray:
    img = Image.fromarray(img)
    text_labels = [[item]]

    inputs = grounding_dino_processor(
        images=img, text=text_labels, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        outputs = grounding_dino(**inputs)

    results = grounding_dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[img.size[::-1]],
    )

    return results[0]["boxes"].cpu().numpy()

def sam2_inference(
    img: np.ndarray,
) -> np.ndarray:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = sam2_mask_generator.generate(img)

    return masks

def warmup_models():
    grounding_dino_inference(np.zeros((1080, 1920, 3), dtype=np.uint8), "warmup")
    sam2_inference(np.zeros((1080, 1920, 3), dtype=np.uint8))


def _wrap_to_90(angle_deg: float) -> float:
    """Wrap angle to the range [-90, 90] degrees."""
    angle_deg = ((angle_deg + 180) % 360) - 180  # -> [-180, 180]
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    return angle_deg


def get_item_offset(
    cam_p: dict, depth: np.ndarray, mask: np.ndarray
) -> dict:
    # Apply mask to depth to get only the depth values of the object
    masked_depth = depth.copy()
    masked_depth[~mask] = np.nan

    # Get 3D points for the segmented object
    valid_points = np.where(~np.isnan(masked_depth))
    y_coords, x_coords = valid_points
    z_coords = masked_depth[valid_points]

    # Calculate center point of the mask
    center_x_loc = int(np.mean(x_coords))
    center_y_loc = int(np.mean(y_coords))

    center_z = masked_depth[center_y_loc, center_x_loc]
    center_x = ((center_x_loc - cam_p["cx"]) * center_z) / cam_p["fx"]
    center_y = ((center_y_loc - cam_p["cy"]) * center_z) / cam_p["fy"]

    x_coords = ((x_coords - cam_p["cx"]) * z_coords) / cam_p["fx"]
    y_coords = ((y_coords - cam_p["cy"]) * z_coords) / cam_p["fy"]

    # Stack coordinates into a point cloud for PCA
    points = np.column_stack((x_coords, y_coords, z_coords))

    # Center the points for PCA
    centered_points = points - np.mean(points, axis=0)

    # Perform PCA to get principal axes
    covariance_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Order eigenvectors by descending eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Enforce right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    # Rotation angles (radians)
    roll = np.arctan2(eigenvectors[1, 2], eigenvectors[2, 2])
    pitch = np.arctan2(
        -eigenvectors[0, 2], np.sqrt(eigenvectors[1, 2] ** 2 + eigenvectors[2, 2] ** 2)
    )
    yaw = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Convert to degrees and wrap to [-90, 90]
    roll_deg = _wrap_to_90(np.degrees(roll))
    pitch_deg = _wrap_to_90(np.degrees(pitch))
    yaw_deg = _wrap_to_90(np.degrees(yaw))

    return {
        "x": float(center_x),
        "y": float(center_y),
        "z": float(center_z),
        "roll": float(roll_deg),
        "pitch": float(pitch_deg),
        "yaw": float(yaw_deg)
    }

def save_mask_overlays(rgb: np.ndarray, masks: list[np.ndarray]):
    overlay = rgb.copy()
    green_overlay = np.zeros_like(overlay)
    for mask in masks:
        green_overlay[mask] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)
    cv2.imwrite("mask_overlays.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def save_mask_overlay(rgb: np.ndarray, mask: np.ndarray):
    overlay = rgb.copy()
    green_overlay = np.zeros_like(overlay)
    green_overlay[mask] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)
    
    # Save the overlay image
    cv2.imwrite("mask_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def bin_picking_inference(rgb: np.ndarray, depth: np.ndarray, item: str, cam_params: dict):
    # Run inference
    bboxes = grounding_dino_inference(rgb, item)
    if len(bboxes) == 0:
        return

    x1, y1, x2, y2 = bboxes[0]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)

    rgb_crop = rgb[y1:y2, x1:x2]
    depth_crop = depth[y1:y2, x1:x2]

    valid_depth_mask = ~np.isnan(depth_crop)
    if not np.any(valid_depth_mask):
        return

    mask_crops = sam2_inference(rgb_crop)
    mask_crops = [m["segmentation"] for m in mask_crops]

    # Remove masks that touch the edge of the image
    mask_crops = [m for m in mask_crops if not (
        np.any(m[0, :]) or np.any(m[-1, :]) or 
        np.any(m[:, 0]) or np.any(m[:, -1])
    )]

    min_depth = np.inf
    mask_crop = None
    for m in mask_crops:
        d = depth_crop.copy()
        d[~m] = np.nan
        d_min = np.nanmin(d)
        if d_min < min_depth:
            min_depth, mask_crop = d_min, m

    if mask_crop is None:
        return
    
    save_mask_overlays(rgb_crop, mask_crops)
    save_mask_overlay(rgb_crop, mask_crop)

    mask = np.zeros(depth.shape, dtype=bool)
    mask[y1:y2, x1:x2] = mask_crop

    item_offset = get_item_offset(cam_params, depth, mask)
    return item_offset
    

@app.post("/")
async def root(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    rgb_shape: str = Form(...),
    depth_shape: str = Form(...),
    item: str = Form(...),
    cam_params: str = Form(...),
):
    h, w = map(int, rgb_shape.split(","))
    dh, dw = map(int, depth_shape.split(","))

    rgb_buf, d_lz4 = await asyncio.gather(rgb.read(), depth.read())

    rgb = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR).reshape((h, w, 3))

    depth_u16 = np.frombuffer(lz4.frame.decompress(d_lz4), np.uint16).reshape((dh, dw))
    depth = depth_u16.astype(np.float32) / 100.0

    cam_params = json.loads(cam_params)

    item_offset = bin_picking_inference(rgb, depth, item, cam_params)

    return item_offset

def main():
    warmup_models()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
