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
    "IDEA-Research/grounding-dino-base"
)
grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to("cuda")

sam2 = build_sam2(
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoints/sam2.1_hiera_large.pt",
    device=torch.device("cuda"),
    apply_postprocessing=False,
)
sam2_mask_generator = SAM2AutomaticMaskGenerator(
    sam2,
    points_per_side=96,
    points_per_batch=256,
    stability_score_thresh=0.1,
    min_mask_region_area=800,
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
        box_threshold=0.2,
        text_threshold=0.15,
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
    cam_params: dict, depth: np.ndarray, mask: np.ndarray
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

    # Get a 5x5 grid around the center point
    grid_size = 2  # This gives us a 5x5 grid (2 pixels in each direction)
    y_indices = np.arange(center_y_loc - grid_size, center_y_loc + grid_size + 1)
    x_indices = np.arange(center_x_loc - grid_size, center_x_loc + grid_size + 1)
    
    # Get all depth values in the grid
    grid_depths = masked_depth[y_indices[:, None], x_indices]
    
    # Calculate average depth, ignoring NaN values
    center_z = np.nanmean(grid_depths)
    
    # Calculate x and y coordinates for each point in the grid
    grid_x = ((x_indices - cam_params["cx"]) * center_z) / cam_params["fx"]
    grid_y = ((y_indices - cam_params["cy"]) * center_z) / cam_params["fy"]
    
    # Calculate average x and y coordinates
    center_x = np.nanmean(grid_x)
    center_y = np.nanmean(grid_y)

    x_coords = ((x_coords - cam_params["cx"]) * z_coords) / cam_params["fx"]
    y_coords = ((y_coords - cam_params["cy"]) * z_coords) / cam_params["fy"]

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

    # Ensure Z-axis points down (negative Z in camera frame)
    if eigenvectors[2, 2] > 0:
        eigenvectors *= -1

    # Rotation angles (radians)
    rx = np.arctan2(eigenvectors[1, 2], eigenvectors[2, 2])
    ry = np.arctan2(
        -eigenvectors[0, 2], np.sqrt(eigenvectors[1, 2] ** 2 + eigenvectors[2, 2] ** 2)
    )

    # Convert to degrees and wrap to [-90, 90]
    rx_deg = _wrap_to_90(np.degrees(rx))
    ry_deg = _wrap_to_90(np.degrees(ry))

    return {
        "x": float(center_x),
        "y": float(center_y),
        "z": float(center_z),
        "rx": float(rx_deg),
        "ry": float(ry_deg),
        "rz": 0
    }

def save_mask_overlays(rgb: np.ndarray, masks: list[np.ndarray]):
    overlay = rgb.copy()
    color_overlay = np.zeros_like(overlay)
    
    # Generate different colors for each mask
    colors = []
    for i in range(len(masks)):
        # Use HSV color space to generate evenly spaced colors
        hue = (i * 180 / len(masks)) % 180
        # Convert HSV to RGB (OpenCV uses BGR)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(color)
    
    for mask, color in zip(masks, colors):
        color_overlay[mask] = color
    
    overlay = cv2.addWeighted(overlay, 0.7, color_overlay, 0.3, 0)
    cv2.imwrite("mask_overlays.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def save_mask_overlay(rgb: np.ndarray, mask: np.ndarray):
    overlay = rgb.copy()
    green_overlay = np.zeros_like(overlay)
    green_overlay[mask] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)
    
    # Save the overlay image
    cv2.imwrite("mask_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def apply_depth_to_rgb(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    depth_norm = depth.copy()
    depth_norm[depth_norm == 0] = np.nanmax(depth_norm)
    depth_norm = np.nan_to_num(depth_norm, nan=np.nanmax(depth_norm))
    depth_norm = cv2.normalize(depth_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    depth_rgb = np.stack([depth_norm]*3, axis=-1)
    rgb_with_depth = cv2.addWeighted(rgb, 0.5, depth_rgb, 0.5, 0)
    return rgb_with_depth

def mask_to_pick(depth_crop: np.ndarray, mask_crops: list[np.ndarray]) -> np.ndarray:
    best_score = -np.inf
    mask_crop = None

    for m in mask_crops:
        d_obj = depth_crop.copy()
        d_obj[~m] = np.nan

        obj_depth = d_obj[m]
        if len(obj_depth) == 0:
            continue

        obj_median = np.nanmedian(obj_depth)
        obj_percentile10 = np.nanpercentile(obj_depth, 10)

        # Use a separate depth copy for background
        d_bg = depth_crop.copy()

        # Erode the object mask to shrink it slightly
        kernel = np.ones((5, 5), np.uint8)
        mask_eroded = cv2.erode(m.astype(np.uint8), kernel, iterations=1).astype(bool)

        bg_mask = ~m
        bg_mask[mask_eroded] = False

        bg_depth = d_bg[bg_mask]
        bg_depth = bg_depth[~np.isnan(bg_depth)]
        if len(bg_depth) == 0:
            bg_median = np.nanmedian(depth_crop[~np.isnan(depth_crop)])
        else:
            bg_median = np.median(bg_depth)

        height_above_bg = bg_median - obj_median
        score = (0.6 * height_above_bg) - (0.4 * obj_percentile10)

        if score > best_score:
            best_score = score
            mask_crop = m

    return mask_crop

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

    rgb_with_depth = apply_depth_to_rgb(rgb_crop, depth_crop)

    mask_crops = sam2_inference(rgb_with_depth)
    mask_crops = [m["segmentation"] for m in mask_crops]

    # Remove masks that touch the edge of the image
    mask_crops = [m for m in mask_crops if not (
        np.any(m[0, :]) or np.any(m[-1, :]) or 
        np.any(m[:, 0]) or np.any(m[:, -1])
    )]

    # Remove masks that are fully enclosed in another mask and are less than 10% of parent mask area
    filtered_masks = []
    for i, mask1 in enumerate(mask_crops):
        is_enclosed = False
        for j, mask2 in enumerate(mask_crops):
            if i != j and np.all(np.logical_or(~mask1, mask2)):
                # Calculate areas
                area1 = np.sum(mask1)
                area2 = np.sum(mask2)
                # Only remove if enclosed mask is less than 10% of parent mask
                if area1 < 0.1 * area2:
                    is_enclosed = True
                    break
        if not is_enclosed:
            filtered_masks.append(mask1)
    mask_crops = filtered_masks

    mask_crop = mask_to_pick(depth_crop, mask_crops)

    if mask_crop is None:
        return

    H, W = rgb_crop.shape[:2]
    final_masks = []

    ys, xs = np.where(mask_crop)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1    

    rgb_reg   = rgb_crop[y0:y1, x0:x1]
    depth_reg = depth_crop[y0:y1, x0:x1]

    rgbd_reg = apply_depth_to_rgb(rgb_reg, depth_reg)

    for o in sam2_inference(rgbd_reg):
        sub = o["segmentation"].astype(bool)
        if not sub.any():
            continue

        sub[mask[y0:y1, x0:x1] == 0] = False

        full = np.zeros((H, W), dtype=bool)
        full[y0:y1, x0:x1] = sub
        final_masks.append(full)

    mask_crop = mask_to_pick(depth_crop, final_masks)

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
