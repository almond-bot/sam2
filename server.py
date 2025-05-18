import asyncio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
import uvicorn
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import cv2

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
sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2)

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

@app.post("/")
async def root(
    rgb_file: UploadFile = File(...),
    depth_file: UploadFile = File(...),
    rgb_shape: str = Form(...),
    depth_shape: str = Form(...),
    item: str = Form(...),
):
    # Read the raw bytes
    rgb_contents, depth_contents = await asyncio.gather(rgb_file.read(), depth_file.read())
    
    # Parse shapes
    rgb_shape = tuple(map(int, rgb_shape.split(",")))
    depth_shape = tuple(map(int, depth_shape.split(",")))

    # Convert bytes to numpy arrays
    rgb = np.frombuffer(rgb_contents, dtype=np.uint8).reshape(rgb_shape)
    depth = np.frombuffer(depth_contents, dtype=np.float32).reshape(depth_shape)

    # Run inference
    bboxes = grounding_dino_inference(rgb, item)
    if len(bboxes) == 0:
        return

    # Use first bbox and shrink by 5%
    x1, y1, x2, y2 = bboxes[0]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = 0.95 * (x2 - x1), 0.95 * (y2 - y1)
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)

    rgb_crop = rgb[y1:y2, x1:x2]
    depth_crop = depth[y1:y2, x1:x2]

    valid_depth_mask = ~np.isnan(depth_crop)
    if not np.any(valid_depth_mask):
        return

    mask_crops = sam2_inference(rgb_crop)
    mask_crops = [m["segmentation"] for m in mask_crops]

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

    mask = np.zeros(depth.shape, dtype=bool)
    mask[y1:y2, x1:x2] = mask_crop

    return Response(content=mask.tobytes(), media_type="application/octet-stream")

def main():
    warmup_models()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
