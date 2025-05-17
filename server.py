from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import numpy as np
from PIL import Image
import io

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI()

sam2 = build_sam2(
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoints/sam2.1_hiera_large.pt",
    device=torch.device("cuda"),
    apply_postprocessing=False,
)
sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2)

def sam2_inference(
    img: np.ndarray,
) -> np.ndarray:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = sam2_mask_generator.generate(img)

    return masks

@app.post("/")
async def root(file: UploadFile = File(...)):
    # Read and convert the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img_array = np.array(image)
    
    # Run inference
    masks = sam2_inference(img_array)
    
    return {"masks": masks.tolist()}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
