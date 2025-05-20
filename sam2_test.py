import cv2
import numpy as np

from bin_picking import sam2_inference, save_mask_overlays

rgb_crop = np.load("rgb_crop.npy")
depth_crop = np.load("depth_crop.npy")

depth_norm = depth_crop.copy()
depth_norm[depth_norm == 0] = np.nanmax(depth_norm)
depth_norm = np.nan_to_num(depth_norm, nan=np.nanmax(depth_norm))

depth_norm = (255 * (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min())).astype(np.uint8)

depth_rgb = np.stack([depth_norm]*3, axis=-1)
rgb_with_depth = cv2.addWeighted(rgb_crop, 0.5, depth_rgb, 0.5, 0)

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

H, W = rgb_crop.shape[:2]
final_masks = []

for mask in mask_crops:
    ys, xs = np.where(mask)
    if ys.size == 0:
        continue
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1    

    rgb_reg   = rgb_crop[y0:y1, x0:x1]
    depth_reg = depth_crop[y0:y1, x0:x1]

    depth_reg[depth_reg == 0] = np.nanmax(depth_reg)
    depth_reg = np.nan_to_num(depth_reg, nan=np.nanmax(depth_reg))
    depth_reg = cv2.normalize(depth_reg, None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)

    rgbd_reg = cv2.addWeighted(
        rgb_reg, 0.5,
        np.repeat(depth_reg[..., None], 3, axis=-1), 0.5, 0
    )

    for o in sam2_inference(rgbd_reg):
        sub = o["segmentation"].astype(bool)
        if not sub.any():
            continue

        sub[mask[y0:y1, x0:x1] == 0] = False

        full = np.zeros((H, W), dtype=bool)
        full[y0:y1, x0:x1] = sub
        final_masks.append(full)

mask_crops = final_masks

save_mask_overlays(rgb_crop, mask_crops)
