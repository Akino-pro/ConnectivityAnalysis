import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Load grayscale image
# ============================================================
img = cv2.imread("ring_pure.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load image. Check the filename/path.")

# Copy for erosion and dilation versions
ero_img = img.astype(np.float32).copy()
dil_img = img.astype(np.float32).copy()

# ============================================================
# 2. Parameters
# ============================================================
iteration   = 50
kernel_size = 3

# Binary kernel (shape only matters; values are 0/1 internally)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

# Threshold to separate shape from background (tune if needed)
# Assumes background is pure white (255) and ring is darker.
shape_thresh = 250

# ============================================================
# 3. Helper: compute binary shape and background masks
# ============================================================
def get_shape_mask(gray):
    """
    Return a binary mask where shape = 1 (non-background),
    background = 0. Threshold can be tuned based on your ring intensities.
    """
    return (gray < shape_thresh).astype(np.uint8)

def get_background_mask(gray):
    return 1 - get_shape_mask(gray)

# ============================================================
# 4. Iterative "binary-like" erosion with grayscale averaging
# ============================================================
# Erosion: move shape boundary inward
for _ in range(iteration):
    shape_mask = get_shape_mask(ero_img)

    # Morphological gradient gives boundary of the shape
    # (pixels where the shape meets background)
    shape_mask_u8 = (shape_mask * 255).astype(np.uint8)
    shape_edge = cv2.morphologyEx(shape_mask_u8, cv2.MORPH_GRADIENT, kernel)
    edge_mask = shape_edge > 0

    # On the edge, move intensity towards white (255) by averaging
    # and rounding up: new = ceil((I + 255) / 2)
    ero_img[edge_mask] = np.ceil(0.5 * (ero_img[edge_mask] + 255.0))

# Clip and convert back to uint8
ero_out = np.clip(ero_img, 0, 255).astype(np.uint8)

# ============================================================
# 5. Iterative "binary-like" dilation with grayscale averaging
# ============================================================
# Dilation: move shape boundary outward into background
for _ in range(iteration):
    # Background is where gray ~ 255 (or above threshold)
    bg_mask = get_background_mask(dil_img)

    bg_mask_u8 = (bg_mask * 255).astype(np.uint8)
    # Background edge: boundary between background and shape
    bg_edge = cv2.morphologyEx(bg_mask_u8, cv2.MORPH_GRADIENT, kernel)
    edge_mask = bg_edge > 0

    # On the edge, move intensity towards black (0) by averaging
    # and rounding up: new = ceil((I + 0) / 2) = ceil(I / 2)
    # Darker pixels (small I) get closer to 0 faster in relative terms.
    dil_img[edge_mask] = np.ceil(0.5 * dil_img[edge_mask])

# Clip and convert back to uint8
dil_out = np.clip(dil_img, 0, 255).astype(np.uint8)

# ============================================================
# 6. Display results
# ============================================================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ero_out, cmap='gray', vmin=0, vmax=255)
plt.title(f"Custom Erosion\n(white 255 averaging, {iteration} iters)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(dil_out, cmap='gray', vmin=0, vmax=255)
plt.title(f"Custom Dilation\n(black 0 averaging, {iteration} iters)")
plt.axis("off")

plt.tight_layout()
plt.show()
