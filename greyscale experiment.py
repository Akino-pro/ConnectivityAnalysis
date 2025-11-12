import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Adjustable parameters
# ============================
FILENAME     = "pure_ring.png"   # pure white background
KERNEL_SIZE  = 3                # kernel size (odd recommended)
ITERATIONS   = 200                # number of steps
DISP_CONST   = 5                 # grayscale change per step
SHAPE_THRESH = 250               # ring (shape) = gray < SHAPE_THRESH
WHITE_VAL    = 255
BLACK_VAL    = 0

# ============================
# Load image & kernel
# ============================
img0 = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
if img0 is None:
    raise FileNotFoundError(f"Could not load {FILENAME}")
img0 = img0.astype(np.float32)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

# ============================
# Propagating EROSION (inward, unchanged)
# ============================
def propagate_erosion(gray_f32, n_iter, step):
    g = gray_f32.copy()
    for _ in range(n_iter):
        white_mask = (g >= WHITE_VAL).astype(np.uint8)
        dilated = cv2.dilate(white_mask, kernel, iterations=1)
        frontier = (dilated == 1) & (g < WHITE_VAL)
        g[frontier] += step
        np.clip(g, 0, 255, out=g)
    return g

# ============================
# Propagating DILATION (hard-skin outward model)
# - Starts from the ring’s edge.
# - Once a pixel becomes black (0), it becomes part of the hard shell.
# - The shell expands outward each iteration.
# ============================
def propagate_dilation_hardskin(gray_f32, n_iter, step):
    g = gray_f32.copy()
    # Original ring acts as the initial seed
    seed_mask = (g < SHAPE_THRESH).astype(np.uint8)
    hard_mask = np.zeros_like(g, dtype=np.uint8)  # tracks fully hardened (black) pixels

    for _ in range(n_iter):
        # Current hard region = original ring + already black pixels
        hard_mask = np.clip(seed_mask + (g <= BLACK_VAL).astype(np.uint8), 0, 1)

        # Dilate to find the next frontier of soft white pixels
        dilated = cv2.dilate(hard_mask, kernel, iterations=1)
        frontier = (dilated == 1) & (hard_mask == 0)

        # Darken frontier (simulate hardening)
        g[frontier] -= step

        # Saturate to 0 so new black regions join the hard shell
        np.clip(g, 0, 255, out=g)

    return g

# ============================
# Run both processes
# ============================
ero = propagate_erosion(img0, ITERATIONS, DISP_CONST)
dil = propagate_dilation_hardskin(img0, ITERATIONS, DISP_CONST)

# ============================
# Display results
# ============================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img0.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ero.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
plt.title(f"Erosion (inward)\nK={KERNEL_SIZE}, it={ITERATIONS}, step={DISP_CONST}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(dil.astype(np.uint8), cmap='gray', vmin=0, vmax=255)
plt.title(f"Dilation (hard-skin outward)\nK={KERNEL_SIZE}, it={ITERATIONS}, step={DISP_CONST}")
plt.axis("off")

plt.tight_layout()
plt.show()
