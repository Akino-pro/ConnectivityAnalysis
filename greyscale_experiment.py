import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Adjustable parameters
# ============================
FILENAME     = "Figure_1.png"   # pure white background
KERNEL_SIZE  = 5                 # kernel size (odd recommended)
DISP_CONST   = 1                # grayscale change per step
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
# Propagating EROSION (1 step)
# ============================
def propagate_erosion(gray_f32, step):
    g = gray_f32.copy()
    white_mask = (g >= WHITE_VAL).astype(np.uint8)
    dilated = cv2.dilate(white_mask, kernel, iterations=1)
    frontier = (dilated == 1) & (g < WHITE_VAL)
    g[frontier] += step
    np.clip(g, 0, 255, out=g)
    return g

# ============================
# Propagating DILATION (1 step)
# ============================
def propagate_dilation_hardskin(gray_f32, step):
    g = gray_f32.copy()
    seed_mask = (g < SHAPE_THRESH).astype(np.uint8)
    hard_mask = (seed_mask + (g <= BLACK_VAL).astype(np.uint8))
    hard_mask = np.clip(hard_mask, 0, 1)

    dilated = cv2.dilate(hard_mask, kernel, iterations=1)
    frontier = (dilated == 1) & (hard_mask == 0)

    g[frontier] -= step
    np.clip(g, 0, 255, out=g)
    return g

# ============================
# Intersection function
# ============================
def intersection(fig1, fig2):
    if fig1.shape != fig2.shape:
        raise ValueError("Images must have the same dimensions for intersection.")
    return np.minimum(fig1, fig2)

# ============================
# Count connected components (<255)
# ============================
def count_components(gray_img, connectivity=4):
    if gray_img.dtype != np.uint8:
        temp = gray_img.copy()
        np.clip(temp, 0, 255, out=temp)
        img_u8 = temp.astype(np.uint8)
    else:
        img_u8 = gray_img

    mask = (img_u8 < 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask, connectivity)
    return num_labels - 1

# ============================
# Connectivity metric + curve vs erosion depth
# ============================
def compute_connectivity(gray_img, erosion_step=DISP_CONST,
                         dilation_step=DISP_CONST, max_dilations=1000):

    if gray_img.dtype != np.float32:
        g_curr = gray_img.astype(np.float32)
    else:
        g_curr = gray_img.copy()

    original = g_curr.copy()

    contributions = []   # per-erosion contributions
    depths        = []   # erosion depth index (1,2,...)

    erosion_count = 0

    while True:
        # ---- 1 STEP OF EROSION ----
        g_curr = propagate_erosion(g_curr, erosion_step)

        # Check vanish (all white)
        if not (g_curr < 255).any():
            break

        erosion_count += 1
        depths.append(erosion_count)

        # Check connectivity after erosion
        comps = count_components(g_curr)
        contribution = 0.0

        if comps == 1:
            # Connected → weight = e^0 = 1
            contribution = 1.0
        else:
            # Disconnected → attempt dilation + intersection
            dilated = g_curr.copy()
            prev_inter = None
            num_dil = 0

            for _ in range(max_dilations):
                num_dil += 1

                # One dilation step
                dilated = propagate_dilation_hardskin(dilated, dilation_step)

                # Intersection with original
                inter = intersection(dilated, original)

                # If intersection is empty → cannot reconnect
                if not (inter < 255).any():
                    break

                # Prevent infinite loop if no change
                if prev_inter is not None and np.array_equal(inter, prev_inter):
                    break
                prev_inter = inter

                # Check connectivity
                comps_inter = count_components(inter)
                if comps_inter == 1:
                    contribution = float(np.exp(-0.5 * num_dil))
                    break

            # if never reconnected, contribution stays 0

        contributions.append(contribution)

    if erosion_count == 0:
        return 0.0, np.array([]), np.array([])

    contributions = np.array(contributions, dtype=np.float32)
    depths = np.array(depths, dtype=int)

    # cumulative average connectivity vs depth
    connectivity_curve = np.cumsum(contributions) / np.arange(1, len(contributions) + 1)
    final_connectivity = connectivity_curve[-1]

    return final_connectivity, depths, connectivity_curve

# ============================
# MAIN: compute connectivity and plot vs erosion depth
# ============================
"""
connectivity_value, depths, conn_curve = compute_connectivity(img0)

print(f"Final connectivity: {connectivity_value:.6f}")

plt.figure(figsize=(6, 4))
plt.plot(depths, conn_curve, marker='o')
plt.xlabel("Erosion depth (number of erosions)")
plt.ylabel("Connectivity (cumulative average)")
plt.title("Connectivity vs Erosion Depth")
plt.grid(True)
plt.tight_layout()
plt.show()
"""