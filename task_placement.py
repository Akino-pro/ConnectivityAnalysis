import ast
import math
import os
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from helper_functions import compute_reliability_by_f_list
from planar3R_FTW_morphological_estimation_ssm import compute_beta_range

# ----------------------- Global config -----------------------
N = 128
joint_reliabilities = [0.5, 0.6, 0.7]
x_range_sample = 5

# Search/robustness knobs
THICKNESS_PX = 1                 # default rasterization thickness
COVERAGE_RETRY_RATIO = 0.85       # retry if best covers < n% of original cells
RETRY_SUPERSAMPLE = 2             # on retry, rotate at 2x resolution then downsample
RETRY_CLOSE = True                # on retry, apply 3x3 morphological close

# Make figure/axes backgrounds white (optional)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

# ========== mapping between continuous [-x,x] and grid indices 0..N-1 ==========
def cont_to_idx(xy_cont, N, xmin=-x_range_sample, xmax=x_range_sample, ymin=-x_range_sample, ymax=x_range_sample):
    xy = np.asarray(xy_cont, np.float32)
    x_idx = (xy[:, 0] - xmin) / (xmax - xmin) * (N - 1)
    y_idx = (xy[:, 1] - ymin) / (ymax - ymin) * (N - 1)
    return np.column_stack([x_idx, y_idx]).astype(np.float32)

def idx_to_cont(x_idx, y_idx, N, xmin=-x_range_sample, xmax=x_range_sample, ymin=-x_range_sample, ymax=x_range_sample):
    x = x_idx / (N - 1) * (xmax - xmin) + xmin
    y = y_idx / (N - 1) * (ymax - ymin) + ymin
    return x, y

# ======================= geometry & kernel utilities ============================
def rotate_points(points_xy, angle_deg, center_xy):
    pts = np.asarray(points_xy, np.float32)
    cx, cy = center_xy
    ang = np.deg2rad(angle_deg)
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]], np.float32)
    return (pts - [cx, cy]) @ R.T + [cx, cy]

def rasterize_path_kernel_with_meta(points_xy_grid, thickness_px=3, pad=4):
    """
    Given path points in GRID pixel coords, rasterize to a tight binary kernel.
    Returns (ker, meta) with:
      - ker: float32 binary (0/1)
      - meta['center']: (cx,cy) center in kernel coords (float)
      - meta['kernel_to_grid_origin']: (ox,oy) such that kernel (0,0) lands at grid (ox,oy)
      - meta['pts_kernel']: the path in kernel coords
    """
    pts = np.asarray(points_xy_grid, np.float32)
    xmin, ymin = np.floor(pts.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(pts.max(axis=0)).astype(int)
    w = int(xmax - xmin + 1 + 2 * pad)
    h = int(ymax - ymin + 1 + 2 * pad)
    shift = np.array([xmin - pad, ymin - pad], np.float32)

    ker_full = np.zeros((h, w), np.uint8)
    cv2.polylines(ker_full, [(pts - shift).astype(np.int32)], False, 255, thickness=thickness_px)

    ys, xs = np.where(ker_full > 0)
    if xs.size == 0:
        raise ValueError("Empty kernel; check path/thickness.")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    ker = (ker_full[y0:y1 + 1, x0:x1 + 1] > 0).astype(np.float32)
    center = (ker.shape[1] / 2.0, ker.shape[0] / 2.0)  # (cx, cy) in kernel coords (float)
    origin = (float(shift[0] + x0), float(shift[1] + y0))  # kernel (0,0) in grid coords
    pts_kernel = (pts - np.array(origin, np.float32))  # path in kernel coords
    return ker, {"center": center, "kernel_to_grid_origin": origin, "pts_kernel": pts_kernel}

def rotate_kernel(ker, angle_deg, supersample=0, close=False):
    """
    Rotate binary kernel with antialiasing, then threshold back to {0,1}.
    If supersample>1, rotate at higher res then downsample (smoother edges).
    """
    ker = ker.astype(np.float32)
    h, w = ker.shape

    if supersample and supersample > 1:
        H, W = h * supersample, w * supersample
        ker_hr = cv2.resize(ker, (W, H), interpolation=cv2.INTER_NEAREST)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
        kr_hr = cv2.warpAffine(
            ker_hr, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        kr = cv2.resize(kr_hr, (w, h), interpolation=cv2.INTER_AREA)
        kr = (kr >= 0.5).astype(np.float32)
    else:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        kr = cv2.warpAffine(
            ker, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        kr = (kr >= 0.5).astype(np.float32)

    if close:
        kr = cv2.morphologyEx(kr.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(np.float32)
    return kr

# ================================ scorers ======================================
def score_sum(F, ker_r):
    """Sum of F under the footprint (binary kernel)."""
    return cv2.filter2D(F.astype(np.float32), -1, ker_r, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)

# ========================== helpers: index extraction ==========================
def mask_to_indices(mask):
    rr, cc = np.where(mask > 0)
    return list(zip(rr.tolist(), cc.tolist()))

def footprint_indices_original(ker, meta, Fshape):
    H, W = Fshape
    gx0, gy0 = meta["kernel_to_grid_origin"]
    r0 = int(round(gy0)); c0 = int(round(gx0))
    h, w = ker.shape

    r_start = max(0, r0); c_start = max(0, c0)
    r_end = min(H, r0 + h); c_end = min(W, c0 + w)

    kr_start = r_start - r0; kc_start = c_start - c0
    kr_end = kr_start + (r_end - r_start); kc_end = kc_start + (c_end - c_start)

    mask = np.zeros(Fshape, np.uint8)
    mask[r_start:r_end, c_start:c_end] = (ker[kr_start:kr_end, kc_start:kc_end] > 0).astype(np.uint8)
    return mask_to_indices(mask)

def footprint_indices_best(Fshape, ker_r, best, meta):
    """
    Place rotated kernel so its integer anchor (h//2,w//2) lands at (best.row, best.col),
    matching cv2.filter2D's convention.
    """
    H, W = Fshape
    h, w = ker_r.shape
    cy_anchor = h // 2
    cx_anchor = w // 2

    top_left_row = best["row"] - cy_anchor
    top_left_col = best["col"] - cx_anchor

    r0 = max(0, top_left_row); c0 = max(0, top_left_col)
    r1 = min(H, top_left_row + h); c1 = min(W, top_left_col + w)

    kr0 = r0 - top_left_row; kc0 = c0 - top_left_col
    kr1 = kr0 + (r1 - r0); kc1 = kc0 + (c1 - c0)

    mask = np.zeros(Fshape, np.uint8)
    mask[r0:r1, c0:c1] = (ker_r[kr0:kr1, kc0:kc1] > 0).astype(np.uint8)
    return mask_to_indices(mask)

# ================================ placement ====================================
def build_angle_list(ker, angles_deg=None):
    if angles_deg is not None:
        return np.asarray(angles_deg, dtype=np.float32)
    # adaptive step based on kernel radius
    ys, xs = np.where(ker > 0)
    cy, cx = ker.shape[0] / 2, ker.shape[1] / 2
    R = np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max() + 1e-6)
    dtheta = np.degrees(np.arcsin(min(1.0, 1.0 / max(1e-6, R))))
    dtheta = float(np.clip(dtheta, 0.5, 3.0))
    return np.arange(-180, 180 + 1e-6, dtheta, dtype=np.float32)

def place_path_on_grid_sum(
    F,
    path_points_cont,
    thickness_px=None,
    angles_deg=None,
    bbox=(-x_range_sample, x_range_sample, -x_range_sample, x_range_sample),
    supersample=0,
    close=False,
):
    """
    Search rotations/translations that maximize TOTAL sum of F under the path footprint.
    Returns best dict and meta; also returns the rotated kernel used at best.
    """
    if thickness_px is None:
        thickness_px = THICKNESS_PX

    N = F.shape[0]
    pts_grid = cont_to_idx(np.asarray(path_points_cont, np.float32), N, *bbox)
    ker, meta = rasterize_path_kernel_with_meta(pts_grid, thickness_px=thickness_px)

    angles = build_angle_list(ker, angles_deg)

    best = {"score": -np.inf, "angle": None, "row": None, "col": None}
    best_ker_r = None
    for a in angles:
        ker_r = rotate_kernel(ker, float(a), supersample=supersample, close=close)
        S = score_sum(F, ker_r)
        r, c = np.unravel_index(np.argmax(S), S.shape)
        s = float(S[r, c])
        if s > best["score"]:
            best.update(score=s, angle=float(a), row=int(r), col=int(c))
            best_ker_r = ker_r
    return best, meta, ker, best_ker_r

def run_search_and_indices(
    F, path_cont, thickness_px=None, angles_deg=None, bbox=(-x_range_sample, x_range_sample, -x_range_sample, x_range_sample),
    supersample=0, close=False
):
    best, meta, ker, best_ker_r = place_path_on_grid_sum(
        F, path_cont, thickness_px=thickness_px, angles_deg=angles_deg, bbox=bbox,
        supersample=supersample, close=close
    )
    orig_indices = footprint_indices_original(ker, meta, F.shape)
    best_indices = footprint_indices_best(F.shape, best_ker_r, best, meta)
    return best, meta, ker, best_ker_r, orig_indices, best_indices

# ========================= reliability / field generation ======================
def compute_all_beta_range():
    """
    If beta_list.txt exists and is valid, load and return it.
    Otherwise compute, save, and return.
    """
    path = "beta_list.txt"

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                content = f.read()
            cached = ast.literal_eval(content)
            if isinstance(cached, list) and len(cached) == N:
                return cached
            print("Cached beta_list.txt found but shape invalid; recomputing...")
        except Exception as e:
            print(f"Failed to load cached beta_list.txt ({e}); recomputing...")

    section_length = x_range_sample / N
    x_values = (np.arange(N) + 0.5) * section_length
    y_values = np.zeros(N)
    points = np.column_stack((x_values, y_values))
    all_reliable_beta_ranges = []
    for i in range(len(points)):
        x, y = points[i]
        beta_ranges, reliable_beta_ranges, F_list = compute_beta_range(x, y)
        all_reliable_beta_ranges.append(reliable_beta_ranges)

    try:
        with open(path, "w") as f:
            f.write(str(all_reliable_beta_ranges))
    except Exception as e:
        print(f"Warning: failed to write {path}: {e}")

    return all_reliable_beta_ranges

def nearest_x_index(x, xmin=0, xmax=x_range_sample):
    dx = (xmax - xmin) / N
    i = int(np.round((x - xmin - dx / 2) / dx))
    return int(np.clip(i, 0, N - 1))

def generate_F(N, xmin=-x_range_sample, xmax=x_range_sample, ymin=-x_range_sample, ymax=x_range_sample):
    print('beta range computation starts')
    all_reliable_beta_ranges = compute_all_beta_range()
    print('beta range computation ends')

    dx = (xmax - xmin) / N
    dy = (ymax - ymin) / N
    xs = xmin + dx / 2 + np.arange(N) * dx
    ys = ymin + dy / 2 + np.arange(N) * dy

    F = np.zeros((N, N), dtype=np.float32)

    for yi, y in enumerate(tqdm(ys, desc="Rows")):
        for xi, x in enumerate(xs):
            dist = math.hypot(x, y)
            theta = math.atan2(y, x)
            nearest_idx = nearest_x_index(dist)
            reliable_beta_ranges = all_reliable_beta_ranges[nearest_idx]

            F_List = []
            for beta_range in reliable_beta_ranges[0]:
                if beta_range[0] <= theta <= beta_range[1]:
                    F_List.append(1)
            for beta_range in reliable_beta_ranges[1]:
                if beta_range[0] <= theta <= beta_range[1]:
                    F_List.append(2)
            for beta_range in reliable_beta_ranges[2]:
                if beta_range[0] <= theta <= beta_range[1]:
                    F_List.append(3)

            F[yi, xi] = compute_reliability_by_f_list(F_List, joint_reliabilities)

    return F

# ============================= example paths ===================================
def path_short():
    x = np.linspace(-0.35, 0.35, 50, dtype=np.float32)
    y = 0.22 * np.sin(2.5 * np.pi * x / 0.7)
    return np.column_stack([x, y])

def path_medium():
    x = np.linspace(-1.1, 1.1, 90, dtype=np.float32)
    y = 0.45 * np.sin(2.0 * np.pi * x / 2.2)
    return np.column_stack([x, y])

def path_long():
    x = np.linspace(-2.5, 2.5, 150, dtype=np.float32)
    y = 0.75 * np.sin(2.5 * np.pi * x / 5.0) + 0.15 * np.sin(7.0 * np.pi * x / 5.0)
    return np.column_stack([x, y])

def _catmull_rom_chain(P, samples_per_seg=40, box=3.0):
    P = np.asarray(P, dtype=np.float32)
    m = len(P)
    if m < 2:
        return P.copy()

    out = []
    for j in range(m - 1):
        p0 = P[j - 1] if j - 1 >= 0 else P[j]
        p1 = P[j]
        p2 = P[j + 1]
        p3 = P[j + 2] if (j + 2) < m else P[j + 1]

        t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False, dtype=np.float32)[:, None]
        t2 = t * t
        t3 = t2 * t
        term = (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        ) * 0.5
        out.append(term)

    out.append(P[-1][None, :])
    curve = np.vstack(out).astype(np.float32)
    curve[:, 0] = np.clip(curve[:, 0], -box, box)
    curve[:, 1] = np.clip(curve[:, 1], -box, box)
    return curve

def _path_length(curve: np.ndarray) -> float:
    if curve.shape[0] < 2:
        return 0.0
    d = np.diff(curve, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

def gen_random_continuous_inputs(n=2, num_ctrl=8, samples_per_seg=36, box=3.0, max_length=7.0, seed=None):
    rng = np.random.default_rng(seed)
    paths = []
    min_possible = 2.0 * box
    target_max = max(max_length, min_possible)

    for _ in range(n):
        amp = 0.35 * box
        for _try in range(12):
            x_ctrl = np.linspace(-box, box, num_ctrl, dtype=np.float32)
            y_ctrl = rng.normal(0.0, amp, size=num_ctrl).astype(np.float32)
            y_ctrl = np.clip(y_ctrl, -0.85 * box, 0.85 * box)
            ctrl = np.column_stack([x_ctrl, y_ctrl]).astype(np.float32)

            path = _catmull_rom_chain(ctrl, samples_per_seg=samples_per_seg, box=box)
            if _path_length(path) <= target_max + 1e-6:
                break
            amp *= 0.65

        paths.append(path)

    return paths

# ================================ plotting =====================================
def plot_indices(ax, indices, N, bbox, label, marker, ms, mfc, mec, z):
    if not indices:
        return
    rows = np.array([r for r, _ in indices], dtype=np.float32)
    cols = np.array([c for _, c in indices], dtype=np.float32)
    x_cont, y_cont = idx_to_cont(cols, rows, N, *bbox)
    ax.plot(x_cont, y_cont, marker=marker, linestyle="None",
            markersize=ms, markerfacecolor=mfc, markeredgecolor=mec,
            label=label, zorder=z)

# =================================== demo ======================================
if __name__ == "__main__":
    start = time.perf_counter()
    F = generate_F(N, xmin=-x_range_sample, xmax=x_range_sample, ymin=-x_range_sample, ymax=x_range_sample)
    end = time.perf_counter()
    print(f"reliability map computation took {end - start:.6f} seconds")

    paths = {
        "Short":  path_short(),
        "Medium": path_medium(),
        "Long":   path_long(),
    }

    rand_paths = gen_random_continuous_inputs(n=3, num_ctrl=7, samples_per_seg=32, box=3.0, max_length=7.0, seed=None)
    for i, p in enumerate(rand_paths, start=1):
        paths[f"Random{i}"] = p

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    bbox = (-x_range_sample, x_range_sample, -x_range_sample, x_range_sample)
    im = None

    for ax, (label, path_cont) in zip(axes, paths.items()):
        # ---- first search
        t0 = time.perf_counter()
        best, meta, ker, best_ker_r, orig_indices, best_indices = run_search_and_indices(
            F, path_cont, thickness_px=THICKNESS_PX, angles_deg=None, bbox=bbox, supersample=0, close=False
        )
        t1 = time.perf_counter()

        # ---- coverage sanity check & retry
        need_retry = (len(orig_indices) > 0 and len(best_indices) < COVERAGE_RETRY_RATIO * len(orig_indices))
        if need_retry:
            print(f"[{label}] coverage drop: best {len(best_indices)} vs original {len(orig_indices)} "
                  f"(< {COVERAGE_RETRY_RATIO*100:.0f}%). Retrying with supersample={RETRY_SUPERSAMPLE}, close={RETRY_CLOSE}...")
            best, meta, ker, best_ker_r, orig_indices, best_indices = run_search_and_indices(
                F, path_cont, thickness_px=THICKNESS_PX, angles_deg=None, bbox=bbox,
                supersample=RETRY_SUPERSAMPLE, close=RETRY_CLOSE
            )

        print(f"Search time [{label}]: {t1 - t0:.6f}s" + (" + retry" if need_retry else ""))

        # ---- print the best indices
        best_indices_arr = np.array(best_indices, dtype=int)
        print(f"[{label}] BEST covered indices (row, col) — {len(best_indices_arr)} cells:")
        #print(best_indices_arr)

        # ---- background field
        im = ax.imshow(
            F, origin="lower", cmap="rainbow", vmin=0.0, vmax=1.0,
            extent=[bbox[0], bbox[1], bbox[2], bbox[3]], zorder=0
        )

        # ---- plot both sets of grid cells (cell centers)
        plot_indices(ax, orig_indices, N, bbox, "original covered", marker="s",
                     ms=2.6, mfc="white", mec="black", z=10)
        plot_indices(ax, best_indices, N, bbox, "best covered", marker="o",
                     ms=2.6, mfc="yellow", mec="black", z=11)

        # ---- reporting
        print(f"[{label}] angle={best['angle']:.2f}°, sum={best['score']:.4f}, "
              f"original |cells|={len(orig_indices)}, best |cells|={len(best_indices)}")

        ax.set_title(f"{label}\nangle {best['angle']:.2f}°, sum {best['score']:.3f}")
        ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
        ax.set_aspect("equal"); ax.set_xlabel("x", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    axes[0].set_ylabel("y", fontsize=11)

    # one shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.02)
    cbar.set_label("Value (0–1)", fontsize=11)

    plt.show()
