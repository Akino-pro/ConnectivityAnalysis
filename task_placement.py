import ast
import math
import os
import time
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from helper_functions import compute_reliability_by_f_list
from planar3R_FTW_morphological_estimation_ssm import compute_beta_range

# max min max% min% angle pivot avg sum size
# 10 times physical exp for 2 trj
# ----------------------- Global config -----------------------
N = 256
#joint_reliabilities = [0.5,0.6,0.7]
#x_range_sample=3
joint_reliabilities = [2.0/3.0,2.0/3.0, 2.0/3.0]
x_range_sample = np.sum([0.4454,0.3143,0.2553])
n_angles=1440
N_MAX = 3

# Search/robustness knobs
THICKNESS_PX = 1               # default rasterization thickness
COVERAGE_RETRY_RATIO = 0.90       # retry if best covers < n% of original cells
RETRY_SUPERSAMPLE = 2             # on retry, rotate at 2x resolution then downsample
RETRY_CLOSE = True                # on retry, apply 3x3 morphological close


# Make figure/axes backgrounds white (optional)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

# ========== mapping between continuous [-x,x] and grid indices 0..N-1 ==========
def pick_optimal_subset(ties, n_max=N_MAX, seed=None):
    """
    From a list of tied optima, return:
      - n_max random distinct entries if len(ties) >= n_max
      - otherwise exactly ONE entry (ties[0]) when len(ties) < n_max
    """
    if not ties:
        return []

    if seed is not None:
        random.seed(seed)

    if len(ties) >= n_max:
        return random.sample(ties, n_max)
    else:
        return [ties[0]]
def shift_to_top_left(path_xy: np.ndarray, box: float):
    """
    Shift a polyline so that xmin -> -box and ymax -> +box.
    Returns: shifted_path, (dx, dy) where new = old - (dx, dy).
    """
    path_xy = np.asarray(path_xy, np.float32)
    xmin, ymin = path_xy.min(axis=0)
    xmax, ymax = path_xy.max(axis=0)

    # Amount we subtract from x to push xmin to -box
    dx = xmin + box
    # Amount we subtract from y to push ymax to +box
    # (i.e., subtract a negative -> add upward)
    dy = -(box - ymax)

    shifted = np.column_stack([path_xy[:, 0] - dx, path_xy[:, 1] - dy]).astype(np.float32)
    return shifted, (dx, dy)

def shift_point(x: float, y: float, dx: float, dy: float):
    """Apply the same shift used for the path to a single point."""
    return float(x - dx), float(y - dy)

def cont_to_idx(xy_cont, N, xmin=-x_range_sample, xmax=x_range_sample, ymin=-x_range_sample, ymax=x_range_sample):
    xy = np.asarray(xy_cont, np.float32)
    x_idx = (xy[:, 0] - xmin) / (xmax - xmin) * (N - 1)
    y_idx = (xy[:, 1] - ymin) / (ymax - ymin) * (N - 1)
    return np.column_stack([x_idx, y_idx]).astype(np.float32)

def idx_to_cont(x_idx, y_idx, N, xmin=-x_range_sample, xmax=x_range_sample, ymin=-x_range_sample, ymax=x_range_sample):
    x = x_idx / (N - 1) * (xmax - xmin) + xmin
    y = y_idx / (N - 1) * (ymax - ymin) + ymin
    return x, y

def clip_to_circle(path, radius=3.0):
    mask = np.hypot(path[:,0], path[:,1]) <= radius
    return path[mask]

def _triangle_wave(x, period, amplitude, phase=0.0):
    # Triangle in [-1,1] via modulo arithmetic, then scale by amplitude
    t = ((x - phase) / period) % 1.0
    tri = 4.0 * np.abs(t - 0.5) - 1.0
    return amplitude * tri

def _square_wave(x, period, amplitude, duty=0.5, phase=0.0):
    # Perfect rectangular levels in {-amplitude, +amplitude}
    t = ((x - phase) / period) % 1.0
    return np.where(t < duty, amplitude, -amplitude).astype(np.float32)

def _insert_verticals(x, y):
    """
    Densify points at each step so cv2.polylines draws true vertical walls,
    not diagonals between two far-apart samples.
    """
    X = [x[0]]; Y = [y[0]]
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        if y0 == y1:
            X.append(x1); Y.append(y1)          # horizontal continuation
        else:
            xm = 0.5 * (x0 + x1)                # insert a vertical wall at xm
            X.extend([xm, xm, x1])
            Y.extend([y0, y1, y1])
    return np.column_stack([np.asarray(X, np.float32), np.asarray(Y, np.float32)])

def _sample_edge(p0, p1, n):
    """n points from p0→p1 (excluding p1)."""
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)[:, None]
    p0 = np.asarray(p0, np.float32); p1 = np.asarray(p1, np.float32)
    return p0 + (p1 - p0) * t

def polyline_rectangle(center, w, h, points_per_edge=120):
    """Axis-aligned rectangle outline as a closed polyline."""
    cx, cy = center
    hw, hh = 0.5 * w, 0.5 * h
    # corners (clockwise)
    v0 = (cx - hw, cy - hh)
    v1 = (cx + hw, cy - hh)
    v2 = (cx + hw, cy + hh)
    v3 = (cx - hw, cy + hh)

    e01 = _sample_edge(v0, v1, points_per_edge)
    e12 = _sample_edge(v1, v2, points_per_edge)
    e23 = _sample_edge(v2, v3, points_per_edge)
    e30 = _sample_edge(v3, v0, points_per_edge)

    poly = np.vstack([e01, e12, e23, e30, np.asarray(v0, np.float32)[None, :]])  # close
    return poly.astype(np.float32)

def polyline_equilateral_triangle(center, side, rot_deg=0.0, points_per_edge=150):
    """Equilateral triangle outline as a closed polyline."""
    cx, cy = center
    R = side / np.sqrt(3)  # circumradius
    ang0 = np.deg2rad(rot_deg)
    a0 = ang0
    a1 = ang0 + 2*np.pi/3
    a2 = ang0 + 4*np.pi/3
    v0 = (cx + R*np.cos(a0), cy + R*np.sin(a0))
    v1 = (cx + R*np.cos(a1), cy + R*np.sin(a1))
    v2 = (cx + R*np.cos(a2), cy + R*np.sin(a2))

    e01 = _sample_edge(v0, v1, points_per_edge)
    e12 = _sample_edge(v1, v2, points_per_edge)
    e20 = _sample_edge(v2, v0, points_per_edge)

    poly = np.vstack([e01, e12, e20, np.asarray(v0, np.float32)[None, :]])  # close
    return poly.astype(np.float32)

def best_pivot_xy(best, N, bbox):
    x, y = idx_to_cont(
        np.array([best["col"]], np.float32),
        np.array([best["row"]], np.float32),
        N, *bbox
    )
    return float(x[0]), float(y[0])

def input_pivot_xy(meta, N, bbox):
    # kernel center in kernel coords
    cx, cy = meta["center"]
    ox, oy = meta["kernel_to_grid_origin"]
    # pivot in grid coordinates
    gx = ox + cx
    gy = oy + cy
    # convert to continuous coordinates
    x, y = idx_to_cont(np.array([gx], np.float32),
                       np.array([gy], np.float32),
                       N, *bbox)
    return float(x[0]), float(y[0])
# ======================= geometry & kernel utilities ============================
def rotate_points(points_xy, angle_deg, center_xy):
    pts = np.asarray(points_xy, np.float32)
    cx, cy = center_xy
    ang = np.deg2rad(angle_deg)
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]], np.float32)
    return (pts - [cx, cy]) @ R.T + [cx, cy]

def rasterize_path_kernel_with_meta(points_xy_grid, thickness_px=3, pad=4, fill_closed=False):
    """
    Given path points in GRID pixel coords, rasterize to a tight binary kernel.
    If fill_closed=True and the path is closed, uses cv2.fillPoly instead of polylines.
    """
    pts = np.asarray(points_xy_grid, np.float32)
    xmin, ymin = np.floor(pts.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(pts.max(axis=0)).astype(int)
    w = int(xmax - xmin + 1 + 2 * pad)
    h = int(ymax - ymin + 1 + 2 * pad)
    shift = np.array([xmin - pad, ymin - pad], np.float32)

    ker_full = np.zeros((h, w), np.uint8)
    pts_i = (pts - shift).astype(np.int32)

    is_closed_path = (pts.shape[0] >= 2) and np.allclose(pts[0], pts[-1])
    if fill_closed and is_closed_path:
        cv2.fillPoly(ker_full, [pts_i], 255)
    else:
        cv2.polylines(ker_full, [pts_i], False, 255, thickness=thickness_px)

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
def plot_two_manual_solutions(
    F,
    path_cont,
    case_optimal,         # (angle_deg, (px, py))  -> white
    case_nonoptimal,      # (angle_deg, (px, py))  -> black
    thickness_px=THICKNESS_PX,
    bbox=(-x_range_sample, x_range_sample, -x_range_sample, x_range_sample),
    ax=None,
):
    """
    Plot two manual placements on a single figure:
      - Optimal task placement (white)
      - Non-optimal task placement (black)
    Also prints their scores.
    """
    N = F.shape[0]
    xmin, xmax, ymin, ymax = bbox

    def cont_to_idx_points(pts_cont):
        return cont_to_idx(pts_cont, N, *bbox)

    def place_kernel_at_pivot_mask(Fshape, ker_r, pivot_xy_cont):
        H, W = Fshape
        h, w = ker_r.shape
        cy_anchor = h // 2
        cx_anchor = w // 2
        x_idx, y_idx = cont_to_idx(
            np.array([[pivot_xy_cont[0], pivot_xy_cont[1]]], dtype=np.float32),
            N, *bbox
        )[0]
        r = int(round(y_idx))
        c = int(round(x_idx))
        top_left_row = r - cy_anchor
        top_left_col = c - cx_anchor
        r0 = max(0, top_left_row)
        c0 = max(0, top_left_col)
        r1 = min(H, top_left_row + h)
        c1 = min(W, top_left_col + w)
        kr0 = r0 - top_left_row
        kc0 = c0 - top_left_col
        kr1 = kr0 + (r1 - r0)
        kc1 = kc0 + (c1 - c0)
        mask = np.zeros(Fshape, np.uint8)
        mask[r0:r1, c0:c1] = (ker_r[kr0:kr1, kc0:kc1] > 0).astype(np.uint8)
        return mask

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        created_fig = True

    # --- background reliability field ---
    ax.imshow(
        F, origin="lower", cmap="rainbow", vmin=0.0, vmax=1.0,
        extent=[xmin, xmax, ymin, ymax], zorder=0
    )

    path_arr = np.asarray(path_cont, np.float32)

    def _draw_one(angle_deg, pivot_xy, color, name):
        pc = path_arr.mean(axis=0)
        rot_pts = rotate_points(path_arr, float(angle_deg), center_xy=(float(pc[0]), float(pc[1])))
        pts_grid_rot = cont_to_idx_points(rot_pts)

        ker_rot, _ = rasterize_path_kernel_with_meta(
            pts_grid_rot, thickness_px=thickness_px, pad=8, fill_closed=False
        )

        mask = place_kernel_at_pivot_mask(F.shape, ker_rot, pivot_xy)

        # ---- SCORE (sum + mean) ----
        vals = F[mask > 0]
        score_sum = float(vals.sum()) if vals.size else 0.0
        score_mean = float(vals.mean()) if vals.size else float("nan")
        n_cells = int(vals.size)

        print(
            f"[{name}] angle={float(angle_deg):.4f} deg, pivot=({float(pivot_xy[0]):.6f}, {float(pivot_xy[1]):.6f}), "
            f"cells={n_cells}, sum={score_sum:.6f}, mean={score_mean:.6f}"
        )

        # ---- DRAW ----
        draw_mask_boundary(ax, mask, N, bbox, lw=2.2, color=color, z=13)
        px, py = float(pivot_xy[0]), float(pivot_xy[1])
        ax.plot(
            px, py, "o", markersize=7,
            color=color, mec="black" if color == "white" else "white",
            mew=1.0, zorder=14
        )

        return {"sum": score_sum, "mean": score_mean, "cells": n_cells}

    # --- draw both placements + print scores ---
    s1 = _draw_one(case_optimal[0], case_optimal[1], "white", "Manual case 1 (white)")
    s2 = _draw_one(case_nonoptimal[0], case_nonoptimal[1], "black", "Manual case 2 (black)")

    # --- styling ---
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=22)
    ax.set_ylabel("y", fontsize=22)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # --- legend (line + dot) ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="white", marker="o", markerfacecolor="white",
               markeredgecolor="black", linewidth=2.2, markersize=7,
               label="Optimal task placement"),
        Line2D([0], [0], color="black", marker="o", markerfacecolor="black",
               markeredgecolor="white", linewidth=2.2, markersize=7,
               label="Non-optimal task placement"),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.05, 0.5),
        fontsize=18,
        frameon=True,
        facecolor="white"
    )
    legend.get_frame().set_alpha(0.9)

    if created_fig:
        plt.show()

    # return scores too (handy if you want to log/save)
    return {"case1": s1, "case2": s2}
        
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

def indices_to_mask(indices, shape):
    m = np.zeros(shape, np.uint8)
    if indices:
        rr, cc = np.array(indices, int).T
        m[rr, cc] = 1
    return m

def draw_mask_boundary(ax, mask, N, bbox, lw=2.0, color="white", z=12):
    """
    Extract a contour from a binary mask and plot it as a continuous outline.
    """
    if mask.max() == 0:
        return
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        pts = cnt.squeeze(1)          # shape (K,2), columns are (x, y) = (col, row)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        cols = pts[:, 0].astype(np.float32)
        rows = pts[:, 1].astype(np.float32)
        x_cont, y_cont = idx_to_cont(cols, rows, N, *bbox)
        ax.plot(x_cont, y_cont, "-", linewidth=lw, color=color, zorder=z)

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
    supersample=0,   # not used now, kept for compatibility
    close=False      # not used now, kept for compatibility
):
    """
    Rotate the *continuous path* for each angle, rasterize fresh (outline),
    and score by SUM under the footprint (as before).
    """
    if thickness_px is None:
        thickness_px = THICKNESS_PX

    N = F.shape[0]
    path_cont = np.asarray(path_points_cont, np.float32)

    # Unrotated outline kernel (for orig_indices)
    pts_grid0 = cont_to_idx(path_cont, N, *bbox)
    ker0, meta0 = rasterize_path_kernel_with_meta(
        pts_grid0, thickness_px=thickness_px, pad=8, fill_closed=False
    )

    # angles
    if angles_deg is None:
        angles = build_angle_list(ker0, None)
    else:
        angles = np.asarray(angles_deg, np.float32)

    pc = path_cont.mean(axis=0)
    best = {"score": -np.inf, "angle": None, "row": None, "col": None}
    best_ker_r = None

    for a in angles:
        rot_pts = rotate_points(path_cont, float(a), center_xy=pc)
        pts_grid = cont_to_idx(rot_pts, N, *bbox)

        ker_r, _ = rasterize_path_kernel_with_meta(
            pts_grid, thickness_px=thickness_px, pad=8, fill_closed=False
        )

        S = score_sum(F, ker_r)  # <-- back to SUM (old behavior)
        r, c = np.unravel_index(np.argmax(S), S.shape)
        s = float(S[r, c])

        if (s > best["score"]) or (s == best["score"] and ker_r.sum() > (best_ker_r.sum() if best_ker_r is not None else -1)):
            best.update(score=s, angle=float(a), row=int(r), col=int(c))
            best_ker_r = ker_r

    return best, meta0, ker0, best_ker_r
def plot_manual_solution(
    F,
    path_cont,
    angle_deg,
    pivot_xy,
    thickness_px=THICKNESS_PX,
    bbox=(-x_range_sample, x_range_sample, -x_range_sample, x_range_sample),
    ax=None,
):
    """
    Plot the standard figure using a user-specified angle (deg) and pivot (continuous x,y),
    with the given input path.

    - F:      (N,N) field
    - path_cont: Nx2 float32 continuous coords (same units as bbox)
    - angle_deg: float, rotation angle to apply to the input path before placement
    - pivot_xy: (px, py) in continuous coords where the rotated footprint will be anchored
    - thickness_px: rasterization thickness
    - bbox:  (xmin, xmax, ymin, ymax) continuous extents
    - ax:    optional Matplotlib Axes; if None, creates its own figure/axes
    """
    import numpy as np
    import matplotlib.pyplot as plt

    N = F.shape[0]
    xmin, xmax, ymin, ymax = bbox

    # ---------- helpers ----------
    def cont_to_idx_points(pts_cont):
        return cont_to_idx(pts_cont, N, *bbox)

    def rotate_points_local(points_xy, angle_deg, center_xy):
        return rotate_points(points_xy, angle_deg, center_xy)

    def place_kernel_at_pivot_mask(Fshape, ker_r, pivot_xy_cont):
        """
        Place a rotated kernel so that its integer anchor (h//2,w//2)
        lands at the grid index corresponding to pivot_xy_cont.
        Returns a binary mask (H,W) with the footprint.
        """
        H, W = Fshape
        h, w = ker_r.shape
        cy_anchor = h // 2
        cx_anchor = w // 2

        # convert pivot continuous -> grid index (col=x, row=y)
        x_idx, y_idx = cont_to_idx(
            np.array([[pivot_xy_cont[0], pivot_xy_cont[1]]], dtype=np.float32),
            N, *bbox
        )[0]
        r = int(round(y_idx))
        c = int(round(x_idx))

        top_left_row = r - cy_anchor
        top_left_col = c - cx_anchor

        r0 = max(0, top_left_row); c0 = max(0, top_left_col)
        r1 = min(H, top_left_row + h); c1 = min(W, top_left_col + w)

        kr0 = r0 - top_left_row; kc0 = c0 - top_left_col
        kr1 = kr0 + (r1 - r0);     kc1 = kc0 + (c1 - c0)

        mask = np.zeros(Fshape, np.uint8)
        mask[r0:r1, c0:c1] = (ker_r[kr0:kr1, kc0:kc1] > 0).astype(np.uint8)
        return mask

    # ---------- build the rotated "solution" kernel ----------
    path_arr = np.asarray(path_cont, np.float32)
    pc = path_arr.mean(axis=0)  # rotate about its own centroid (matches your search)
    rot_pts = rotate_points_local(path_arr, float(angle_deg), center_xy=(float(pc[0]), float(pc[1])))
    pts_grid_rot = cont_to_idx_points(rot_pts)

    ker_rot, _ = rasterize_path_kernel_with_meta(
        pts_grid_rot, thickness_px=thickness_px, pad=8, fill_closed=False
    )

    # ---------- make solution footprint mask at the requested pivot ----------
    px, py = float(pivot_xy[0]), float(pivot_xy[1])
    best_mask = place_kernel_at_pivot_mask(F.shape, ker_rot, (px, py))

    # ---------- build the "shifted input" footprint (top-left) and its pivot ----------
    # - use same shift_to_top_left convention (with a tight box)
    shifted_path, (dx, dy) = shift_to_top_left(path_arr, x_range_sample - x_range_sample / 30.0)
    ix, iy = path_arr.mean(axis=0)  # input pivot = kernel center in unshifted coords
    ix_shift, iy_shift = shift_point(ix, iy, dx, dy)

    pts_grid_shifted = cont_to_idx_points(shifted_path)
    ker_shift, meta_shift = rasterize_path_kernel_with_meta(
        pts_grid_shifted, thickness_px=thickness_px
    )
    # indices → mask for outline plotting
    shifted_input_indices = footprint_indices_original(ker_shift, meta_shift, F.shape)
    shifted_mask = indices_to_mask(shifted_input_indices, F.shape)

    # ---------- plotting ----------
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        created_fig = True

    # background field
    im = ax.imshow(
        F, origin="lower", cmap="rainbow", vmin=0.0, vmax=1.0,
        extent=[xmin, xmax, ymin, ymax], zorder=0
    )

    # outlines
    draw_mask_boundary(ax, shifted_mask, N, bbox, lw=1.8, color="white", z=12)
    draw_mask_boundary(ax, best_mask,    N, bbox, lw=2.2, color="black", z=13)

    # pivots
    ax.plot(ix_shift, iy_shift, marker="o", markersize=7,
            color="white", mec="black", mew=1.0, zorder=12, label="input pivot (shifted)")
    ax.plot(px, py, marker="o", markersize=7,
            color="black", mec="white", mew=1.0, zorder=13, label="solution pivot")

    # axes & labels
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=22)
    ax.set_ylabel("y", fontsize=22)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    # ax.set_title(f"angle {angle_deg:.2f}°, pivot=({px:.3f},{py:.3f})")

    if created_fig:
        plt.show()

    # return some useful artifacts if caller wants stats
    return {
        "solution_mask": best_mask,
        "shifted_mask": shifted_mask,
        "solution_pivot": (px, py),
        "input_pivot_shifted": (ix_shift, iy_shift),
        "angle_deg": float(angle_deg),
    }
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
    print(len(points))
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
            """
            if dist > 3.0:
                continue
            """
            theta = math.atan2(y, x)
            nearest_idx = nearest_x_index(dist)
            reliable_beta_ranges = all_reliable_beta_ranges[nearest_idx]
            #print(len(reliable_beta_ranges))

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
            for beta_range in reliable_beta_ranges[-1]:
                if beta_range[0] <= theta <= beta_range[1]:
                    F_List.append(0)

            F[yi, xi] = compute_reliability_by_f_list(F_List, joint_reliabilities)

    return F

def score_mean(F, ker_r):
    """
    Mean of F under the footprint (normalizes out coverage differences).
    """
    F32 = F.astype(np.float32)
    S_sum = cv2.filter2D(F32, -1, ker_r, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)
    S_cnt = cv2.filter2D(np.ones_like(F32, np.float32), -1, ker_r, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)
    return S_sum / np.maximum(S_cnt, 1e-6)

# ============================= example paths ===================================
def path_half_moon(center=(0.0, 0.0), R=1.6, r=1.0, dx=0.6,
                   n_outer=240, n_inner=240):
    """
    Crescent (half-moon) outline as a closed polyline.
    - 'center' is the center of the OUTER circle.
    - INNER circle center is shifted by (dx, 0) from 'center'.
    - Requires the two circles to intersect (|R-r| < dx < R+r).
    Returns Nx2 float32 array, first point == last point.
    """
    cx, cy = center
    c0 = np.array([cx, cy], np.float32)
    c1 = np.array([cx + dx, cy], np.float32)

    d = float(np.linalg.norm(c1 - c0))
    if not (abs(R - r) < d < (R + r)):
        # Fallback: half-annulus (upper half of outer, back on inner)
        theta0, theta1 = 0.0, np.pi
        outer = np.column_stack([
            cx + R * np.cos(np.linspace(theta0, theta1, n_outer, endpoint=False, dtype=np.float32)),
            cy + R * np.sin(np.linspace(theta0, theta1, n_outer, endpoint=False, dtype=np.float32)),
        ])
        inner = np.column_stack([
            cx + dx + r * np.cos(np.linspace(theta1, theta0, n_inner, endpoint=False, dtype=np.float32)),
            cy + r * np.sin(np.linspace(theta1, theta0, n_inner, endpoint=False, dtype=np.float32)),
        ])
        poly = np.vstack([outer, inner, outer[:1]])
        return poly.astype(np.float32)

    # Circle–circle intersection points (on the line from c0 to c1)
    ex = (c1 - c0) / d
    a = (R*R - r*r + d*d) / (2*d)
    h_sq = R*R - a*a
    h = math.sqrt(max(h_sq, 0.0))
    p2 = c0 + a * ex
    # Perpendicular to ex
    perp = np.array([-ex[1], ex[0]], np.float32)
    i1 = p2 + h * perp
    i2 = p2 - h * perp

    # Angles on outer/inner circles for the two intersection points
    th1, th2 = math.atan2(i1[1]-cy, i1[0]-cx), math.atan2(i2[1]-cy, i2[0]-cx)
    ph1 = math.atan2(i1[1]-(cy), i1[0]-(cx+dx))
    ph2 = math.atan2(i2[1]-(cy), i2[0]-(cx+dx))

    # Generate CCW outer arc from th1 -> th2 (short way), then
    # CW inner arc from ph2 -> ph1 to close the crescent boundary
    def arc_ccw(a0, a1, n):
        # unwrap to ensure CCW
        a0u, a1u = np.unwrap([a0, a1])
        if a1u < a0u:
            a1u += 2*np.pi
        return np.linspace(a0u, a1u, n, endpoint=False, dtype=np.float32)

    outer_t = arc_ccw(th1, th2, n_outer)
    inner_t = arc_ccw(ph1, ph2, n_inner)[::-1]  # reverse for CW

    outer = np.column_stack([cx + R*np.cos(outer_t), cy + R*np.sin(outer_t)])
    inner = np.column_stack([cx + dx + r*np.cos(inner_t), cy + r*np.sin(inner_t)])

    poly = np.vstack([outer, inner, outer[:1]])
    return poly.astype(np.float32)

def path_short():
    x = np.linspace(-1.5, 1.5, 500, dtype=np.float32)
    y = 0.5 * np.sin(2.5 * np.pi * x / 0.7)
    return clip_to_circle(np.column_stack([x, y]))
    #return np.column_stack([x, y])

def path_medium():
    # Closed EQUILATERAL TRIANGLE, roughly centered; sized to fit inside r=3
    tri = polyline_equilateral_triangle(center=(0.0, -0.2),
                                        side=2.0,        # adjust size as you like
                                        rot_deg=0.0,
                                        points_per_edge=160)
    return clip_to_circle(tri)  # keeps it inside your radius=3 window


def path_long():
    # Closed AXIS-ALIGNED RECTANGLE
    rect = polyline_rectangle(center=(0.8, -0.2),
                              w=2.8, h=0.6,   # width/height; tweak to taste
                              points_per_edge=260)
    return clip_to_circle(rect)

def path_line():
    # Horizontal straight line from -2 to +2
    #y = np.linspace(-1.7, 1.7, 300, dtype=np.float32)
    #x = np.zeros_like(y)
    y = np.linspace(-0.20, 0.20, 100, dtype=np.float32)
    x = np.zeros_like(y)
    return np.column_stack([x, y])

def path_circle(num_pts=300, radius=1.2):
    # Circle centered at origin
    theta = np.linspace(0, 2*np.pi, num_pts, endpoint=True, dtype=np.float32)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])

def path_square(side=1, pts_per_edge=50):
    # Axis-aligned closed square centered at origin
    half = side / 2.0
    # define corners (clockwise)
    v0 = (-half, -half)
    v1 = ( half, -half)
    v2 = ( half,  half)
    v3 = (-half,  half)

    # sample each edge
    x0 = np.linspace(v0[0], v1[0], pts_per_edge, endpoint=False)
    y0 = np.full_like(x0, v0[1])

    x1 = np.full(pts_per_edge, v1[0])
    y1 = np.linspace(v1[1], v2[1], pts_per_edge, endpoint=False)

    x2 = np.linspace(v2[0], v3[0], pts_per_edge, endpoint=False)
    y2 = np.full_like(x2, v2[1])

    x3 = np.full(pts_per_edge, v3[0])
    y3 = np.linspace(v3[1], v0[1], pts_per_edge, endpoint=False)

    xs = np.concatenate([x0, x1, x2, x3, [v0[0]]])
    ys = np.concatenate([y0, y1, y2, y3, [v0[1]]])
    return np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])

def _catmull_rom_chain(P, samples_per_seg=40, box=x_range_sample):
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

def gen_random_continuous_inputs(n=2, num_ctrl=8, samples_per_seg=36, box=x_range_sample, max_length=7, seed=None):
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

        path = clip_to_circle(path, radius=3.0)
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
        #"Sin curve":  path_short(),
        #"Triangle": path_medium(),
        #"Rectangle":   path_long(),
        "Line": path_line(),
        #"Half-Moon": clip_to_circle(path_half_moon(center=(1.1, -1.0), R=1.1, r=0.6, dx=0.7)),
        #"Circle": path_circle(),
        #"Square": path_square(),
        #"Rectangle":   path_long(),
    }

    rand_paths = gen_random_continuous_inputs(n=0, num_ctrl=7, samples_per_seg=32, box=1, max_length=4, seed=None)
    for i, p in enumerate(rand_paths, start=1):
        paths[f"Random curve"] = p

    """

    fig, axes = plt.subplots(3, 2, figsize=(9, 14), constrained_layout=True)
    axes = axes.ravel()
    bbox = (-x_range_sample, x_range_sample, -x_range_sample, x_range_sample)
    im = None
    angles = np.linspace(-180, 180, n_angles, dtype=np.float32)
    for ax, (label, path_cont) in zip(axes, paths.items()):
        t0 = time.perf_counter()
        best, meta, ker, best_ker_r, orig_indices, best_indices = run_search_and_indices(
            F, path_cont, thickness_px=THICKNESS_PX, angles_deg=angles , bbox=bbox, supersample=0, close=False
        )
        t1 = time.perf_counter()

        need_retry = (len(orig_indices) > 0 and len(best_indices) < COVERAGE_RETRY_RATIO * len(orig_indices))
        if need_retry:
            best, meta, ker, best_ker_r, orig_indices, best_indices = run_search_and_indices(
                F, path_cont, thickness_px=THICKNESS_PX, angles_deg=angles, bbox=bbox,
                supersample=RETRY_SUPERSAMPLE, close=RETRY_CLOSE
            )
        print(f"Search time [{label}]: {t1 - t0:.6f}s" + (" + retry" if need_retry else ""))



        # ---- compute pivots (solution in global coords; input in original coords)
        px, py = best_pivot_xy(best, N, bbox)  # solution pivot (keep global)
        ix, iy = input_pivot_xy(meta, N, bbox)  # input pivot (original)

        # ---- compute the shifted input (to top-left of the global window)
        path_arr = np.asarray(path_cont, np.float32)
        shifted_path, (dx, dy) = shift_to_top_left(path_arr, 2.9)
        ix_shift, iy_shift = shift_point(ix, iy, dx, dy)

        pts_grid_shifted = cont_to_idx(shifted_path, N, *bbox)
        ker_shift, meta_shift = rasterize_path_kernel_with_meta(
            pts_grid_shifted, thickness_px=THICKNESS_PX
        )
        shifted_input_indices = footprint_indices_original(ker_shift, meta_shift, F.shape)

        best_sum = float(best["score"])
        n_cells_best = len(best_indices)
        best_avg = (best_sum / n_cells_best) if n_cells_best > 0 else float("nan")

        # Shifted-input footprint (sum the field values at those cells)
        if shifted_input_indices:
            rr_s, cc_s = np.array(shifted_input_indices, dtype=int).T
            sum_shifted = float(F[rr_s, cc_s].sum())
            n_cells_shifted = len(shifted_input_indices)
            avg_shifted = sum_shifted / n_cells_shifted
        else:
            sum_shifted, n_cells_shifted, avg_shifted = 0.0, 0, float("nan")

        # Print a concise report line
        print(
            f"[{label}] cells: input_shifted={n_cells_shifted}, solution={n_cells_best}; "
            f"avg_shifted={avg_shifted:.4f}, avg_solution={best_avg:.4f}"
        )

        # ---- gather stats on the solution footprint
        if best_indices:
            rr_b, cc_b = np.array(best_indices, dtype=int).T
            vals_b = F[rr_b, cc_b].astype(np.float32)
            solution_sum = float(vals_b.sum())
            n_cells_best = int(vals_b.size)
            solution_avg = float(solution_sum / max(n_cells_best, 1))
            vmin = float(vals_b.min())
            vmax = float(vals_b.max())
            pct_at_max = 100.0 * float((vals_b == vmax).sum()) / n_cells_best
            pct_at_min = 100.0 * float((vals_b == vmin).sum()) / n_cells_best
        else:
            solution_sum = 0.0
            n_cells_best = 0
            solution_avg = float("nan")
            vmin = float("nan")
            vmax = float("nan")
            pct_at_max = float("nan")
            pct_at_min = float("nan")

        # ---- input coverage stats (shifted input you already computed)
        n_input_cells = int(len(shifted_input_indices))

        # ---- print the 10 requested values
        if label == list(paths.keys())[0]:
            print("label,n_input_cells,solution_sum,solution_avg,pivot_x,pivot_y,angle_deg,"
                  "min_value_in_solution,max_value_in_solution,pct_cells_at_max(%),pct_cells_at_min(%)")

        print(f"{label},"
              f"{n_input_cells},"
              f"{solution_sum:.6f},"
              f"{solution_avg:.6f},"
              f"{px:.6f},"
              f"{py:.6f},"
              f"{best['angle']:.4f},"
              f"{vmin:.6f},"
              f"{vmax:.6f},"
              f"{pct_at_max:.2f},"
              f"{pct_at_min:.2f}")

        # ---- background field only (no extra markers)

        im = ax.imshow(
            F, origin="lower", cmap="rainbow", vmin=0.0, vmax=1.0,
            extent=[bbox[0], bbox[1], bbox[2], bbox[3]], zorder=0
        )


        # ---- PLOT ONLY THE TWO COVERAGE SETS (no input polyline)
        # Shifted input coverage (top-left)
        shifted_mask = indices_to_mask(shifted_input_indices, F.shape)
        draw_mask_boundary(ax, shifted_mask, N, bbox, lw=1.8, color="white", z=12)

        # Solution OUTLINE at placed pivot (black stroke)
        best_mask = indices_to_mask(best_indices, F.shape)
        draw_mask_boundary(ax, best_mask, N, bbox, lw=2.2, color="black", z=13)

        # ---- pivots
        ax.plot(ix_shift, iy_shift, marker="o", markersize=7,
                color="white", mec="black", mew=1.0, zorder=12, label="input pivot (shifted)")
        ax.plot(px, py, marker="o", markersize=7,
                color="black", mec="white", mew=1.0, zorder=13, label="solution pivot")

        # ---- tidy axes
        
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])  # <-- note: parentheses, not brackets
        ax.set_aspect("equal")
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)

    # one shared colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create an axis on the right side of the figure
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.02)
    #cbar.set_label("Value (0–1)", fontsize=11)
    #fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.2, hspace=0.05, wspace=0.05)

    plt.show()
    """

    bbox = (-x_range_sample, x_range_sample, -x_range_sample, x_range_sample)
    angles = np.linspace(-180, 180, n_angles, dtype=np.float32)
    for i, (label, path_cont) in enumerate(list(paths.items())[:6], start=1):
        # Create an individual figure per input
        fig_i, ax = plt.subplots(figsize=(5.8, 5.8))
        im = None

        t0 = time.perf_counter()
        best, meta, ker, best_ker_r, orig_indices, best_indices = run_search_and_indices(
            F, path_cont, thickness_px=THICKNESS_PX, angles_deg=angles, bbox=bbox,
            supersample=0, close=False
        )
        t1 = time.perf_counter()

        need_retry = (len(orig_indices) > 0 and len(best_indices) < COVERAGE_RETRY_RATIO * len(orig_indices))
        if need_retry:
            best, meta, ker, best_ker_r, orig_indices, best_indices = run_search_and_indices(
                F, path_cont, thickness_px=THICKNESS_PX, angles_deg=angles, bbox=bbox,
                supersample=RETRY_SUPERSAMPLE, close=RETRY_CLOSE
            )
        print(f"Search time [{label}]: {t1 - t0:.6f}s" + (" + retry" if need_retry else ""))

        # ---- compute pivots
        px, py = best_pivot_xy(best, N, bbox)  # solution pivot (global)
        ix, iy = input_pivot_xy(meta, N, bbox)  # input pivot (original)

        # ---- shifted input to top-left
        path_arr = np.asarray(path_cont, np.float32)
        shifted_path, (dx, dy) = shift_to_top_left(path_arr, x_range_sample-x_range_sample/30.0)
        ix_shift, iy_shift = shift_point(ix, iy, dx, dy)
        pts_grid_shifted = cont_to_idx(shifted_path, N, *bbox)
        ker_shift, meta_shift = rasterize_path_kernel_with_meta(
            pts_grid_shifted, thickness_px=THICKNESS_PX
        )
        shifted_input_indices = footprint_indices_original(ker_shift, meta_shift, F.shape)

        # ---- compute stats (10 values, including max)
        if best_indices:
            rr_b, cc_b = np.array(best_indices, dtype=int).T
            vals_b = F[rr_b, cc_b].astype(np.float32)
            solution_sum = float(vals_b.sum())
            n_cells_best = int(vals_b.size)
            solution_avg = float(solution_sum / max(n_cells_best, 1))
            vmin = float(vals_b.min())
            vmax = float(vals_b.max())
            pct_at_max = 100.0 * float((vals_b == vmax).sum()) / n_cells_best
            pct_at_min = 100.0 * float((vals_b == vmin).sum()) / n_cells_best
        else:
            solution_sum = 0.0
            n_cells_best = 0
            solution_avg = float("nan")
            vmin = float("nan")
            vmax = float("nan")
            pct_at_max = float("nan")
            pct_at_min = float("nan")

        n_input_cells = int(len(shifted_input_indices))

        if i == 1:
            print("label,n_input_cells,solution_sum,solution_avg,pivot_x,pivot_y,angle_deg,"
                  "min_value_in_solution,max_value_in_solution,pct_cells_at_max(%),pct_cells_at_min(%)")
        print(f"{label},"
              f"{n_input_cells},"
              f"{solution_sum:.6f},"
              f"{solution_avg:.6f},"
              f"{px:.6f},"
              f"{py:.6f},"
              f"{best['angle']:.4f},"
              f"{vmin:.6f},"
              f"{vmax:.6f},"
              f"{pct_at_max:.2f},"
              f"{pct_at_min:.2f}")

        # ---- plot background field
        im = ax.imshow(
            F, origin="lower", cmap="rainbow", vmin=0.0, vmax=1.0,
            extent=[bbox[0], bbox[1], bbox[2], bbox[3]], zorder=0
        )

        # ---- coverage outlines
        shifted_mask = indices_to_mask(shifted_input_indices, F.shape)
        draw_mask_boundary(ax, shifted_mask, N, bbox, lw=1.8, color="white", z=12)

        best_mask = indices_to_mask(best_indices, F.shape)
        draw_mask_boundary(ax, best_mask, N, bbox, lw=2.2, color="black", z=13)

        # ---- pivots
        ax.plot(ix_shift, iy_shift, marker="o", markersize=7,
                color="white", mec="black", mew=1.0, zorder=12, label="input pivot (shifted)")
        ax.plot(px, py, marker="o", markersize=7,
                color="black", mec="white", mew=1.0, zorder=13, label="solution pivot")

        # ---- axes & title
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        ax.set_aspect("equal")
        ax.set_xlabel("x", fontsize=22)
        ax.set_ylabel("y", fontsize=22)
        ax.tick_params(axis='x', labelsize=18)  # Increase font size for X-axis ticks
        ax.tick_params(axis='y', labelsize=18)
        #ax.set_title(f"{label}: angle {best['angle']:.2f}°, avg {solution_avg:.3f}")

        # per-figure colorbar on the right
        #cbar = fig_i.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        #cbar.set_label("Value (0–1)", fontsize=10)

    # show all figures at once
    plt.show()
    """
    bbox = (-x_range_sample, x_range_sample, -x_range_sample, x_range_sample)
    angles = np.linspace(-180, 180, n_angles, dtype=np.float32)
    path_cont = path_line()  # or any of your generators
    
    angle_deg = -160
    pivot_xy = (0.67,0.04)
    _ = plot_manual_solution(F, path_cont, angle_deg, pivot_xy,
                             thickness_px=THICKNESS_PX, bbox=bbox, ax=None)
    angle_deg = -160
    pivot_xy = (0.47, -0.56)
    _ = plot_manual_solution(F, path_cont, angle_deg, pivot_xy,
                             thickness_px=THICKNESS_PX, bbox=bbox, ax=None)
    """
    bbox = (-x_range_sample, x_range_sample, -x_range_sample, x_range_sample)
    path_cont = path_line()

    case1 = (-160, (0.67, 0.04))
    case2 = (-160, (0.47, -0.56))

    plot_two_manual_solutions(
        F,
        path_cont,
        case1,
        case2,
        thickness_px=THICKNESS_PX,
        bbox=bbox,
        ax=None,  # or pass an existing ax to embed in a larger figure
    )

