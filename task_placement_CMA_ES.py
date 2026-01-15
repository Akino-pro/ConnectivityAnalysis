"""
CMA-ES task placement script (multi-task)

- Builds reliability map F (same as your brute-force script)
- For each task/path (Line/Triangle/Rectangle/Half-Moon/Circle/Square/Random curves):
    * Runs CMA-ES on (pivot_x, pivot_y, angle_deg) to maximize SUM of F under rasterized footprint
    * Prints the SAME CSV fields as your brute-force script:
        label,n_input_cells,solution_sum,solution_avg,pivot_x,pivot_y,angle_deg,
        min_value_in_solution,max_value_in_solution,pct_cells_at_max(%),pct_cells_at_min(%)

Notes:
- Objective is SUM under footprint (same type as your brute-force "score_sum")
- Footprint rasterization uses cv2.polylines with thickness THICKNESS_PX
- Pivot placement uses (h//2,w//2) anchor convention (matching your brute-force placement mask logic)
"""

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


# ----------------------- Global config -----------------------
N = 256

joint_reliabilities = [0.5,0.6,0.7]
x_range_sample=3
#joint_reliabilities = [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
#x_range_sample = float(np.sum([0.4454, 0.3143, 0.2553]))  # = 1.015

# tasks / plotting
THICKNESS_PX = 1
RADIUS_CLIP = 3.0

# CMA-ES defaults (you can edit here)
CMA_MAX_ITERS = 140
CMA_SEED = 1
CMA_POPSIZE = 7          # set None to use default rule
CMA_SIGMA0 = 0.8          # set None to use default bbox-based rule
CMA_RESTARTS = 2

# white background
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

BBOX = (-x_range_sample, x_range_sample, -x_range_sample, x_range_sample)


# ===================== mapping between continuous and grid =====================
def cont_to_idx(xy_cont, N, xmin, xmax, ymin, ymax):
    xy = np.asarray(xy_cont, np.float32)
    x_idx = (xy[:, 0] - xmin) / (xmax - xmin) * (N - 1)
    y_idx = (xy[:, 1] - ymin) / (ymax - ymin) * (N - 1)
    return np.column_stack([x_idx, y_idx]).astype(np.float32)


def idx_to_cont(x_idx, y_idx, N, xmin, xmax, ymin, ymax):
    x = x_idx / (N - 1) * (xmax - xmin) + xmin
    y = y_idx / (N - 1) * (ymax - ymin) + ymin
    return x, y


def clip_to_circle(path, radius=3.0):
    path = np.asarray(path, np.float32)
    mask = np.hypot(path[:, 0], path[:, 1]) <= float(radius)
    return path[mask]


# ============================ reliability map =================================
def compute_all_beta_range():
    """
    Load beta_list.txt if valid; otherwise compute and save.
    """
    path = "beta_list.txt"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cached = ast.literal_eval(f.read())
            if isinstance(cached, list) and len(cached) == N:
                return cached
            print("Cached beta_list.txt invalid; recomputing...")
        except Exception as e:
            print(f"Failed to load cached beta_list.txt ({e}); recomputing...")

    section_length = x_range_sample / N
    x_values = (np.arange(N) + 0.5) * section_length
    points = np.column_stack((x_values, np.zeros(N)))

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


def nearest_x_index(x, xmin=0.0, xmax=x_range_sample):
    dx = (xmax - xmin) / N
    i = int(np.round((x - xmin - dx / 2) / dx))
    return int(np.clip(i, 0, N - 1))


def generate_F(N, xmin, xmax, ymin, ymax):
    print("beta range computation starts")
    all_reliable_beta_ranges = compute_all_beta_range()
    print("beta range computation ends")

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
            # if your compute_beta_range returns a "bad" or "unreliable" bin, keep it (as your brute-force code did)
            if isinstance(reliable_beta_ranges, list) and len(reliable_beta_ranges) >= 4:
                for beta_range in reliable_beta_ranges[-1]:
                    if beta_range[0] <= theta <= beta_range[1]:
                        F_List.append(0)

            F[yi, xi] = compute_reliability_by_f_list(F_List, joint_reliabilities)

    return F


# ============================ brute-force style footprint ======================
def wrap_angle_deg(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def rotation_matrix(phi_rad):
    c, s = float(np.cos(phi_rad)), float(np.sin(phi_rad))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def rotate_points_about_centroid(points_xy, angle_deg):
    P = np.asarray(points_xy, np.float32)
    pc = P.mean(axis=0)
    ang = np.deg2rad(float(angle_deg))
    R = rotation_matrix(ang)
    return (P - pc) @ R.T + pc


def rasterize_path_kernel(points_xy_grid, thickness_px=1, pad=8):
    """
    Rasterize an OPEN polyline (outline) to a tight binary kernel.
    Returns (ker_uint8, origin_grid_xy_float) where origin is ker[0,0] location in GRID coords.
    """
    pts = np.asarray(points_xy_grid, np.float32)
    xmin, ymin = np.floor(pts.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(pts.max(axis=0)).astype(int)
    w = int(xmax - xmin + 1 + 2 * pad)
    h = int(ymax - ymin + 1 + 2 * pad)
    shift = np.array([xmin - pad, ymin - pad], np.float32)

    ker_full = np.zeros((h, w), np.uint8)
    pts_i = (pts - shift).astype(np.int32)
    cv2.polylines(ker_full, [pts_i], False, 255, thickness=int(thickness_px))

    ys, xs = np.where(ker_full > 0)
    if xs.size == 0:
        return np.zeros((1, 1), np.uint8), (0.0, 0.0)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    ker = (ker_full[y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8)

    origin = (float(shift[0] + x0), float(shift[1] + y0))
    return ker, origin


def place_kernel_at_pivot_mask(Fshape, ker, pivot_xy_cont, bbox):
    """
    Place ker so its integer anchor (h//2,w//2) lands at pivot (rounded to nearest grid cell).
    Returns: mask (H,W) uint8 with 1 under footprint.
    """
    H, W = Fshape
    h, w = ker.shape
    cy_anchor = h // 2
    cx_anchor = w // 2

    xmin, xmax, ymin, ymax = bbox
    x_idx, y_idx = cont_to_idx(
        np.array([[pivot_xy_cont[0], pivot_xy_cont[1]]], dtype=np.float32),
        H, xmin, xmax, ymin, ymax
    )[0]

    r = int(round(float(y_idx)))
    c = int(round(float(x_idx)))

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

    mask = np.zeros((H, W), np.uint8)
    if r1 > r0 and c1 > c0:
        mask[r0:r1, c0:c1] = (ker[kr0:kr1, kc0:kc1] > 0).astype(np.uint8)

    return mask


def placement_stats_from_mask(F, mask):
    vals = F[mask > 0].astype(np.float32)
    if vals.size == 0:
        return {
            "solution_sum": 0.0,
            "solution_avg": float("nan"),
            "min_value": float("nan"),
            "max_value": float("nan"),
            "pct_at_max": float("nan"),
            "pct_at_min": float("nan"),
            "n_cells": 0,
        }

    s = float(vals.sum())
    n = int(vals.size)
    avg = float(s / n)
    vmin = float(vals.min())
    vmax = float(vals.max())
    pct_max = 100.0 * float((vals == vmax).sum()) / n
    pct_min = 100.0 * float((vals == vmin).sum()) / n

    return {
        "solution_sum": s,
        "solution_avg": avg,
        "min_value": vmin,
        "max_value": vmax,
        "pct_at_max": pct_max,
        "pct_at_min": pct_min,
        "n_cells": n,
    }


def score_sum_for_pose(F, path_cont, pivot_xy, angle_deg, bbox, thickness_px=1, penalty=1e12):
    """
    Fitness: SUM under footprint (brute-force style).
    Returns (fitness_sum, mask, stats_dict).
    """
    rot_pts = rotate_points_about_centroid(path_cont, angle_deg)
    pts_grid = cont_to_idx(rot_pts, F.shape[0], *bbox)

    ker, _ = rasterize_path_kernel(pts_grid, thickness_px=thickness_px, pad=8)
    mask = place_kernel_at_pivot_mask(F.shape, ker, pivot_xy, bbox)

    stats = placement_stats_from_mask(F, mask)
    if stats["n_cells"] == 0:
        return -penalty, mask, stats

    return stats["solution_sum"], mask, stats


# ====================== shifted-input cell count (reporting) ===================
def shift_to_top_left(path_xy: np.ndarray, box: float):
    """
    Shift a polyline so xmin -> -box and ymax -> +box.
    Returns shifted_path, (dx, dy) where new = old - (dx, dy).
    """
    path_xy = np.asarray(path_xy, np.float32)
    xmin, ymin = path_xy.min(axis=0)
    xmax, ymax = path_xy.max(axis=0)

    dx = xmin + box
    dy = -(box - ymax)

    shifted = np.column_stack([path_xy[:, 0] - dx, path_xy[:, 1] - dy]).astype(np.float32)
    return shifted, (dx, dy)


def input_shifted_cell_count(Fshape, path_cont, bbox, thickness_px=1):
    """
    For CSV: n_input_cells = footprint cell count of the 'shifted-to-top-left' input path.
    Matches your brute-force reporting intention (not used in optimization).
    """
    path_arr = np.asarray(path_cont, np.float32)
    shifted_path, _ = shift_to_top_left(path_arr, x_range_sample - x_range_sample / 30.0)

    pts_grid = cont_to_idx(shifted_path, Fshape[0], *bbox)
    ker, origin = rasterize_path_kernel(pts_grid, thickness_px=thickness_px, pad=8)

    H, W = Fshape
    h, w = ker.shape
    ox, oy = origin
    r0 = int(round(oy))
    c0 = int(round(ox))

    r_start = max(0, r0)
    c_start = max(0, c0)
    r_end = min(H, r0 + h)
    c_end = min(W, c0 + w)

    kr0 = r_start - r0
    kc0 = c_start - c0
    kr1 = kr0 + (r_end - r_start)
    kc1 = kc0 + (c_end - c_start)

    mask = np.zeros((H, W), np.uint8)
    if r_end > r_start and c_end > c_start:
        mask[r_start:r_end, c_start:c_end] = (ker[kr0:kr1, kc0:kc1] > 0).astype(np.uint8)

    return int(mask.sum())


# =============================== CMA-ES (3D pose) ==============================
def cma_es_optimize_pose_sum(
    F,
    path_cont,
    bbox,
    thickness_px=1,
    max_iters=140,
    seed=1,
    popsize=10,     # None => default rule
    sigma0=0.8,     # None => bbox-based default
    restarts=2,
):
    """
    CMA-ES over 3 variables: [px, py, angle_deg]
    Fitness: SUM under rasterized footprint mask.
    Returns dict:
        best["score_sum"], best["pivot"], best["angle_deg"], best["stats"], best["mask"]
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bbox

    n_dim = 3  # px, py, angle_deg

    if popsize is None:
        lam = 4 + int(3 * np.log(n_dim))
    else:
        lam = int(popsize)
    mu = lam // 2

    # log weights
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights**2)

    # strategy params
    cc = (4 + mueff / n_dim) / (n_dim + 4 + 2 * mueff / n_dim)
    cs = (mueff + 2) / (n_dim + mueff + 5)
    c1 = 2 / ((n_dim + 1.3) ** 2 + mueff)
    cmu = min(
        1 - c1,
        2 * (mueff - 2 + 1 / mueff) / ((n_dim + 2) ** 2 + mueff),
    )
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n_dim + 1)) - 1) + cs

    chiN = math.sqrt(n_dim) * (1 - 1 / (4 * n_dim) + 1 / (21 * n_dim**2))

    if sigma0 is None:
        sigma0 = 0.25 * max((xmax - xmin), (ymax - ymin))

    best = {
        "score_sum": -np.inf,
        "pivot": (0.0, 0.0),
        "angle_deg": 0.0,
        "stats": None,
        "mask": None,
    }

    for _r in range(restarts):
        # mean init
        m = np.zeros(n_dim, dtype=np.float64)
        m[0] = rng.uniform(xmin * 0.2, xmax * 0.2)
        m[1] = rng.uniform(ymin * 0.2, ymax * 0.2)
        m[2] = rng.uniform(-180.0, 180.0)

        sigma = float(sigma0)

        C = np.eye(n_dim, dtype=np.float64)
        p_c = np.zeros(n_dim, dtype=np.float64)
        p_s = np.zeros(n_dim, dtype=np.float64)

        for it in range(max_iters):
            D2, B = np.linalg.eigh(C)
            D2 = np.maximum(D2, 1e-20)
            D = np.sqrt(D2)
            inv_sqrt_C = (B * (1.0 / D)) @ B.T

            Z = rng.normal(size=(lam, n_dim)).astype(np.float64)
            Y = Z * D
            X = (m[None, :] + sigma * (Y @ B.T))

            # enforce bounds / wrapping
            X[:, 0] = np.clip(X[:, 0], xmin, xmax)
            X[:, 1] = np.clip(X[:, 1], ymin, ymax)
            X[:, 2] = np.array([wrap_angle_deg(v) for v in X[:, 2]], dtype=np.float64)

            fits = np.empty(lam, dtype=np.float64)

            for j in range(lam):
                px, py, ang = float(X[j, 0]), float(X[j, 1]), float(X[j, 2])
                fval, mask, stats = score_sum_for_pose(
                    F, path_cont, (px, py), ang, bbox=bbox, thickness_px=thickness_px
                )
                fits[j] = fval

                if fval > best["score_sum"]:
                    best.update(
                        score_sum=float(fval),
                        pivot=(px, py),
                        angle_deg=float(ang),
                        stats=stats,
                        mask=mask,
                    )

            # sort
            idx = np.argsort(-fits)
            Xs = X[idx]
            Zs = Z[idx]

            X_mu = Xs[:mu]
            Z_mu = Zs[:mu]

            m_old = m.copy()
            m = np.sum(weights[:, None] * X_mu, axis=0)

            # update p_s and sigma
            y_w = np.sum(weights[:, None] * Z_mu, axis=0)
            p_s = (1 - cs) * p_s + math.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ (B @ (D * y_w)))
            sigma *= math.exp((cs / damps) * (np.linalg.norm(p_s) / chiN - 1))

            # update p_c
            hsig = (np.linalg.norm(p_s) / math.sqrt(1 - (1 - cs) ** (2 * (it + 1))) / chiN) < (
                1.4 + 2 / (n_dim + 1)
            )
            hsig = 1.0 if hsig else 0.0
            p_c = (1 - cc) * p_c + hsig * math.sqrt(cc * (2 - cc) * mueff) * (
                (m - m_old) / max(1e-12, sigma)
            )

            # covariance update
            artmp = (Xs[:mu] - m_old[None, :]) / max(1e-12, sigma)
            C = (1 - c1 - cmu) * C + c1 * (
                np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C
            ) + cmu * (artmp.T @ (np.diag(weights) @ artmp))

            # stop if sigma tiny
            if sigma < 1e-4 * max((xmax - xmin), (ymax - ymin)):
                break

    return best


# ============================= example task paths ==============================
def _sample_edge(p0, p1, n):
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)[:, None]
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    return p0 + (p1 - p0) * t


def polyline_rectangle(center, w, h, points_per_edge=120):
    cx, cy = center
    hw, hh = 0.5 * w, 0.5 * h
    v0 = (cx - hw, cy - hh)
    v1 = (cx + hw, cy - hh)
    v2 = (cx + hw, cy + hh)
    v3 = (cx - hw, cy + hh)

    e01 = _sample_edge(v0, v1, points_per_edge)
    e12 = _sample_edge(v1, v2, points_per_edge)
    e23 = _sample_edge(v2, v3, points_per_edge)
    e30 = _sample_edge(v3, v0, points_per_edge)

    poly = np.vstack([e01, e12, e23, e30, np.asarray(v0, np.float32)[None, :]])
    return poly.astype(np.float32)


def polyline_equilateral_triangle(center, side, rot_deg=0.0, points_per_edge=150):
    cx, cy = center
    R = side / np.sqrt(3)
    ang0 = np.deg2rad(rot_deg)
    a0 = ang0
    a1 = ang0 + 2 * np.pi / 3
    a2 = ang0 + 4 * np.pi / 3
    v0 = (cx + R * np.cos(a0), cy + R * np.sin(a0))
    v1 = (cx + R * np.cos(a1), cy + R * np.sin(a1))
    v2 = (cx + R * np.cos(a2), cy + R * np.sin(a2))

    e01 = _sample_edge(v0, v1, points_per_edge)
    e12 = _sample_edge(v1, v2, points_per_edge)
    e20 = _sample_edge(v2, v0, points_per_edge)

    poly = np.vstack([e01, e12, e20, np.asarray(v0, np.float32)[None, :]])
    return poly.astype(np.float32)


def path_half_moon(center=(0.0, 0.0), R=1.6, r=1.0, dx=0.6, n_outer=240, n_inner=240):
    cx, cy = center
    c0 = np.array([cx, cy], np.float32)
    c1 = np.array([cx + dx, cy], np.float32)

    d = float(np.linalg.norm(c1 - c0))
    if not (abs(R - r) < d < (R + r)):
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

    ex = (c1 - c0) / d
    a = (R * R - r * r + d * d) / (2 * d)
    h_sq = R * R - a * a
    h = math.sqrt(max(h_sq, 0.0))
    p2 = c0 + a * ex
    perp = np.array([-ex[1], ex[0]], np.float32)
    i1 = p2 + h * perp
    i2 = p2 - h * perp

    th1, th2 = math.atan2(i1[1] - cy, i1[0] - cx), math.atan2(i2[1] - cy, i2[0] - cx)
    ph1 = math.atan2(i1[1] - cy, i1[0] - (cx + dx))
    ph2 = math.atan2(i2[1] - cy, i2[0] - (cx + dx))

    def arc_ccw(a0, a1, n):
        a0u, a1u = np.unwrap([a0, a1])
        if a1u < a0u:
            a1u += 2 * np.pi
        return np.linspace(a0u, a1u, n, endpoint=False, dtype=np.float32)

    outer_t = arc_ccw(th1, th2, n_outer)
    inner_t = arc_ccw(ph1, ph2, n_inner)[::-1]

    outer = np.column_stack([cx + R * np.cos(outer_t), cy + R * np.sin(outer_t)])
    inner = np.column_stack([cx + dx + r * np.cos(inner_t), cy + r * np.sin(inner_t)])

    poly = np.vstack([outer, inner, outer[:1]])
    return poly.astype(np.float32)


def path_line():
    y = np.linspace(-0.20, 0.20, 100, dtype=np.float32)
    x = np.zeros_like(y)
    return np.column_stack([x, y])


def path_triangle():
    tri = polyline_equilateral_triangle(center=(0.0, -0.2), side=2.0, rot_deg=0.0, points_per_edge=160)
    return clip_to_circle(tri, radius=RADIUS_CLIP)


def path_rectangle():
    rect = polyline_rectangle(center=(0.8, -0.2), w=2.8, h=0.6, points_per_edge=260)
    return clip_to_circle(rect, radius=RADIUS_CLIP)


def path_circle(num_pts=300, radius=1.2):
    theta = np.linspace(0, 2 * np.pi, num_pts, endpoint=True, dtype=np.float32)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y]).astype(np.float32)


def path_square(side=1.0, pts_per_edge=60):
    half = side / 2.0
    v0 = (-half, -half)
    v1 = (half, -half)
    v2 = (half, half)
    v3 = (-half, half)

    x0 = np.linspace(v0[0], v1[0], pts_per_edge, endpoint=False)
    y0 = np.full_like(x0, v0[1])

    x1 = np.full(pts_per_edge, v1[0])
    y1 = np.linspace(v1[1], v2[1], pts_per_edge, endpoint=False)

    x2 = np.linspace(v2[0], v3[0], pts_per_edge, endpoint=False)
    y2 = np.full_like(x2, v2[1])

    x3 = np.full(pts_per_edge, v3[0])
    y3 = np.linspace(v3[1], v0[1], pts_per_edge, endpoint=False)

    xs = np.concatenate([x0, x1, x2, x3, [v0[0]]]).astype(np.float32)
    ys = np.concatenate([y0, y1, y2, y3, [v0[1]]]).astype(np.float32)
    return clip_to_circle(np.column_stack([xs, ys]), radius=RADIUS_CLIP)


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

        path = clip_to_circle(path, radius=RADIUS_CLIP)
        paths.append(path)

    return paths


# ============================= example task paths ==============================
def path_short():
    # (matches your brute-force "Sin curve" definition)
    x = np.linspace(-1.5, 1.5, 500, dtype=np.float32)
    y = 0.5 * np.sin(2.5 * np.pi * x / 0.7)
    return clip_to_circle(np.column_stack([x, y]), radius=RADIUS_CLIP)

def path_medium():
    # (matches your brute-force "Triangle" definition)
    tri = polyline_equilateral_triangle(
        center=(0.0, -0.2),
        side=2.0,
        rot_deg=0.0,
        points_per_edge=160
    )
    return clip_to_circle(tri, radius=RADIUS_CLIP)

def path_long():
    # (matches your brute-force "Rectangle" definition)
    rect = polyline_rectangle(
        center=(0.8, -0.2),
        w=2.8,
        h=0.6,
        points_per_edge=260
    )
    return clip_to_circle(rect, radius=RADIUS_CLIP)

# (path_line, path_circle already match your brute-force ones)



"""
ADD-ON: plotting the CMA-ES placement like your brute-force script.

This assumes you already have in your CMA-ES script:
- cont_to_idx, idx_to_cont, clip_to_circle
- rasterize_path_kernel, place_kernel_at_pivot_mask
- shift_to_top_left
- cma_es_optimize_pose_sum (returns best["mask"], best["pivot"], best["angle_deg"], best["stats"])
- F, bbox, THICKNESS_PX, N, x_range_sample

Paste the functions below into your CMA-ES script (near other plotting/helpers),
then inside the main loop call `plot_cma_solution_like_bruteforce(...)`.
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np


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
    if mask is None or mask.max() == 0:
        return
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    xmin, xmax, ymin, ymax = bbox
    for cnt in cnts:
        pts = cnt.squeeze(1)  # (K,2) with columns=(col,row)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        cols = pts[:, 0].astype(np.float32)
        rows = pts[:, 1].astype(np.float32)
        x_cont, y_cont = idx_to_cont(cols, rows, N, xmin, xmax, ymin, ymax)
        ax.plot(x_cont, y_cont, "-", linewidth=lw, color=color, zorder=z)


def rotate_points_about_centroid(points_xy, angle_deg):
    P = np.asarray(points_xy, np.float32)
    pc = P.mean(axis=0)
    ang = np.deg2rad(float(angle_deg))
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    return (P - pc) @ R.T + pc


def build_shifted_input_mask_and_pivot(path_cont, Fshape, bbox, thickness_px=1):
    """
    Mimic your brute-force 'shifted input' outline (top-left) for plotting + n_input_cells.
    Returns:
        shifted_mask (H,W) uint8, shifted_input_pivot (ix_shift, iy_shift)
    """
    path_arr = np.asarray(path_cont, np.float32)

    # same shift used in your brute force:
    shifted_path, (dx, dy) = shift_to_top_left(path_arr, x_range_sample - x_range_sample / 30.0)

    # choose input pivot as the centroid of the *original* path, then apply same shift
    pc = path_arr.mean(axis=0)
    ix_shift = float(pc[0] - dx)
    iy_shift = float(pc[1] - dy)

    # rasterize shifted path to a kernel and place it at its own origin (like brute force orig kernel placement)
    pts_grid = cont_to_idx(shifted_path, Fshape[0], *bbox)
    ker, origin = rasterize_path_kernel(pts_grid, thickness_px=thickness_px, pad=8)

    H, W = Fshape
    h, w = ker.shape
    ox, oy = origin  # kernel (0,0) in grid coords
    r0 = int(round(oy))
    c0 = int(round(ox))

    r_start = max(0, r0)
    c_start = max(0, c0)
    r_end = min(H, r0 + h)
    c_end = min(W, c0 + w)

    kr0 = r_start - r0
    kc0 = c_start - c0
    kr1 = kr0 + (r_end - r_start)
    kc1 = kc0 + (c_end - c_start)

    shifted_mask = np.zeros((H, W), np.uint8)
    if r_end > r_start and c_end > c_start:
        shifted_mask[r_start:r_end, c_start:c_end] = (ker[kr0:kr1, kc0:kc1] > 0).astype(np.uint8)

    return shifted_mask, (ix_shift, iy_shift)


def plot_cma_solution_like_bruteforce(
    F,
    path_cont,
    best,                 # output of cma_es_optimize_pose_sum
    label,
    bbox,
    thickness_px=1,
):
    """
    Plot exactly like your brute-force per-case figure:
      - background F
      - shifted-input outline (white) + its pivot (white dot)
      - CMA solution outline (black) + solution pivot (black dot)
    """
    xmin, xmax, ymin, ymax = bbox
    N = F.shape[0]

    # --- build shifted input outline + pivot ---
    shifted_mask, (ix_shift, iy_shift) = build_shifted_input_mask_and_pivot(
        path_cont, F.shape, bbox, thickness_px=thickness_px
    )

    # --- CMA best outline mask and pivot ---
    best_mask = best.get("mask", None)
    px, py = best["pivot"]
    ang = best["angle_deg"]

    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    ax.imshow(
        F, origin="lower", cmap="rainbow", vmin=0.0, vmax=1.0,
        extent=[xmin, xmax, ymin, ymax], zorder=0
    )

    # outlines (match brute force styling)
    draw_mask_boundary(ax, shifted_mask, N, bbox, lw=1.8, color="white", z=12)
    draw_mask_boundary(ax, best_mask,   N, bbox, lw=2.2, color="black", z=13)

    # pivots
    ax.plot(ix_shift, iy_shift, marker="o", markersize=7,
            color="white", mec="black", mew=1.0, zorder=12)
    ax.plot(px, py, marker="o", markersize=7,
            color="black", mec="white", mew=1.0, zorder=13)

    # axes
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=22)
    ax.set_ylabel("y", fontsize=22)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    # optional title
    # ax.set_title(f"{label}: angle {ang:.2f}°", fontsize=14)

    plt.show()


# =================================== main =====================================
if __name__ == "__main__":
    xmin, xmax, ymin, ymax = BBOX

    start = time.perf_counter()
    F = generate_F(N, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    end = time.perf_counter()
    print(f"reliability map computation took {end - start:.6f} seconds")

    # ---- EXACT 6 cases (same naming as your brute-force printout) ----
    paths = {
        "Sin curve": path_short(),
        "Triangle":  path_medium(),
        "Line":      path_line(),
        "Circle":    clip_to_circle(path_circle(), radius=RADIUS_CLIP),
        "Rectangle": path_long(),
    }

    # One random curve named exactly "Random curve"
    # (brute-force settings shown in your script: num_ctrl=7, samples_per_seg=32, box=1, max_length=4)
    rand_paths = gen_random_continuous_inputs(
        n=1, num_ctrl=7, samples_per_seg=32, box=1.0, max_length=4.0, seed=1
    )
    if len(rand_paths) > 0:
        paths["Random curve"] = rand_paths[0]

    # ---- header (same as brute-force print) ----
    print(
        "label,n_input_cells,solution_sum,solution_avg,pivot_x,pivot_y,angle_deg,"
        "min_value_in_solution,max_value_in_solution,pct_cells_at_max(%),pct_cells_at_min(%)"
    )

    bbox = BBOX

    for label, path_cont in paths.items():
        # same "n_input_cells" convention as your brute-force report
        n_input_cells = input_shifted_cell_count(
            F.shape, path_cont, bbox=bbox, thickness_px=THICKNESS_PX
        )

        t0 = time.perf_counter()
        best = cma_es_optimize_pose_sum(
            F,
            path_cont,
            bbox=bbox,
            thickness_px=THICKNESS_PX,
            max_iters=CMA_MAX_ITERS,
            seed=CMA_SEED,
            popsize=CMA_POPSIZE,
            sigma0=CMA_SIGMA0,
            restarts=CMA_RESTARTS,
        )
        t1 = time.perf_counter()

        # match your brute-force timing line format
        print(f"Search time [{label}]: {t1 - t0:.6f}s")

        px, py = best["pivot"]
        ang = best["angle_deg"]
        st = best["stats"] if best["stats"] is not None else {}

        solution_sum = float(best["score_sum"])
        solution_avg = float(st.get("solution_avg", float("nan")))
        vmin = float(st.get("min_value", float("nan")))
        vmax = float(st.get("max_value", float("nan")))
        pct_max = float(st.get("pct_at_max", float("nan")))
        pct_min = float(st.get("pct_at_min", float("nan")))

        print(
            f"{label},"
            f"{n_input_cells},"
            f"{solution_sum:.6f},"
            f"{solution_avg:.6f},"
            f"{px:.6f},"
            f"{py:.6f},"
            f"{ang:.4f},"
            f"{vmin:.6f},"
            f"{vmax:.6f},"
            f"{pct_max:.2f},"
            f"{pct_min:.2f}"
        )

        plot_cma_solution_like_bruteforce(
             F,
             path_cont,
             best,
             label=label,
             bbox=bbox,
             thickness_px=THICKNESS_PX,
        )

