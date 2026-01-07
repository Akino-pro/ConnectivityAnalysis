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
joint_reliabilities = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
x_range_sample = np.sum([0.4454, 0.3143, 0.2553])

# plotting style
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

# bbox in continuous coords
BBOX = (-x_range_sample, x_range_sample, -x_range_sample, x_range_sample)
RADIUS_CLIP = 3.0


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
    mask = np.hypot(path[:, 0], path[:, 1]) <= radius
    return path[mask]


# ============================ reliability map =================================
def compute_all_beta_range():
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


def nearest_x_index(x, xmin=0, xmax=x_range_sample):
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

            F[yi, xi] = compute_reliability_by_f_list(F_List, joint_reliabilities)

    return F


# ============================ geometry helpers ================================
def rotation_matrix(phi):
    c, s = float(np.cos(phi)), float(np.sin(phi))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def transform_path_about_centroid(path_cont, t_xy, phi):
    """Rotate about path centroid in its *local definition*, then translate centroid to t."""
    P = np.asarray(path_cont, np.float32)
    pc = P.mean(axis=0)
    Pl = P - pc
    R = rotation_matrix(phi)
    Q = (Pl @ R.T) + np.asarray(t_xy, np.float32)
    return Q, pc


# ============================ bilinear sampling ===============================
def bilinear_sample_F(F, xy_cont, bbox):
    """
    Sample F at continuous points xy_cont using bilinear interpolation.
    Returns values and an in-bounds mask (for bbox only; circle handled separately).
    """
    xmin, xmax, ymin, ymax = bbox
    pts = np.asarray(xy_cont, np.float32)

    # bbox mask
    inb = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)

    # convert to fractional indices
    ij = cont_to_idx(pts, F.shape[0], xmin, xmax, ymin, ymax)
    x = ij[:, 0]
    y = ij[:, 1]

    # clamp to valid interpolation range [0, N-1]
    x = np.clip(x, 0.0, F.shape[1] - 1.000001)
    y = np.clip(y, 0.0, F.shape[0] - 1.000001)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, F.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, F.shape[0] - 1)

    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)

    v00 = F[y0, x0]
    v10 = F[y0, x1]
    v01 = F[y1, x0]
    v11 = F[y1, x1]

    v0 = v00 * (1 - fx) + v10 * fx
    v1 = v01 * (1 - fx) + v11 * fx
    v = v0 * (1 - fy) + v1 * fy

    return v.astype(np.float32), inb


# ============================ objective (mean) ================================
def score_mean_along_path(F, path_cont, t_xy, phi, bbox, radius=3.0, penalty=1e6):
    """
    Mean reliability along sampled points on the *path polyline* after transform.
    Hard-penalize if any point leaves bbox/circle.
    """
    Q, pc = transform_path_about_centroid(path_cont, t_xy, phi)

    # circle mask (execution window)
    in_circle = (np.hypot(Q[:, 0], Q[:, 1]) <= radius)
    vals, in_bbox = bilinear_sample_F(F, Q, bbox)

    feasible = bool(np.all(in_bbox & in_circle))
    if not feasible:
        # penalize by fraction infeasible (smooth-ish for CMA)
        frac_bad = 1.0 - float(np.mean(in_bbox & in_circle))
        return -penalty * frac_bad, Q, feasible

    return float(np.mean(vals)), Q, feasible


# ============================ plotting utilities ==============================
def rasterize_path_kernel(points_xy_grid, thickness_px=1, pad=8):
    pts = np.asarray(points_xy_grid, np.float32)
    xmin, ymin = np.floor(pts.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(pts.max(axis=0)).astype(int)
    w = int(xmax - xmin + 1 + 2 * pad)
    h = int(ymax - ymin + 1 + 2 * pad)
    shift = np.array([xmin - pad, ymin - pad], np.float32)

    ker_full = np.zeros((h, w), np.uint8)
    pts_i = (pts - shift).astype(np.int32)
    cv2.polylines(ker_full, [pts_i], False, 255, thickness=thickness_px)

    ys, xs = np.where(ker_full > 0)
    if xs.size == 0:
        return np.zeros((1, 1), np.uint8), (0, 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    ker = ker_full[y0:y1 + 1, x0:x1 + 1]
    origin = (float(shift[0] + x0), float(shift[1] + y0))  # kernel (0,0) in grid coords
    return ker, origin


def draw_mask_boundary(ax, mask, N, bbox, lw=2.0, color="white", z=12):
    if mask.max() == 0:
        return
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    xmin, xmax, ymin, ymax = bbox
    for cnt in cnts:
        pts = cnt.squeeze(1)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        cols = pts[:, 0].astype(np.float32)
        rows = pts[:, 1].astype(np.float32)
        x_cont, y_cont = idx_to_cont(cols, rows, N, xmin, xmax, ymin, ymax)
        ax.plot(x_cont, y_cont, "-", linewidth=lw, color=color, zorder=z)


def outline_mask_from_polyline(Fshape, path_cont_transformed, thickness_px=1, bbox=BBOX):
    """
    Convert a transformed continuous polyline into a footprint mask for outlining.
    """
    H, W = Fshape
    xmin, xmax, ymin, ymax = bbox
    pts_grid = cont_to_idx(path_cont_transformed, H, xmin, xmax, ymin, ymax)

    ker, origin = rasterize_path_kernel(pts_grid, thickness_px=thickness_px, pad=8)
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
    return mask


def plot_single_solution(
    F,
    path_cont,
    sol,          # (phi, (tx,ty))
    thickness_px=1,
    bbox=BBOX,
    title=None,
    color="white"
):
    xmin, xmax, ymin, ymax = bbox
    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    ax.imshow(
        F, origin="lower", cmap="rainbow", vmin=0.0, vmax=1.0,
        extent=[xmin, xmax, ymin, ymax], zorder=0
    )

    phi, txy = sol
    Q, _ = transform_path_about_centroid(path_cont, txy, phi)
    mask = outline_mask_from_polyline(F.shape, Q, thickness_px=thickness_px, bbox=bbox)
    draw_mask_boundary(ax, mask, F.shape[0], bbox, lw=2.2, color=color, z=13)

    ax.plot(
        txy[0], txy[1], "o", markersize=7,
        color=color, mec="black", mew=1.0, zorder=14
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=22)
    ax.set_ylabel("y", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)

    if title:
        ax.set_title(title, fontsize=16)

    plt.show()


def plot_manual_solutions(
    F,
    path_cont,
    sols,         # list of (phi, (tx,ty), color)
    thickness_px=1,
    bbox=BBOX,
    title=None
):
    xmin, xmax, ymin, ymax = bbox
    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    ax.imshow(
        F, origin="lower", cmap="rainbow", vmin=0.0, vmax=1.0,
        extent=[xmin, xmax, ymin, ymax], zorder=0
    )

    for phi, txy, color in sols:
        Q, _ = transform_path_about_centroid(path_cont, txy, phi)
        mask = outline_mask_from_polyline(F.shape, Q, thickness_px=thickness_px, bbox=bbox)
        draw_mask_boundary(ax, mask, F.shape[0], bbox, lw=2.2, color=color, z=13)

        ax.plot(
            txy[0], txy[1], "o", markersize=7,
            color=color,
            mec=("black" if color == "white" else "white"),
            mew=1.0, zorder=14
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=22)
    ax.set_ylabel("y", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)

    if title:
        ax.set_title(title, fontsize=16)

    plt.show()


# =============================== CMA-ES (numpy) ===============================
def wrap_angle(phi):
    # wrap to [-pi, pi)
    return (phi + np.pi) % (2 * np.pi) - np.pi


def cma_es_optimize(
    F,
    path_cont,
    bbox=BBOX,
    radius=3.0,
    max_iters=120,
    seed=0,
    popsize=None,
    sigma0=None,
    restarts=1,
):
    """
    Optimize over x,y,phi (centroid translation + rotation) using CMA-ES.
    Returns best (score, (x,y), phi, transformed_path_points).
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bbox

    n_dim = 3  # x,y,phi

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

    # helper
    chiN = math.sqrt(n_dim) * (1 - 1 / (4 * n_dim) + 1 / (21 * n_dim**2))

    best_global = (-np.inf, (0.0, 0.0), 0.0, None)

    # bounds for x,y (phi wrapped)
    x_lo, x_hi = xmin, xmax
    y_lo, y_hi = ymin, ymax

    if sigma0 is None:
        sigma0 = 0.25 * max((x_hi - x_lo), (y_hi - y_lo))

    for r in range(restarts):
        # init mean: random feasible-ish
        m = np.zeros(n_dim, dtype=np.float32)
        m[0] = rng.uniform(x_lo * 0.2, x_hi * 0.2)
        m[1] = rng.uniform(y_lo * 0.2, y_hi * 0.2)
        m[2] = rng.uniform(-np.pi, np.pi)

        sigma = float(sigma0)

        C = np.eye(n_dim, dtype=np.float64)
        p_c = np.zeros(n_dim, dtype=np.float64)
        p_s = np.zeros(n_dim, dtype=np.float64)

        for it in range(max_iters):
            # eigendecomposition
            D2, B = np.linalg.eigh(C)
            D2 = np.maximum(D2, 1e-20)
            D = np.sqrt(D2)
            inv_sqrt_C = (B * (1.0 / D)) @ B.T

            # sample
            Z = rng.normal(size=(lam, n_dim)).astype(np.float64)
            Y = Z * D  # scale
            X = (m.astype(np.float64)[None, :] + sigma * (Y @ B.T))  # candidates in parameter space

            # enforce bounds
            X[:, 0] = np.clip(X[:, 0], x_lo, x_hi)
            X[:, 1] = np.clip(X[:, 1], y_lo, y_hi)
            X[:, 2] = np.array([wrap_angle(v) for v in X[:, 2]], dtype=np.float64)

            # evaluate
            fits = np.empty(lam, dtype=np.float64)
            for j in range(lam):
                tx, ty, phi = float(X[j, 0]), float(X[j, 1]), float(X[j, 2])
                fval, Q, feasible = score_mean_along_path(
                    F, path_cont, (tx, ty), phi, bbox=bbox, radius=radius, penalty=1e6
                )
                fits[j] = fval

                if fval > best_global[0]:
                    best_global = (float(fval), (tx, ty), float(phi), Q)

            # sort by fitness (descending)
            idx = np.argsort(-fits)
            Xs = X[idx]
            Zs = Z[idx]

            # recombination
            X_mu = Xs[:mu]
            Z_mu = Zs[:mu]

            m_old = m.astype(np.float64)
            m = np.sum((weights[:, None] * X_mu), axis=0).astype(np.float32)

            # evolution path for sigma
            y_w = np.sum(weights[:, None] * (Z_mu), axis=0)  # in z-space
            p_s = (1 - cs) * p_s + math.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ (B @ (D * y_w)))

            # sigma update
            sigma *= math.exp((cs / damps) * (np.linalg.norm(p_s) / chiN - 1))

            # evolution path for covariance
            hsig = (np.linalg.norm(p_s) / math.sqrt(1 - (1 - cs) ** (2 * (it + 1))) / chiN) < (
                1.4 + 2 / (n_dim + 1)
            )
            hsig = 1.0 if hsig else 0.0
            p_c = (1 - cc) * p_c + hsig * math.sqrt(cc * (2 - cc) * mueff) * (
                (m.astype(np.float64) - m_old) / max(1e-12, sigma)
            )

            # covariance update
            artmp = (Xs[:mu] - m_old[None, :]) / max(1e-12, sigma)
            C = (1 - c1 - cmu) * C + c1 * (
                np.outer(p_c, p_c) + (1 - hsig) * cc * (2 - cc) * C
            ) + cmu * (artmp.T @ (np.diag(weights) @ artmp))

            # stopping heuristics
            if sigma < 1e-4 * max((x_hi - x_lo), (y_hi - y_lo)):
                break

    return best_global


# ============================= example paths ==================================
def path_line():
    y = np.linspace(-0.20, 0.20, 100, dtype=np.float32)
    x = np.zeros_like(y)
    return np.column_stack([x, y])


# =================================== main =====================================
if __name__ == "__main__":
    xmin, xmax, ymin, ymax = BBOX

    start = time.perf_counter()
    F = generate_F(N, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    end = time.perf_counter()
    print(f"reliability map computation took {end - start:.6f} seconds")

    paths = {
        "Line": path_line(),
    }

    # manual cases
    path_cont = paths["Line"]
    case1 = (-160 * np.pi / 180.0, (0.67, 0.04))
    case2 = (-160 * np.pi / 180.0, (0.47, -0.56))

    # compute and print their mean scores
    s1, Q1, feas1 = score_mean_along_path(F, path_cont, case1[1], case1[0], bbox=BBOX, radius=RADIUS_CLIP)
    s2, Q2, feas2 = score_mean_along_path(F, path_cont, case2[1], case2[0], bbox=BBOX, radius=RADIUS_CLIP)
    print(f"Manual1 score_mean={s1:.6f}, feasible={feas1}")
    print(f"Manual2 score_mean={s2:.6f}, feasible={feas2}")

    # CMA-ES optimize (x,y,phi)
    best_score, best_t, best_phi, best_Q = cma_es_optimize(
        F,
        path_cont,
        bbox=BBOX,
        radius=RADIUS_CLIP,
        max_iters=140,
        seed=1,
        popsize=10,
        sigma0=0.8,
        restarts=2,
    )

    print("\nCMA-ES best:")
    print(f"  score_mean = {best_score:.6f}")
    print(f"  t = ({best_t[0]:.6f}, {best_t[1]:.6f})")
    print(f"  phi(deg) = {best_phi * 180.0 / np.pi:.4f}")

    # ------------------- PLOT 1: optimal only -------------------
    plot_single_solution(
        F,
        path_cont,
        sol=(best_phi, best_t),
        thickness_px=1,
        bbox=BBOX,
        title="CMA-ES optimal solution",
        color="white",
    )

    # ------------------- PLOT 2: two manual solutions -------------------
    plot_manual_solutions(
        F,
        path_cont,
        sols=[
            (case1[0], case1[1], "white"),
            (case2[0], case2[1], "black"),
        ],
        thickness_px=1,
        bbox=BBOX,
        title="Manual solutions",
    )
