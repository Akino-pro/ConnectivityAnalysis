import os
from pathlib import Path
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import linprog
import polytope as poly
from rtree import index

_REGIONS = None
_RTREE = None

def _init_worker(atlas_path: str, rtree_prefix: str):
    """Runs once per worker process."""
    global _REGIONS, _RTREE

    with open(atlas_path, "rb") as f:
        _REGIONS = pickle.load(f)

    p = index.Property()
    p.dimension = 2
    _RTREE = index.Index(rtree_prefix, properties=p)


def _check_region_idx(idx, constraint_xy):
    try:
        x, y = constraint_xy
        constraint = np.array([x, y])
        proj = _REGIONS[idx][[0, 1]] == constraint
        if poly.poly_empty(proj.domain):
            return None
        return proj
    except Exception:
        return None


def _get_aabb(proj_region, constraint_xy):
    try:
        x, y = constraint_xy
        constraint = np.array([x, y])

        A = proj_region.base_domain.A
        b = proj_region.base_domain.b
        C = proj_region.C
        d = proj_region.d

        n = A.shape[1]
        aabb = [0.0] * (2 * n)

        rhs = constraint - d
        for i in range(n):
            c = np.zeros(n)
            c[i] = 1.0

            res_min = linprog(c,  A_ub=A, b_ub=b, A_eq=C, b_eq=rhs, bounds=(None, None))
            res_max = linprog(-c, A_ub=A, b_ub=b, A_eq=C, b_eq=rhs, bounds=(None, None))

            if res_min.success and res_max.success:
                aabb[i] = res_min.x[i]
                aabb[i + n] = res_max.x[i]
            else:
                return None

        return tuple(aabb)
    except Exception:
        return None


def _merge_intervals(intervals, tol=1e-3):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def compute_smm_ranges(x, y, n_workers=None, tol=1e-3, verbose=False):
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # absolute paths relative to THIS file (spawn-safe)
    base = Path(__file__).resolve().parent
    atlas_path = str(base / "data" / "4R_atlas.pkl")
    rtree_prefix = str(base / "data" / "4R_rtree")

    constraint_xy = (float(x), float(y))

    # IMPORTANT: Use one pool and do both stages inside it.
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(atlas_path, rtree_prefix),
    ) as ex:

        # candidates must be computed in parent (needs rtree)
        # easiest: open a small parent rtree just for intersection
        p = index.Property()
        p.dimension = 2
        parent_rtree = index.Index(rtree_prefix, properties=p)

        candidates = list(parent_rtree.intersection((x, y, x, y)))
        if verbose:
            # regions length known only in workers; print candidate count only
            print(f"R-tree candidates: {len(candidates)}")

        futures = [ex.submit(_check_region_idx, i, constraint_xy) for i in candidates]
        valid = []
        for i, f in enumerate(as_completed(futures)):
            res = f.result()
            if res is not None:
                valid.append(res)
            if verbose:
                print(f"\rChecking regions: {i+1}/{len(candidates)}, valid: {len(valid)}", end="")
        if verbose:
            print()

        if not valid:
            return []

        futures = [ex.submit(_get_aabb, v, constraint_xy) for v in valid]
        aabbs = []
        for i, f in enumerate(as_completed(futures)):
            res = f.result()
            if res is not None:
                aabbs.append(res)
            if verbose:
                print(f"\rComputing AABBs: {i+1}/{len(valid)}", end="")
        if verbose:
            print()

    if not aabbs:
        return []

    n_dims = len(aabbs[0]) // 2
    result = []
    for dim in range(n_dims):
        intervals = [(aabb[dim], aabb[dim + n_dims]) for aabb in aabbs]
        merged = _merge_intervals(intervals, tol)
        result.append([(float(lo), float(hi)) for lo, hi in merged])

    return result

"""
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    ranges = compute_smm_ranges(3.0, 0.0, n_workers=10, verbose=True)
    print(ranges)
"""
