import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import pickle
from scipy.optimize import linprog
import polytope as poly
from rtree import index

with open('./data/4R_atlas.pkl', 'rb') as _f:
    _REGIONS = pickle.load(_f)

_p = index.Property()
_p.dimension = 2
_RTREE = index.Index('./data/4R_rtree', properties=_p)

_CONSTRAINT = None


def _check_region_idx(idx, constraint):
    try:
        proj = _REGIONS[idx][[0, 1]] == constraint
        if poly.poly_empty(proj.domain):
            return None
        return proj
    except:
        return None


def _get_bounded_range(args):
    proj_region, bounds, constraint = args
    try:
        A = proj_region.base_domain.A
        b = proj_region.base_domain.b
        C = proj_region.C
        d = proj_region.d
        n = A.shape[1]

        # Only search along first dimension with bounds applied
        c = np.zeros(n)
        c[0] = 1.0
        res_min = linprog(c, A_ub=A, b_ub=b, A_eq=C, b_eq=constraint - d, bounds=bounds)
        res_max = linprog(-c, A_ub=A, b_ub=b, A_eq=C, b_eq=constraint - d, bounds=bounds)

        if res_min.success and res_max.success:
            return (res_min.x[0], res_max.x[0])
        return None
    except:
        return None


def _get_aabb(args):
    proj_region, constraint = args
    try:
        A = proj_region.base_domain.A
        b = proj_region.base_domain.b
        C = proj_region.C
        d = proj_region.d
        n = A.shape[1]
        aabb = [0.0] * n * 2
        for i in range(n):
            c = np.zeros(n)
            c[i] = 1.0
            res_min = linprog(c, A_ub=A, b_ub=b, A_eq=C, b_eq=constraint - d, bounds=(None, None))
            res_max = linprog(-c, A_ub=A, b_ub=b, A_eq=C, b_eq=constraint - d, bounds=(None, None))
            if res_min.success and res_max.success:
                aabb[i] = res_min.x[i]
                aabb[i + n] = res_max.x[i]
            else:
                return None
        return tuple(aabb)
    except:
        return None


def _merge_intervals(intervals, tol=1e-3):
    if not intervals:
        return []
    sorted_ivs = sorted(intervals)
    merged = [sorted_ivs[0]]
    for lo, hi in sorted_ivs[1:]:
        if lo <= merged[-1][1] + tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def compute_smm_ranges(x, y, bounds=None, n_workers=None, tol=1e-3, verbose=False):
    global _CONSTRAINT

    # Validate x and y
    if not isinstance(x, (int, float)):
        raise TypeError("x must be a float")
    if not isinstance(y, (int, float)):
        raise TypeError("y must be a float")

    # Validate bounds if provided
    if bounds is not None:
        try:
            bounds = list(bounds)
            if len(bounds) != 4:
                raise ValueError("bounds must be a sequence of length 4")
            for i, b in enumerate(bounds):
                b = list(b)
                if len(b) != 2:
                    raise ValueError("each bound must be a sequence of length 2")
                bounds[i] = (float(b[0]), float(b[1]))
        except (TypeError, ValueError) as e:
            raise ValueError("bounds must be a sequence of length 4 with sequences of length 2") from e

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    constraint = np.array([x, y])
    _CONSTRAINT = constraint  # keep original global assignment (harmless), but workers do NOT rely on it

    candidates = list(_RTREE.intersection((x, y, x, y)))
    n_candidates = len(candidates)
    if verbose:
        print(f"R-tree candidates: {n_candidates}/{len(_REGIONS)}")

    valid = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_check_region_idx, i, constraint) for i in candidates]
        for i, f in enumerate(as_completed(futures)):
            res = f.result()
            if res is not None:
                valid.append(res)
            if verbose:
                print(f"\rChecking regions: {i+1}/{n_candidates}, valid: {len(valid)}", end="")
    if verbose:
        print()

    if not valid:
        return None

    n_valid = len(valid)

    # Bounded range stage (first dimension only, with bounds applied)
    bounded_result = []
    if bounds is not None:
        bounded_intervals = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_get_bounded_range, (v, bounds, constraint)) for v in valid]
            for i, f in enumerate(as_completed(futures)):
                res = f.result()
                if res is not None:
                    bounded_intervals.append(res)
                if verbose:
                    print(f"\rComputing bounded ranges: {i+1}/{n_valid}", end="")
        if verbose:
            print()
        if bounded_intervals:
            merged = _merge_intervals(bounded_intervals, tol)
            bounded_result = [(float(lo), float(hi)) for lo, hi in merged]

    if not bounded_result:
        return None

    aabbs = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_get_aabb, (v, constraint)) for v in valid]
        for i, f in enumerate(as_completed(futures)):
            res = f.result()
            if res is not None:
                aabbs.append(res)
            if verbose:
                print(f"\rComputing AABBs: {i+1}/{n_valid}", end="")
    if verbose:
        print()

    if not aabbs:
        return None

    n_dims = len(aabbs[0]) // 2
    aabb_results = []
    for dim in range(n_dims):
        intervals = [(aabb[dim], aabb[dim + n_dims]) for aabb in aabbs]
        merged = _merge_intervals(intervals, tol)
        aabb_results.append([(float(lo), float(hi)) for lo, hi in merged])

    if not aabb_results or any(not r for r in aabb_results):
        return None

    return [bounded_result] + aabb_results


"""
if __name__ == '__main__':
    bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    ranges = compute_smm_ranges(3.0, 0.0, bounds=bounds, n_workers=1, verbose=True)
    print(ranges)
"""
