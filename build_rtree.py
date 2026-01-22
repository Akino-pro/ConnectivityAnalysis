import multiprocessing as mp
mp.set_start_method('fork', force=True)

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import pickle
from scipy.optimize import linprog
from rtree import index

_REGIONS = None


def _output_bounds_idx(idx):
    region = _REGIONS[idx]
    A, b = region.domain.A, region.domain.b
    C, d = region.C, region.d
    m = C.shape[0]
    bounds = []
    for i in range(m):
        res_min = linprog(C[i], A_ub=A, b_ub=b, bounds=(None, None))
        res_max = linprog(-C[i], A_ub=A, b_ub=b, bounds=(None, None))
        if res_min.success and res_max.success:
            bounds.append((res_min.fun + d[i], -res_max.fun + d[i]))
        else:
            return idx, None
    return idx, bounds


def build_rtree(regions, path=None, n_workers=None, verbose=False):
    global _REGIONS
    _REGIONS = regions

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    n = len(regions)
    all_bounds = [None] * n

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_output_bounds_idx, i) for i in range(n)]
        for i, f in enumerate(as_completed(futures)):
            idx, bounds = f.result()
            all_bounds[idx] = bounds
            if verbose:
                print(f"\rComputing bounds: {i+1}/{n}", end="")
    if verbose:
        print()

    p = index.Property()
    p.dimension = regions[0].m
    if path:
        idx = index.Index(path, properties=p)
    else:
        idx = index.Index(properties=p)

    for i, bounds in enumerate(all_bounds):
        if bounds is None:
            continue
        mins = [b[0] for b in bounds]
        maxs = [b[1] for b in bounds]
        idx.insert(i, tuple(mins + maxs))

    if verbose:
        print(f"Inserted {sum(1 for b in all_bounds if b)} regions into R-tree")
    return idx


def load_rtree(path, dimension=2):
    p = index.Property()
    p.dimension = dimension
    return index.Index(path, properties=p)


if __name__ == '__main__':
    with open('./data/4R_atlas.pkl', 'rb') as f:
        regions = pickle.load(f)

    rtree = build_rtree(regions, path='./data/4R_rtree', verbose=True)
    rtree.close()
