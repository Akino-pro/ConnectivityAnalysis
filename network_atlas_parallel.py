"""
Parallel Layer-wise Neural Network Cell Enumeration

Implementation of the algorithm from:
"Parallel Algorithms for Neural Network Activation Region Enumeration"
(arXiv:2403.00860)

This implements the ParLayerWise1-NNCE algorithm which:
1. Enumerates all first-layer sign vectors exhaustively (2^n1 possibilities)
2. For each valid first-layer cell, incrementally enumerates deeper layer cells
3. Parallelizes across first-layer cells using a worker pool
"""

import numpy as np
from scipy.optimize import linprog as scipy_linprog
from scipy.linalg import null_space
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import itertools
from collections import deque

import networkx as nx
import torch.nn as nn

from regions import LinearOutputRegion
import polytope as poly
import numbotics.learning as lrn

import time


@dataclass
class LayerInfo:
    """Information about a single layer's hyperplanes."""
    W: np.ndarray  # Weight matrix (n_neurons, input_dim)
    b: np.ndarray  # Bias vector (n_neurons,)
    alpha: np.ndarray  # PReLU/LeakyReLU slopes


@dataclass
class CellInfo:
    """Information about a cell (activation region)."""
    sign_vector: np.ndarray  # Full sign pattern across all layers
    A: np.ndarray  # Constraint matrix
    b: np.ndarray  # Constraint bounds
    C: np.ndarray  # Output linear transform
    d: np.ndarray  # Output bias


def check_cell_feasibility(A: np.ndarray, b: np.ndarray) -> bool:
    """
    Check if polytope {x : Ax <= b} is non-empty using scipy LP (HiGHS).

    Returns:
        True if feasible, False otherwise
    """
    n = A.shape[1]
    c = np.zeros(n)

    try:
        res = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method='highs')
        return res.success
    except:
        return False


def get_layer_constraints(
    W: np.ndarray,
    b: np.ndarray,
    signs: np.ndarray,
    W_hat: np.ndarray,
    b_hat: np.ndarray,
    alpha: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get constraints for a layer given sign pattern and accumulated transform.

    The pre-activation is: z = W @ (W_hat @ x + b_hat) + b
    Sign pattern determines: z >= 0 (sign=1) or z < 0 (sign=0)

    Returns:
        A_layer: Constraint matrix for this layer
        b_layer: Constraint bounds for this layer
        W_hat_new: Updated accumulated weight matrix
        b_hat_new: Updated accumulated bias
    """
    n_neurons = W.shape[0]

    # Accumulated transform: current input to layer is W_hat @ x + b_hat
    # Pre-activation: z = W @ (W_hat @ x + b_hat) + b = (W @ W_hat) @ x + (W @ b_hat + b)
    W_acc = W @ W_hat
    b_acc = W @ b_hat + b

    # Build constraints based on sign pattern
    # sign=1: z >= 0  =>  -W_acc @ x <= b_acc
    # sign=0: z < 0   =>   W_acc @ x <= -b_acc (strict, but we use <=)
    A_layer = np.zeros((n_neurons, W_hat.shape[1]))
    b_layer = np.zeros(n_neurons)

    for i in range(n_neurons):
        if signs[i] == 1:
            A_layer[i] = -W_acc[i]
            b_layer[i] = b_acc[i]
        else:
            A_layer[i] = W_acc[i]
            b_layer[i] = -b_acc[i]

    # Update accumulated transform through activation
    # Post-activation: a = max(z, alpha * z) = D @ z where D = diag(sign ? 1 : alpha)
    D = np.diag(np.where(signs == 1, 1.0, alpha))
    W_hat_new = D @ W_acc
    b_hat_new = D @ b_acc

    return A_layer, b_layer, W_hat_new, b_hat_new


class ParallelNetworkAtlas:
    """
    Parallel layer-wise neural network cell enumeration.

    Implements the ParLayerWise1-NNCE algorithm from arXiv:2403.00860.
    """

    def __init__(
        self,
        A_weights: List[np.ndarray],
        b_weights: List[np.ndarray],
        C_weight: np.ndarray,
        d_weight: np.ndarray,
        alphas: List[np.ndarray],
        l_bnd: np.ndarray,
        u_bnd: np.ndarray
    ):
        """
        Args:
            A_weights: List of weight matrices for each hidden layer
            b_weights: List of bias vectors for each hidden layer
            C_weight: Output layer weight matrix
            d_weight: Output layer bias
            alphas: PReLU/LeakyReLU negative slopes for each layer
            l_bnd: Lower bounds on input domain
            u_bnd: Upper bounds on input domain
        """
        self.layers = []
        for W, b, alpha in zip(A_weights, b_weights, alphas):
            alpha_arr = np.atleast_1d(alpha).astype(np.float64)
            if alpha_arr.shape[0] == 1:
                alpha_arr = np.full(W.shape[0], alpha_arr[0])
            self.layers.append(LayerInfo(
                W=W.astype(np.float64),
                b=b.astype(np.float64),
                alpha=alpha_arr
            ))

        self.C_weight = C_weight.astype(np.float64)
        self.d_weight = d_weight.astype(np.float64)
        self.l_bnd = np.atleast_1d(l_bnd).astype(np.float64)
        self.u_bnd = np.atleast_1d(u_bnd).astype(np.float64)

        self.n_input = self.layers[0].W.shape[1]
        self.n_layers = len(self.layers)
        self.neurons_per_layer = [layer.W.shape[0] for layer in self.layers]
        self.total_neurons = sum(self.neurons_per_layer)

    @property
    def n(self):
        return self.n_input

    @property
    def w_dim(self):
        return self.total_neurons

    def _get_box_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get box constraints for input domain."""
        n = self.n_input
        A_box = np.vstack([np.eye(n), -np.eye(n)])
        b_box = np.hstack([self.u_bnd, -self.l_bnd])
        return A_box, b_box

    def _build_full_constraints(
        self,
        sign_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build full constraint system for a complete sign vector.

        Returns:
            A: Full constraint matrix
            b: Full constraint bounds
            C: Output transform matrix
            d: Output transform bias
        """
        A_box, b_box = self._get_box_constraints()

        A_layers = []
        b_layers = []

        W_hat = np.eye(self.n_input)
        b_hat = np.zeros(self.n_input)

        idx = 0
        for layer in self.layers:
            n_neurons = layer.W.shape[0]
            signs = sign_vector[idx:idx + n_neurons]

            A_layer, b_layer, W_hat, b_hat = get_layer_constraints(
                layer.W, layer.b, signs, W_hat, b_hat, layer.alpha
            )

            A_layers.append(A_layer)
            b_layers.append(b_layer)
            idx += n_neurons

        A = np.vstack(A_layers + [A_box])
        b = np.hstack(b_layers + [b_box])

        C = self.C_weight @ W_hat
        d = self.C_weight @ b_hat + self.d_weight

        return A, b, C, d

    def _enumerate_first_layer_cells_bruteforce(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Enumerate all valid first-layer cells using brute-force enumeration.
        Checks all 2^n1 possible sign patterns.

        Returns:
            List of (sign_vector, A, b, W_hat, b_hat) for each valid first-layer cell
        """
        n1 = self.neurons_per_layer[0]
        layer = self.layers[0]
        A_box, b_box = self._get_box_constraints()

        valid_cells = []

        # Enumerate all 2^n1 possible sign patterns
        for signs_tuple in itertools.product([0, 1], repeat=n1):
            signs = np.array(signs_tuple, dtype=np.float64)

            W_hat = np.eye(self.n_input)
            b_hat = np.zeros(self.n_input)

            A_layer, b_layer, W_hat_new, b_hat_new = get_layer_constraints(
                layer.W, layer.b, signs, W_hat, b_hat, layer.alpha
            )

            A = np.vstack([A_layer, A_box])
            b = np.hstack([b_layer, b_box])

            if check_cell_feasibility(A, b):
                valid_cells.append((signs, A, b, W_hat_new, b_hat_new))

        return valid_cells

    def _enumerate_first_layer_cells_bfs(self, verbose: bool = False) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Enumerate valid first-layer cells using BFS from an interior point.
        Only discovers reachable cells by walking through neighbors.

        Returns:
            List of (sign_vector, A, b, W_hat, b_hat) for each valid first-layer cell
        """
        n1 = self.neurons_per_layer[0]
        layer = self.layers[0]
        A_box, b_box = self._get_box_constraints()

        def signs_to_cell_data(signs: np.ndarray):
            """Convert sign pattern to (signs, A, b, W_hat, b_hat) tuple."""
            W_hat = np.eye(self.n_input)
            b_hat = np.zeros(self.n_input)

            A_layer, b_layer, W_hat_new, b_hat_new = get_layer_constraints(
                layer.W, layer.b, signs, W_hat, b_hat, layer.alpha
            )

            A = np.vstack([A_layer, A_box])
            b = np.hstack([b_layer, b_box])

            return (signs, A, b, W_hat_new, b_hat_new)

        def point_to_first_layer_signs(x: np.ndarray) -> np.ndarray:
            """Get first-layer sign pattern for a point."""
            z = layer.W @ x + layer.b
            return np.where(z >= 0, 1.0, 0.0)

        # Start from center of input domain
        x_init = (self.l_bnd + self.u_bnd) / 2
        signs_init = point_to_first_layer_signs(x_init)

        # BFS setup
        found = {tuple(signs_init.astype(int))}
        queue = deque([signs_init])
        valid_cells = []

        # Add initial cell
        cell_data = signs_to_cell_data(signs_init)
        if check_cell_feasibility(cell_data[1], cell_data[2]):
            valid_cells.append(cell_data)

        rejected = 0

        while queue:
            if verbose:
                print(f"  BFS queue: {len(queue)}, found: {len(valid_cells)}, rejected: {rejected}\r", end="")

            curr_signs = queue.popleft()

            # Try flipping each bit (neighbor = flip one neuron's sign)
            for i in range(n1):
                neighbor_signs = curr_signs.copy()
                neighbor_signs[i] = 1.0 - neighbor_signs[i]

                neighbor_key = tuple(neighbor_signs.astype(int))

                if neighbor_key not in found:
                    found.add(neighbor_key)

                    cell_data = signs_to_cell_data(neighbor_signs)

                    if check_cell_feasibility(cell_data[1], cell_data[2]):
                        valid_cells.append(cell_data)
                        queue.append(neighbor_signs)
                    else:
                        rejected += 1

        if verbose:
            print()

        return valid_cells

    def _enumerate_first_layer_cells(self, method: str = "bfs", verbose: bool = False) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Enumerate all valid first-layer cells.

        Args:
            method: "bfs" for BFS-based discovery (recommended),
                    "bruteforce" for exhaustive 2^n1 enumeration
            verbose: Print progress for BFS

        Returns:
            List of (sign_vector, A, b, W_hat, b_hat) for each valid first-layer cell
        """
        if method == "bruteforce":
            return self._enumerate_first_layer_cells_bruteforce()
        elif method == "bfs":
            return self._enumerate_first_layer_cells_bfs(verbose=verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bfs' or 'bruteforce'")

    def _incremental_enumerate_layer(
        self,
        layer_idx: int,
        parent_A: np.ndarray,
        parent_b: np.ndarray,
        W_hat: np.ndarray,
        b_hat: np.ndarray,
        parent_signs: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Incrementally enumerate cells for a layer given parent cell.

        This implements Bound-IncEnum from the paper: add hyperplanes one at a time
        and track how existing cells split.

        Returns:
            List of (full_signs, A, b, W_hat_new, b_hat_new) for each valid cell
        """
        layer = self.layers[layer_idx]
        n_neurons = layer.W.shape[0]

        # Pre-activation: z_i = (W @ W_hat) @ x + (W @ b_hat + b)
        W_acc = layer.W @ W_hat
        b_acc = layer.W @ b_hat + layer.b

        # Start with the parent cell (no new constraints yet)
        # Each cell is: (partial_signs, A, b)
        # partial_signs[i] is the sign for neuron i, or -1 if not yet determined
        initial_signs = np.full(n_neurons, -1, dtype=np.float64)
        cells = [(initial_signs, parent_A.copy(), parent_b.copy())]

        # Add hyperplanes one at a time
        for i in range(n_neurons):
            new_cells = []
            h_pos = W_acc[i]  # z_i >= 0 constraint: -h @ x <= b_acc[i]
            h_neg = -W_acc[i]  # z_i < 0 constraint: h @ x <= -b_acc[i]

            for partial_signs, A, b in cells:
                # Try positive side (sign = 1): z_i >= 0
                A_pos = np.vstack([A, -h_pos.reshape(1, -1)])
                b_pos = np.hstack([b, [b_acc[i]]])

                if check_cell_feasibility(A_pos, b_pos):
                    signs_pos = partial_signs.copy()
                    signs_pos[i] = 1.0
                    new_cells.append((signs_pos, A_pos, b_pos))

                # Try negative side (sign = 0): z_i < 0
                A_neg = np.vstack([A, -h_neg.reshape(1, -1)])
                b_neg = np.hstack([b, [-b_acc[i]]])

                if check_cell_feasibility(A_neg, b_neg):
                    signs_neg = partial_signs.copy()
                    signs_neg[i] = 0.0
                    new_cells.append((signs_neg, A_neg, b_neg))

            cells = new_cells

            if not cells:
                break

        # Build output for each valid cell
        results = []
        for layer_signs, A, b in cells:
            full_signs = np.concatenate([parent_signs, layer_signs])

            # Compute new accumulated transform
            D = np.diag(np.where(layer_signs == 1, 1.0, layer.alpha))
            W_hat_new = D @ W_acc
            b_hat_new = D @ b_acc

            results.append((full_signs, A, b, W_hat_new, b_hat_new))

        return results

    def _enumerate_from_first_layer_cell(
        self,
        first_layer_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> List[np.ndarray]:
        """
        Enumerate all cells reachable from a first-layer cell.

        Args:
            first_layer_data: (signs, A, b, W_hat, b_hat) from first layer

        Returns:
            List of sign_vectors
        """
        signs_l1, A, b, W_hat, b_hat = first_layer_data

        # Start with first layer cell (signs, A, b, W_hat, b_hat)
        current_cells = [(signs_l1, A, b, W_hat, b_hat)]

        # Process remaining layers
        for layer_idx in range(1, self.n_layers):
            next_cells = []

            for parent_signs, parent_A, parent_b, parent_W_hat, parent_b_hat in current_cells:
                layer_cells = self._incremental_enumerate_layer(
                    layer_idx, parent_A, parent_b, parent_W_hat, parent_b_hat, parent_signs
                )
                next_cells.extend(layer_cells)

            current_cells = next_cells

            if not current_cells:
                break

        # Return sign vectors
        return [cell[0] for cell in current_cells]

    def get_all_charts_parallel(
        self,
        n_workers: Optional[int] = None,
        return_valid_inds: bool = False,
        verbose: bool = True,
        first_layer_method: str = "bfs"
    ) -> List[LinearOutputRegion]:
        """
        Enumerate all activation regions using parallel layer-wise algorithm.

        Args:
            n_workers: Number of parallel workers (default: cpu_count)
            return_valid_inds: If True, also return sign vectors
            verbose: Print progress
            first_layer_method: Method for first-layer enumeration:
                - "bfs": BFS from center point (recommended, avoids 2^n1 blowup)
                - "bruteforce": Exhaustive 2^n1 enumeration (only for small n1)

        Returns:
            List of LinearOutputRegion objects
        """
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)

        start_time = time.time()

        # Phase 1: Enumerate first-layer cells
        if verbose:
            n1 = self.neurons_per_layer[0]
            if first_layer_method == "bruteforce":
                print(f"Phase 1: Enumerating first-layer cells (bruteforce: 2^{n1} = {2**n1} patterns)...")
            else:
                print(f"Phase 1: Enumerating first-layer cells (BFS from center, n1={n1} neurons)...")

        first_layer_cells = self._enumerate_first_layer_cells(method=first_layer_method, verbose=verbose)

        if verbose:
            print(f"  Found {len(first_layer_cells)} valid first-layer cells")

        # Phase 2: Parallel enumeration of deeper layers
        if verbose:
            print(f"Phase 2: Enumerating deeper layers with {n_workers} workers...")

        all_sign_vectors = []

        if n_workers == 1 or len(first_layer_cells) == 1:
            # Sequential execution
            for i, cell_data in enumerate(first_layer_cells):
                cell_results = self._enumerate_from_first_layer_cell(cell_data)
                all_sign_vectors.extend(cell_results)
                if verbose:
                    print(f"  Processed {i+1}/{len(first_layer_cells)} first-layer cells, "
                          f"total regions: {len(all_sign_vectors)}\r", end="")
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self._enumerate_from_first_layer_cell, cell_data): i
                    for i, cell_data in enumerate(first_layer_cells)
                }

                completed = 0
                for future in as_completed(futures):
                    cell_results = future.result()
                    all_sign_vectors.extend(cell_results)
                    completed += 1
                    if verbose:
                        print(f"  Processed {completed}/{len(first_layer_cells)} first-layer cells, "
                              f"total regions: {len(all_sign_vectors)}\r", end="")

        if verbose:
            print()

        # Phase 3: Convert sign vectors to LinearOutputRegion objects
        if verbose:
            print(f"Phase 3: Building {len(all_sign_vectors)} region objects...")

        regions = []
        valid_sign_vectors = []
        for sign_vector in all_sign_vectors:
            A, b, C, d = self._build_full_constraints(sign_vector)
            try:
                region = LinearOutputRegion(A, b, C, d)
                regions.append(region)
                valid_sign_vectors.append(sign_vector)
            except:
                pass

        elapsed = time.time() - start_time
        if verbose:
            print(f"Done! Found {len(regions)} regions in {elapsed:.2f}s")

        if return_valid_inds:
            return regions, valid_sign_vectors
        return regions

    def get_all_charts(self, return_valid_inds: bool = False, use_active_constraints: bool = False):
        """
        Compatibility method - calls parallel version.
        """
        return self.get_all_charts_parallel(
            n_workers=1,  # Use single worker for compatibility
            return_valid_inds=return_valid_inds,
            verbose=True
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        _x = np.copy(x)
        for layer in self.layers:
            z = layer.W @ _x + layer.b
            _x = np.where(z >= 0, z, layer.alpha * z)
        return self.C_weight @ _x + self.d_weight

    def point_to_ind(self, x: np.ndarray) -> np.ndarray:
        """Get activation pattern (sign vector) for a point."""
        _x = np.copy(x)
        ind = []
        for layer in self.layers:
            z = layer.W @ _x + layer.b
            ind.append(np.where(z >= 0, 1.0, 0.0))
            _x = np.where(z >= 0, z, layer.alpha * z)
        return np.concatenate(ind)

    def ind_to_poly(self, ind: np.ndarray, _reduce: bool = False, _validate: bool = False):
        """Convert sign vector to LinearOutputRegion."""
        A, b, C, d = self._build_full_constraints(ind)

        try:
            return LinearOutputRegion(A, b, C, d, _reduce=_reduce, _validate=_validate)
        except ValueError:
            return None


def get_network_info_parallel(model_name: str, l_bnd, u_bnd) -> ParallelNetworkAtlas:
    """
    Load a trained network and create a ParallelNetworkAtlas.

    Args:
        model_name: Path to saved model
        l_bnd: Lower bounds on input domain
        u_bnd: Upper bounds on input domain

    Returns:
        ParallelNetworkAtlas instance
    """
    net = lrn.FeedforwardNet.load(model_name)

    L = (sum(1 for _ in net.model) - 1) // 2

    A_weights = []
    b_weights = []
    alphas = []

    for layer in range(L):
        A_weights.append(lrn.get_lin_weight(net, layer))
        b_weights.append(lrn.get_lin_bias(net, layer))

        activ = net.model._modules[f"activ{layer}"]
        if isinstance(activ, nn.PReLU):
            alphas.append(activ.weight.cpu().data.numpy())
        elif isinstance(activ, nn.LeakyReLU):
            alphas.append(activ.negative_slope)
        else:
            raise ValueError(f"Unknown activation: {activ}")

    C_weight = lrn.get_lin_weight(net, L)
    d_weight = lrn.get_lin_bias(net, L)

    return ParallelNetworkAtlas(
        A_weights, b_weights, C_weight, d_weight, alphas, l_bnd, u_bnd
    )


if __name__ == "__main__":
    import numbotics.robot as rob
    import pickle
    import torch

    # Test configuration
    arm_name = '4R_planar'
    example = '_ik'

    arm = rob.Robot(f"../arms/{arm_name}.rob")
    ll = [-np.pi for _ in range(arm.n)]
    ul = [np.pi for _ in range(arm.n)]

    print(f"Testing parallel enumeration for {arm_name}{example}")
    print(f"Input dimension: {arm.n}")
    print()

    # Load network
    atlas = get_network_info_parallel(
        f"../models/{arm_name}{example}_net.pt",
        ll, ul
    )

    regions = atlas.get_all_charts_parallel(n_workers=10)

    with open('./data/4R_atlas.pkl', 'wb') as f:
        pickle.dump(regions, f, protocol=4)

    print(f"Network: {atlas.n_layers} layers, neurons per layer: {atlas.neurons_per_layer}")
    print(f"Total neurons: {atlas.total_neurons}")
    print()
