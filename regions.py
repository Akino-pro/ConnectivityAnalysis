from collections.abc import Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import null_space

import polytope as poly


class ProjectedLinearOutputRegion:

    __array_priority__ = 5

    def __init__(self, base_region, z, _reduce=False, _validate=False):

        self.base_domain = base_region.domain

        self._C = base_region.C
        self._d = base_region.d

        C_inv = np.linalg.pinv(self._C)
        N = np.atleast_1d(null_space(self._C))
        self._N = N
        self._Projector = self._N @ np.linalg.inv(self._N.T @ self._N) @ self._N.T

        assert z.shape[0] == self.m
        assert z.ndim <= 1
        self._z = np.atleast_1d(z)

        q0 = C_inv @ (self._z - self._d)
        if _validate:
            if not np.all(np.isclose((self._C @ q0) + self._d, z)):
                raise ValueError("unobtainable output")
        self._q0 = q0

        D = self.base_domain.A @ self._N
        B = self.base_domain.b - (self.base_domain.A @ self._q0)

        domain = poly.Poly(D, B)

        if _validate and poly.poly_empty(domain):
            raise ValueError("empty projection")

        if _reduce:
            self.domain = poly.poly_rem_redundant(domain)
        else:
            self.domain = domain

        self._aabb = None

    def __bool__(self):
        raise NotImplementedError("chained inequalities are not implemented")

    def __le__(self, z):
        if isinstance(z, np.ndarray):
            assert z.ndim == 1
            assert self.m == z.shape[0]

            A_hat = np.vstack(
                [
                    self.base_domain.A,
                    np.atleast_2d(self._C),
                ]
            )

            b_hat = np.hstack(
                [
                    self.base_domain.b,
                    np.atleast_1d(z - self._d),
                ]
            )

            domain = LinearOutputRegion(A_hat, b_hat, self._C, self._d)

            return ProjectedLinearOutputRegion(domain, self._z)

        raise NotImplementedError(
            "projected linear output upper bound can only be set for numpy arrays"
        )

    def __ge__(self, z):
        if isinstance(z, np.ndarray):
            assert z.ndim == 1
            assert self.m == z.shape[0]

            A_hat = np.vstack(
                [
                    self.base_domain.A,
                    np.atleast_2d(-self._C),
                ]
            )

            b_hat = np.hstack(
                [
                    self.base_domain.b,
                    np.atleast_1d(self._d - z),
                ]
            )

            domain = LinearOutputRegion(A_hat, b_hat, self._C, self._d)

            return ProjectedLinearOutputRegion(domain, self._z)

        raise NotImplementedError(
            "projected linear output lower bound can only be set for numpy arrays"
        )

    def reflect(self, alpha):
        assert alpha.shape[-1] == self.proj_n, "alpha and nullspace dimension mis-match"
        assert alpha.ndim <= 2, "alpha should be a vector or a batch of vectors"
        if alpha.ndim == 1:
            return self._q0 + self._N @ alpha
        elif alpha.ndim == 2:
            return self._q0[None] + (self._N @ alpha.T).T
        else:
            raise ValueError(
                f"only supports 1 or 2 dimensional inputs, not {alpha.ndim}"
            )

    @property
    def C(self):
        return np.copy(self._C)

    @property
    def d(self):
        return np.copy(self._d)

    @property
    def n(self):
        return self.base_domain.n

    @property
    def proj_n(self):
        return self.domain.n

    @property
    def m(self):
        return self._C.shape[0]

    @property
    def aabb(self):
        if self._aabb is not None:
            return self._aabb

        self._aabb = poly.poly_aabb(self.domain)
        return self._aabb

    def __call__(self, alpha):
        assert isinstance(alpha, np.ndarray)
        q = self.reflect(alpha)
        if alpha.ndim == 1:
            return self._C @ q + self._d
        elif alpha.ndim == 2:
            return (self._C @ q.T).T + self._d[None]
        else:
            raise ValueError(
                f"only supports 1 or 2 dimensional inputs, not {alpha.ndim}"
            )

    def project(self, q):
        return self._q0 + self._Projector @ (q - self._q0)


class LinearOutputRegion:

    __array_priority__ = 5

    def __init__(self, A, b, C, d, _reduce=False, _validate=False):

        domain = poly.Poly(A, b)

        if _validate and poly.poly_empty(domain):
            raise ValueError("empty linear output region")

        if _reduce:
            self.domain = poly.poly_rem_redundant(domain)
        else:
            self.domain = domain

        assert (
            C.shape[0] == d.shape[0]
        ), "output dimension of output variables is inconsistent"
        assert (C.ndim <= 2) and (d.ndim <= 1), "incorrect output variable dimensions"
        if C.ndim == 1:
            C = np.atleast_2d(C)
        if d.ndim == 0:
            d = np.atleast_1d(d)

        self._C = C
        self._d = d

    def __getitem__(self, inds):
        if isinstance(inds, slice):
            inds = list(range(inds.start or 0, inds.stop or self.m, inds.step or 1))

        assert any(isinstance(inds, check_type) for check_type in [int, Iterable])
        if isinstance(inds, Iterable):
            assert all(isinstance(ind, int) for ind in inds)
            assert all(0 <= ind < self.m for ind in inds)
            inds_sorted = sorted(inds)
            assert all(
                ind_i < ind_j for ind_i, ind_j in zip(inds_sorted[:-1], inds_sorted[1:])
            )

            return LinearOutputRegion(
                self.domain.A, self.domain.b, self._C[inds], self._d[inds]
            )

        elif isinstance(inds, int):
            assert 0 <= inds < self.m
            return LinearOutputRegion(
                self.domain.A,
                self.domain.b,
                np.atleast_2d(self._C[inds]),
                np.atleast_1d(self._d[inds]),
            )

        else:
            raise ValueError(f"cannot reduce cost dimension with type: {type(inds)}")

    def __eq__(self, z):
        if isinstance(z, np.ndarray):
            assert self.n >= self.m
            assert self._d.shape[0] == z.shape[0]
            return ProjectedLinearOutputRegion(self, z)
        raise NotImplementedError(
            "linear output equality can only be set for numpy arrays"
        )

    def __bool__(self):
        raise NotImplementedError("chained inequalities are not implemented")

    def __le__(self, z):
        if isinstance(z, np.ndarray):
            assert z.ndim == 1
            assert self.m == z.shape[0]

            A_hat = np.vstack(
                [
                    self.domain.A,
                    np.atleast_2d(self._C),
                ]
            )

            b_hat = np.hstack(
                [
                    self.domain.b,
                    np.atleast_1d(z - self._d),
                ]
            )

            return LinearOutputRegion(A_hat, b_hat, self._C, self._d)

        raise NotImplementedError(
            "linear output upper bound can only be set for numpy arrays"
        )

    def __ge__(self, z):
        if isinstance(z, np.ndarray):
            assert z.ndim == 1
            assert self.m == z.shape[0]

            A_hat = np.vstack(
                [
                    self.domain.A,
                    np.atleast_2d(-self._C),
                ]
            )

            b_hat = np.hstack(
                [
                    self.domain.b,
                    np.atleast_1d(self._d - z),
                ]
            )

            return LinearOutputRegion(A_hat, b_hat, self._C, self._d)

        raise NotImplementedError(
            "linear output lower bound can only be set for numpy arrays"
        )

    @property
    def C(self):
        return np.copy(self._C)

    @property
    def d(self):
        return np.copy(self._d)

    @property
    def n(self):
        return self.domain.n

    @property
    def m(self):
        return self._C.shape[0]

    @property
    def aabb(self):
        return poly.poly_aabb(self.domain)

    def __call__(self, q):
        assert self._C.shape[1] == q.shape[0], "output function dimension mis-match"
        return self._C @ q + self._d


if __name__ == "__main__":
    A = np.random.uniform(-1, 1, (100, 4))
    b = np.ones((100,))

    C = np.random.uniform(-1, 1, (2, 4))
    d = np.zeros((2))

    reg = LinearOutputRegion(A, b, C, d)
    proj_reg = reg == np.zeros((2,))

    qs = poly.poly_to_vert(proj_reg.domain)
    # print(qs)
    for q in qs:
        print(proj_reg.reflect(q))
        # print(proj_reg(q))

    lte_reg = reg <= np.zeros((2,))
    print(lte_reg.domain.A.shape)
    lte_reg = np.zeros((2,)) <= lte_reg
    print(lte_reg.domain.A.shape)
    print(poly.poly_aabb(lte_reg.domain))
