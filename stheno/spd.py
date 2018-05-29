# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from lab import B

__all__ = ['SPD']


class SPD(object):
    """Symmetric positive-definite matrix.

    Args:
        mat (matrix): Symmetric positive-definite matrix.
    """

    def __init__(self, mat):
        self.mat = mat
        self._cholesky = None
        self._log_det = None
        self._root = None

    def cholesky(self):
        """Compute the Cholesky decomposition."""
        if self._cholesky is None:
            self._cholesky = B.cholesky(B.reg(self.mat))
        return self._cholesky

    def log_det(self):
        """Compute the log-determinant."""
        if self._log_det is None:
            self._log_det = 2 * B.sum(B.log(B.diag(self.cholesky())))
        return self._log_det

    def mah_dist2(self, a, b=None):
        """Compute the square of the Mahalanobis distance between two matrices.

        Args:
            a (matrix): First matrix.
            b (matrix): Second matrix. Defaults to first matrix.
        """
        diff = a if b is None else a - b
        iL_diff = B.trisolve(self.cholesky(), diff)
        return B.sum(iL_diff ** 2)

    def ratio(self, denom):
        """Compute the ratio with respect to another positive-definite matrix.

        Args:
            denom (instance of :class:`.pdmat.PDMat`): Denominator in the
                ratio.
        """
        return B.sum(B.trisolve(denom.cholesky(), self.cholesky()) ** 2)

    def quadratic_form(self, a, b=None):
        """Compute the quadratic form `transpose(a) inv(self.mat) b`.

        Args:
            a (matrix): `a`.
            b (matrix): `b`.
        """
        if b is None:
            return self.mah_dist2(a)
        else:
            left = B.trisolve(self.cholesky(), a)
            right = B.trisolve(self.cholesky(), b)
            return B.dot(left, right, tr_a=True)

    def inv_prod(self, a):
        """Compute the matrix-vector product `inv(self.mat) a`.

        Args:
            a (matrix): `a`
        """
        return B.trisolve(self.cholesky(), B.trisolve(self.cholesky(), a),
                           tr_a=True)

    def root(self):
        """Compute a square root."""
        if self._root is None:
            vals, vecs = B.eig(B.reg(self.mat))
            self._root = B.dot(vecs * vals[None, :] ** .5, vecs, tr_b=True)
        return self._root
