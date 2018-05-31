# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from lab import B
from abc import ABCMeta, abstractmethod, abstractproperty
from plum import Referentiable, Self, Dispatcher

__all__ = ['SPD', 'Dense', 'Diagonal', 'UniformDiagonal']


class SPD(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def dtype(self):
        """Data type of the matrix"""

    @abstractproperty
    def shape(self):
        """Shape of the matrix"""

    @abstractproperty
    def mat(self):
        """The matrix"""

    @abstractproperty
    def diag(self):
        """Diagonal of the matrix"""

    @abstractmethod
    def cholesky(self):
        """Compute the Cholesky decomposition."""

    @abstractmethod
    def log_det(self):
        """Compute the log-determinant."""

    @abstractmethod
    def mah_dist2(self, a, b=None, sum=True):
        """Compute the square of the Mahalanobis distance between two matrices.

        Args:
            a (matrix): First matrix.
            b (matrix): Second matrix. Defaults to first matrix.
            sum (bool, optional): Compute the sum of all distances instead
                of returning all distances.
        """

    @abstractmethod
    def ratio(self, denom):
        """Compute the ratio with respect to another positive-definite matrix.

        Args:
            denom (instance of :class:`.pdmat.PDMat`): Denominator in the
                ratio.
        """

    @abstractmethod
    def quadratic_form(self, a, b=None):
        """Compute the quadratic form `transpose(a) inv(self.mat) b`.

        Args:
            a (matrix): `a`.
            b (matrix): `b`.
        """

    @abstractmethod
    def inv_prod(self, a):
        """Compute the matrix-vector product `inv(self.mat) a`.

        Args:
            a (matrix): `a`
        """

    @abstractmethod
    def root(self):
        """Compute a square root."""


class Dense(SPD):
    """Dense symmetric positive-definite matrix.

    Args:
        mat (matrix): Symmetric positive-definite matrix.
    """

    def __init__(self, mat):
        self._mat = mat
        self._cholesky = None
        self._log_det = None
        self._root = None

    @property
    def dtype(self):
        return self._mat.dtype

    @property
    def mat(self):
        return self._mat

    @property
    def diag(self):
        return B.diag(self._mat)

    @property
    def shape(self):
        return B.shape(self.mat)

    def cholesky(self):
        if self._cholesky is None:
            self._cholesky = B.cholesky(B.reg(self.mat))
        return self._cholesky

    def log_det(self):
        if self._log_det is None:
            self._log_det = 2 * B.sum(B.log(B.diag(self.cholesky())))
        return self._log_det

    def mah_dist2(self, a, b=None, sum=True):
        diff = a if b is None else a - b
        iL_diff = B.trisolve(self.cholesky(), diff)
        return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)

    def ratio(self, denom):
        return B.sum(B.trisolve(denom.cholesky(), self.cholesky()) ** 2)

    def quadratic_form(self, a, b=None):
        if b is None:
            return self.mah_dist2(a)
        else:
            left = B.trisolve(self.cholesky(), a)
            right = B.trisolve(self.cholesky(), b)
            return B.dot(left, right, tr_a=True)

    def inv_prod(self, a):
        return B.cholesky_solve(self.cholesky(), a)

    def root(self):
        if self._root is None:
            vals, vecs = B.eig(B.reg(self.mat))
            self._root = B.dot(vecs * vals[None, :] ** .5, vecs, tr_b=True)
        return self._root


class Diagonal(Dense, Referentiable):
    """Diagonal symmetric positive-definite matrix.

    Args:
        diag (vector): Diagonal of the matrix.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag):
        Dense.__init__(self, None)
        self._diag = diag

    @property
    def dtype(self):
        return self._diag.dtype

    @property
    def diag(self):
        return self._diag

    @property
    def shape(self):
        return B.shape(self.diag)[0], B.shape(self.diag)[0]

    @property
    def mat(self):
        return B.diag(self.diag)

    def cholesky(self):
        return self.mat ** .5

    def log_det(self):
        return B.sum(B.log(self.diag))

    def mah_dist2(self, a, b=None, sum=True):
        diff = a if b is None else a - b
        iL_diff = diff / self.diag[:, None] ** .5
        return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)

    @dispatch(Self)
    def ratio(self, denom):
        return B.sum(self.diag / denom.diag)

    def quadratic_form(self, a, b=None):
        if b is None:
            return self.mah_dist2(a)
        else:
            isqrt_diag = self.diag[:, None] ** .5
            return B.dot(a / isqrt_diag, b / isqrt_diag, tr_a=True)

    def inv_prod(self, a):
        return a / self.diag[:, None]

    def root(self):
        return self.cholesky()


class UniformDiagonal(Diagonal, Referentiable):
    """Uniformly diagonal symmetric positive-definite matrix.

    Args:
        diag_scal (vector): Scale of the diagonal of the matrix.
        n (int): Size of the the diagonal.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag_scale, n):
        Diagonal.__init__(self, None)
        self.diag_scale = diag_scale
        self._n = n

    @property
    def dtype(self):
        return self.diag_scale.dtype

    @property
    def shape(self):
        return self._n, self._n

    @property
    def mat(self):
        return B.eye(self._n) * self.diag_scale

    @property
    def diag(self):
        return B.ones(self._n) * self.diag_scale

    def log_det(self):
        return self._n * B.log(self.diag_scale)

    def mah_dist2(self, a, b=None, sum=True):
        diff = a if b is None else a - b
        iL_diff = diff / self.diag_scale ** .5
        return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)

    @dispatch(Self)
    def ratio(self, denom):
        return self._n * self.diag_scale / denom.diag_scale

    def quadratic_form(self, a, b=None):
        if b is None:
            return self.mah_dist2(a)
        else:
            return B.dot(a / self.diag_scale ** .5, b / self.diag_scale ** .5,
                         tr_a=True)

    def inv_prod(self, a):
        return a / self.diag_scale

    def root(self):
        return self.cholesky()
