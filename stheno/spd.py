# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from lab import B
from plum import Referentiable, Self, Dispatcher, Kind, PromisedType
from future.utils import with_metaclass
from .field import Element, mul, add, dispatch_field

__all__ = ['SPD', 'Diagonal', 'UniformDiagonal']

_dispatch = Dispatcher()


class SPD(Element, Referentiable):
    """Symmetric positive-definite matrix."""
    _dispatch = Dispatcher(in_class=Self)

    @property
    def dtype(self):
        """Data type."""
        raise NotImplementedError()

    @property
    def mat(self):
        """Matrix in dense form."""
        raise NotImplementedError()

    @property
    def diag(self):
        """Diagonal of the matrix."""
        return B.diag(self._mat)

    @property
    def shape(self):
        """Shape of the matrix."""
        return B.shape(self.mat)

    def cholesky(self):
        """Compute the Cholesky decomposition.

        Returns:
            tensor: Cholesky decomposition.
        """
        raise NotImplementedError()

    def cholesky_mul(self, a):
        """Multiply the Cholesky decomposition of this matrix with `a`.

        Args:
            a (tensor): Matrix to multiply with.

        Returns:
            tensor: Multiplication of the Cholesky of this matrix with `a`.
        """
        raise NotImplementedError()

    def logdet(self):
        """Compute the log-determinant.

        Returns:
            scalar: Log-determinant.
        """
        raise NotImplementedError()

    @_dispatch(object, object)
    def mah_dist2(self, a, b, sum=True):
        """Compute the square of the Mahalanobis distance between vectors.

        Args:
            a (tensor): First matrix.
            b (tensor, optional): Second matrix. If omitted, `a` is assumed
                to be the differences.
            sum (bool, optional): Compute the sum of all distances instead
                of returning all distances.

        Returns:
            tensor: Distance or distances.
        """
        raise NotImplementedError()

    @_dispatch(object)
    def mah_dist2(self, diff, sum=True):
        raise NotImplementedError()

    def ratio(self, denom):
        """Compute the ratio with respect to another positive-definite matrix.

        Args:
            denom (:class:`.spd.SPD`): Denominator in the ratio.

        Returns:
            tensor: Ratio.
        """
        raise NotImplementedError()

    @_dispatch(object)
    def quadratic_form(self, a):
        """Compute the quadratic form `transpose(a) inv(self.mat) b`.

        Args:
            a (tensor): `a`.
            b (tensor, optional): `b`. Defaults to `a`.

        Returns:
            tensor: Quadratic form.
        """
        raise NotImplementedError()

    @_dispatch(object, object)
    def quadratic_form(self, a, b):
        raise NotImplementedError()

    @_dispatch(object)
    def quadratic_form_diag(self, a):
        """Compute the diagonal of the quadratic form
        `transpose(a) inv(self.mat) b`.

        Args:
            a (tensor): `a`.
            b (tensor, optional): `b`. Defaults to `a`.

        Returns:
            tensor: Diagonal of quadratic form as a rank 1 tensor.
        """
        raise NotImplementedError()

    @_dispatch(object, object)
    def quadratic_form_diag(self, a, b):
        raise NotImplementedError()

    def inv_prod(self, a):
        """Compute the matrix-vector product `inv(self.mat) a`.

        Args:
            a (tensor): `a`

        Returns:
            tensor: Product.
        """
        raise NotImplementedError()

    def root(self):
        """Compute a square root."""
        return NotImplementedError()


class Dense(SPD, Referentiable):
    """Symmetric positive-definite matrix.

    Args:
        mat (tensor): Symmetric positive-definite matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, mat):
        self._mat = mat
        self._cholesky = None
        self._logdet = None
        self._root = None

    @property
    def dtype(self):
        return B.dtype(self._mat)

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

    def cholesky_mul(self, a):
        return B.matmul(self.cholesky(), a)

    def logdet(self):
        if self._logdet is None:
            self._logdet = 2 * B.sum(B.log(B.diag(self.cholesky())))
        return self._logdet

    @_dispatch(object, object)
    def mah_dist2(self, a, b, sum=True):
        return self.mah_dist2(a - b, sum=sum)

    @_dispatch(object)
    def mah_dist2(self, diff, sum=True):
        iL_diff = B.trisolve(self.cholesky(), diff)
        return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)

    def ratio(self, denom):
        return B.sum(B.trisolve(denom.cholesky(), self.cholesky()) ** 2)

    @_dispatch(object)
    def quadratic_form(self, a):
        prod = B.trisolve(self.cholesky(), a)
        return B.dot(prod, prod, tr_a=True)

    @_dispatch(object, object)
    def quadratic_form(self, a, b):
        left = B.trisolve(self.cholesky(), a)
        right = B.trisolve(self.cholesky(), b)
        return B.dot(left, right, tr_a=True)

    @_dispatch(object)
    def quadratic_form_diag(self, a):
        prod = B.trisolve(self.cholesky(), a)
        return B.sum(prod ** 2, axis=0)

    @_dispatch(object, object)
    def quadratic_form_diag(self, a, b):
        left = B.trisolve(self.cholesky(), a)
        right = B.trisolve(self.cholesky(), b)
        return B.sum(left * right, axis=0)

    def inv_prod(self, a):
        return B.cholesky_solve(self.cholesky(), a)

    def root(self):
        if self._cholesky is None:
            vals, vecs = B.eig(B.reg(self.mat))
            self._cholesky = B.dot(vecs * vals[None, :] ** .5, vecs, tr_b=True)
        return self._cholesky


class LowRank(Dense, Referentiable):
    """Low-rank symmetric positive-definite matrix.

    Args:
        inner (tensor): The matrix such that `inner * inner'` equals this
        matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, inner):
        Dense.__init__(self, None)
        self._inner = inner

    @property
    def dtype(self):
        return B.dtype(self._inner)

    @property
    def diag(self):
        return B.sum(self._inner ** 2, 1)

    @property
    def shape(self):
        return B.shape(self._inner)[0], B.shape(self._inner)[0]

    @property
    def mat(self):
        return B.matmul(self._inner, self._inner, tr_b=True)

    def logdet(self):
        raise RuntimeError('This matrix is singular.')

    @_dispatch(object)
    def mah_dist2(self, diff, sum=True):
        raise RuntimeError('This matrix is singular.')

    @_dispatch(SPD)
    def ratio(self, denom):
        return B.sum(self._inner * denom.inv_prod(self._inner))

    @_dispatch(object, [object])
    def quadratic_form(self, a, b=None):
        raise RuntimeError('This matrix is singular.')

    @_dispatch(object, [object])
    def quadratic_form_diag(self, a, b=None):
        raise RuntimeError('This matrix is singular.')

    def inv_prod(self, a):
        raise RuntimeError('This matrix is singular.')


class Diagonal(Dense, Referentiable):
    """Diagonal symmetric positive-definite matrix.

    Args:
        diag (vector): Diagonal of the matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag):
        Dense.__init__(self, None)
        self._diag = diag

    @property
    def dtype(self):
        return B.dtype(self._diag)

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
        return B.diag(self.diag ** .5)

    def cholesky_mul(self, a):
        return a * self.diag[:, None] ** .5

    def logdet(self):
        return B.sum(B.log(self.diag))

    @_dispatch(object)
    def mah_dist2(self, diff, sum=True):
        iL_diff = diff / self.diag[:, None] ** .5
        return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)

    @_dispatch(Self)
    def ratio(self, denom):
        return B.sum(self.diag / denom.diag)

    @_dispatch(object, object)
    def quadratic_form(self, a, b):
        isqrt_diag = self.diag[:, None] ** .5
        return B.dot(a / isqrt_diag, b / isqrt_diag, tr_a=True)

    @_dispatch(object)
    def quadratic_form(self, a):
        iL_a = a / self.diag[:, None] ** .5
        return B.dot(iL_a, iL_a, tr_a=True)

    @_dispatch(object, object)
    def quadratic_form_diag(self, a, b):
        return B.sum(a * b / self.diag[:, None], axis=0)

    @_dispatch(object)
    def quadratic_form_diag(self, a):
        return B.sum(a ** 2 / self.diag[:, None], axis=0)

    def inv_prod(self, a):
        return a / self.diag[:, None]

    def root(self):
        return self.cholesky()


class UniformDiagonal(Diagonal, Referentiable):
    """Uniformly diagonal symmetric positive-definite matrix.

    Args:
        diag_scal (scalar): Scale of the diagonal of the matrix.
        n (int): Size of the the diagonal.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag_scale, n):
        diag = diag_scale * B.ones([n], dtype=B.dtype(diag_scale))
        Diagonal.__init__(self, diag)


@_dispatch(SPD)
def spd(a):
    """Type a matrix as SPD.

    Args:
        a (tensor): Matrix to type.

    Returns:
        :class:`.spd.SPD`: Matrix as type SPD.
    """
    return a


@_dispatch(B.Numeric)
def spd(a):
    return Dense(a)


# In LAB, redirect addition addition and multiplication for SPDs.

B.add.extend(SPD, object)(add)
B.add.extend(object, SPD)(add)
B.add.extend(SPD, SPD)(add)

B.multiply.extend(SPD, object)(mul)
B.multiply.extend(object, SPD)(mul)
B.multiply.extend(SPD, SPD)(mul)


# Simplify addiction and multiplication between SPDs.

@dispatch_field(SPD, SPD, precedence=1)
def mul(a, b): return Dense(a.mat * b.mat)


@dispatch_field(Diagonal, Diagonal, precedence=1)
def mul(a, b): return Diagonal(a.diag * b.diag)


@dispatch_field(SPD, SPD, precedence=1)
def add(a, b): return Dense(a.mat + b.mat)


@dispatch_field(Diagonal, Diagonal, precedence=1)
def add(a, b): return Diagonal(a.diag + b.diag)


# Simplify addiction and multiplication between SPDs and other objects.

@dispatch_field(SPD, object)
def mul(a, b): return Dense(a.mat * b)


@dispatch_field(object, SPD)
def mul(a, b): return Dense(b.mat * a)


@dispatch_field(Diagonal, object)
def mul(a, b): return Diagonal(a.diag * b)


@dispatch_field(object, Diagonal)
def mul(a, b): return Diagonal(b.diag * a)


@dispatch_field(SPD, object)
def add(a, b): return Dense(a.mat + b)


@dispatch_field(object, SPD)
def add(a, b): return Dense(b.mat + a)
