# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import logging

from lab import B
from plum import Referentiable, Self, Dispatcher

from .field import Element, OneElement, ZeroElement, mul, add, get_field

__all__ = ['spd', 'SPD', 'LowRank', 'Diagonal', 'UniformDiagonal', 'OneSPD',
           'ZeroSPD', 'dense', 'Woodbury']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


class SPD(Element, Referentiable):
    """Symmetric positive-definite matrix.

    Args:
        mat (tensor): Symmetric positive-definite matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, mat):
        self.mat = mat
        self._cholesky = None
        self._logdet = None
        self._root = None

    def cholesky(self):
        """Compute the Cholesky decomposition.

        Returns:
            tensor: Cholesky decomposition.
        """
        if self._cholesky is None:
            self._cholesky = B.cholesky(B.reg(dense(self)))
        return self._cholesky

    def logdet(self):
        """Compute the log-determinant.

        Returns:
            scalar: Log-determinant.
        """
        if self._logdet is None:
            self._logdet = 2 * B.sum(B.log(B.diag(B.cholesky(self))))
        return self._logdet

    def root(self):
        """Compute a square root."""
        if self._root is None:
            vals, vecs = B.eig(B.reg(dense(self)))
            self._root = B.matmul(vecs * vals[None, :] ** .5, vecs, tr_b=True)
        return self._root

    def inv_prod(self, a):
        """Compute the product `inv(self.mat) a`.

        Args:
            a (tensor): `a`

        Returns:
            tensor: Product.
        """
        return B.cholesky_solve(B.cholesky(self), a)


# Register field.
@get_field.extend(SPD)
def get_field(a): return SPD


class OneSPD(SPD, OneElement):
    """Dense matrix full of ones."""


class ZeroSPD(SPD, ZeroElement):
    """Dense matrix full of ones."""


class LowRank(SPD, Referentiable):
    """Low-rank symmetric positive-definite matrix.

    Args:
        inner (tensor): The matrix such that `inner * inner'` equals this
        matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, inner):
        SPD.__init__(self, None)
        self.inner = inner

    def logdet(self):
        raise RuntimeError('This matrix is singular.')

    def inv_prod(self, a):
        raise RuntimeError('This matrix is singular.')


class Diagonal(SPD, Referentiable):
    """Diagonal symmetric positive-definite matrix.

    Args:
        diag (vector): Diagonal of the matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag):
        SPD.__init__(self, None)
        self.diag = diag

    def cholesky(self):
        return B.diag(self.diag ** .5)

    def logdet(self):
        return B.sum(B.log(self.diag))

    def root(self):
        return B.cholesky(self)

    def inv_prod(self, a):
        return a / self.diag[:, None]


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


class Woodbury(SPD, Referentiable):
    """Sum of a low-rank and diagonal symmetric positive-definite matrix.

    Args:
        lr (:class:`.spd.LowRank`): Low-rank part.
        diag (:class:`.spd.Diagonal`): Diagonal part.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(LowRank, Diagonal)
    def __init__(self, lr, diag):
        SPD.__init__(self, None)
        self.lr_part = lr
        self.diag_part = diag


# Conversion between SPDs and dense matrices:

@_dispatch(SPD)
def spd(a):
    """Matrix as SPD.

    Args:
        a (tensor): Matrix to type.

    Returns:
        :class:`.spd.SPD`: Matrix as SPD.
    """
    return a


@_dispatch(B.Numeric)
def spd(a): return SPD(a)


@_dispatch(SPD)
def dense(a):
    """SPD as matrix.

    Args:
        a (:class:`.spd.SPD`): SPD to unwrap.

    Returns:
        tensor: SPD as matrix.
    """
    return a.mat


@_dispatch(Diagonal)
def dense(a): return B.diag(a.diag)


@_dispatch(LowRank)
def dense(a): return B.matmul(a.inner, a.inner, tr_b=True)


@_dispatch(Woodbury)
def dense(a): return dense(a.lr_part) + dense(a.diag_part)


@_dispatch(B.Numeric)
def dense(a): return a


# Get data type of SPDs.

@B.dtype.extend(SPD)
def dtype(a): return B.dtype(dense(a))


@B.dtype.extend(Diagonal)
def dtype(a): return B.dtype(a.diag)


@B.dtype.extend(LowRank)
def dtype(a): return B.dtype(a.inner)


@B.dtype.extend(Woodbury)
def dtype(a):
    if B.dtype(a.lr_part) != B.dtype(a.diag_part):
        raise RuntimeError('Data types of low-rank part and diagonal part are '
                           'different.')
    return B.dtype(a.lr_part)


# Get diagonals of SPDs.

@B.diag.extend(SPD)
def diag(a): return B.diag(dense(a))


@B.diag.extend(Diagonal)
def diag(a): return a.diag


@B.diag.extend(LowRank)
def diag(a): return B.sum(a.inner ** 2, 1)


@B.diag.extend(Woodbury)
def diag(a): return B.diag(a.lr_part) + B.diag(a.diag_part)


# Choleksy decompose SPDs.

@B.cholesky.extend(SPD)
def cholesky(a): return a.cholesky()


# Multiply Cholesky decompositions of SPDs.

@B.cholesky_mul.extend(SPD, object)
def cholesky_mul(a, b):
    """Multiply the Cholesky decomposition of `a` with `b`.

    Args:
        a (:class:`.spd.SPD`): Matrix to compute Cholesky decomposition of.
        b (tensor): Matrix to multiply with.

    Returns:
        tensor: Multiplication of the Cholesky of `a` with `b`.
    """
    return B.matmul(B.cholesky(a), b)


@B.cholesky_mul.extend(Diagonal, object)
def cholesky_mul(a, b): return b * a.diag[:, None] ** .5


# Compute the log-determinant of SPDs.

@B.logdet.extend(SPD)
def logdet(a): return a.logdet()


# Compute roots of SPDs.

@B.root.extend(SPD)
def root(a): return a.root()


# Compute Mahalanobis distances.

@B.mah_dist2.extend(SPD, object, object)
def mah_dist2(a, b, c, sum=True):
    """Compute the square of the Mahalanobis distance.

    Args:
        a (:class:`.spd.SPD`): Covariance matrix.
        b (tensor): First matrix in distance.
        c (tensor, optional): Second matrix in distance. If omitted, `b` is
            assumed to be the differences.
        sum (bool, optional): Compute the sum of all distances instead
            of returning all distances. Defaults to `True`.

    Returns:
        tensor: Distance or distances.
    """
    return B.mah_dist2(a, b - c, sum=sum)


@B.mah_dist2.extend(SPD, object)
def mah_dist2(a, diff, sum=True):
    iL_diff = B.trisolve(B.cholesky(a), diff)
    return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)


@B.mah_dist2.extend(Diagonal, object)
def mah_dist2(a, diff, sum=True):
    iL_diff = diff / a.diag[:, None] ** .5
    return B.sum(iL_diff ** 2) if sum else B.sum(iL_diff ** 2, axis=0)


@B.mah_dist2.extend(LowRank, object)
def mah_dist2(a, diff, sum=True):
    raise RuntimeError('Matrix is singular.')


# Compute quadratic forms and diagonals thereof.

@B.qf.extend(object, object)
def qf(a, b):
    """Compute the quadratic form `transpose(b) inv(a) c`.

    Args:
        a (tensor): Covariance matrix.
        b (tensor): `b`.
        c (tensor, optional): `c`. Defaults to `b`.

    Returns:
        :class:`.spd.SPD`: Quadratic form.
    """
    prod = B.trisolve(B.cholesky(a), b)
    return spd(B.matmul(prod, prod, tr_a=True))


@B.qf.extend(object, object, object)
def qf(a, b, c):
    if b is c:
        return B.qf(a, b)
    left = B.trisolve(B.cholesky(a), b)
    right = B.trisolve(B.cholesky(a), c)
    return B.matmul(left, right, tr_a=True)


@B.qf.extend(Diagonal, object)
def qf(a, b):
    iL_b = b / a.diag[:, None] ** .5
    return B.matmul(iL_b, iL_b, tr_a=True)


@B.qf.extend(Diagonal, object, object)
def qf(a, b, c):
    if b is c:
        return B.qf(a, b)
    isqrt_diag = a.diag[:, None] ** .5
    return B.matmul(b / isqrt_diag, c / isqrt_diag, tr_a=True)


@B.qf.extend_multi((LowRank, object), (LowRank, object, object))
def qf(a, b, c=None):
    raise RuntimeError('Matrix is singular.')


@B.qf_diag.extend(object, object)
def qf_diag(a, b):
    """Compute the diagonal of `transpose(b) inv(a) c`.

    Args:
        a (:class:`.spd.SPD`): Covariance matrix.
        b (tensor): `b`.
        c (tensor, optional): `c`. Defaults to `b`.

    Returns:
        tensor: Diagonal of the quadratic form.
    """
    prod = B.trisolve(B.cholesky(a), b)
    return B.sum(prod ** 2, axis=0)


@B.qf_diag.extend(object, object, object)
def qf_diag(a, b, c):
    if b is c:
        return B.qf_diag(a, b)
    left = B.trisolve(B.cholesky(a), b)
    right = B.trisolve(B.cholesky(a), c)
    return B.sum(left * right, axis=0)


@B.qf_diag.extend(Diagonal, object)
def qf_diag(a, b):
    return B.sum(b ** 2 / a.diag[:, None], axis=0)


@B.qf_diag.extend(Diagonal, object, object)
def qf_diag(a, b, c):
    if b is c:
        return B.qf(a, b)
    return B.sum(b * c / a.diag[:, None], axis=0)


@B.qf_diag.extend_multi((LowRank, object), (LowRank, object, object))
def qf_diag(a, b, c=None):
    raise RuntimeError('Matrix is singular.')


# Compute ratio's between SPDs.

@B.ratio.extend(object, object)
def ratio(a, b):
    """Compute the ratio between two positive-definite matrices.

    Args:
        a (tensor): Numerator.
        b (tensor): Denominator.

    Returns:
        tensor: Ratio.
    """
    return B.mah_dist2(b, B.cholesky(a), sum=True)


@B.ratio.extend(LowRank, SPD)
def ratio(a, b):
    return B.mah_dist2(b, a.inner, sum=True)


@B.ratio.extend(Diagonal, Diagonal)
def ratio(a, b):
    return B.sum(a.diag / b.diag)


# Extend LAB to work with SPDs.

B.add.extend(SPD, B.Numeric)(add)
B.add.extend(B.Numeric, SPD)(add)
B.add.extend(SPD, SPD)(add)

B.multiply.extend(SPD, B.Numeric)(mul)
B.multiply.extend(B.Numeric, SPD)(mul)
B.multiply.extend(SPD, SPD)(mul)


@B.transpose.extend(SPD)
def transpose(a): return a


@B.subtract.extend({B.Numeric, SPD}, {B.Numeric, SPD})
def subtract(a, b): return B.add(a, -b)


@B.divide.extend({B.Numeric, SPD}, {B.Numeric, SPD})
def divide(a, b): return B.multiply(a, 1 / b)


# Get shapes of SPDs.

@B.shape.extend(SPD)
def shape(a): return B.shape(dense(a))


@B.shape.extend(Diagonal)
def shape(a): return B.shape(a.diag)[0], B.shape(a.diag)[0]


@B.shape.extend(LowRank)
def shape(a): return B.shape(a.inner)[0], B.shape(a.inner)[0]


@B.shape.extend(Woodbury)
def shape(a): return B.shape(a.lr_part)


# Setup promotion and conversion of SPDs as a fallback mechanism.

B.add_promotion_rule(B.Numeric, SPD, B.Numeric)
B.convert.extend(SPD, B.Numeric)(lambda x, _: dense(x))


# Simplify addiction and multiplication between SPDs.

@mul.extend(SPD, SPD)
def mul(a, b): return SPD(dense(a) * dense(b))


@mul.extend(Diagonal, Diagonal)
def mul(a, b): return Diagonal(a.diag * b.diag)


@mul.extend(LowRank, LowRank)
def mul(a, b): return LowRank(a.inner * b.inner)


@add.extend(SPD, SPD)
def add(a, b): return SPD(dense(a) + dense(b))


@add.extend(Diagonal, Diagonal)
def add(a, b): return Diagonal(a.diag + b.diag)


@add.extend(LowRank, LowRank)
def add(a, b): return LowRank(B.concat([a.inner, b.inner], axis=1))


# Simplify addiction and multiplication between SPDs and other objects.

@mul.extend(SPD, B.Numeric)
def mul(a, b): return SPD(dense(a) * b)


@mul.extend(B.Numeric, SPD)
def mul(a, b): return SPD(a * dense(b))


@mul.extend(Diagonal, B.Numeric)
def mul(a, b): return Diagonal(a.diag * b)


@mul.extend(B.Numeric, Diagonal)
def mul(a, b): return Diagonal(b.diag * a)


@add.extend(SPD, B.Numeric)
def add(a, b): return SPD(dense(a) + b)


@add.extend(B.Numeric, SPD)
def add(a, b): return SPD(a + dense(b))


# Prevent creation of scaled elements; this is work for later.

@add.extend(ZeroSPD, B.Numeric, precedence=2)
def add(a, b): return b


@add.extend(B.Numeric, ZeroSPD, precedence=2)
def add(a, b): return a


# Woodbury matrices:

@add.extend(LowRank, Diagonal)
def add(a, b): return Woodbury(a, b)


@add.extend(Diagonal, LowRank)
def add(a, b): return Woodbury(b, a)


@add.extend(LowRank, Woodbury)
def add(a, b): return Woodbury(a + b.lr_part, b.diag_part)


@add.extend(Woodbury, LowRank)
def add(a, b): return Woodbury(a.lr_part + b, a.diag_part)


@add.extend(Diagonal, Woodbury)
def add(a, b): return Woodbury(b.lr_part, a + b.diag_part)


@add.extend(Woodbury, Diagonal)
def add(a, b): return Woodbury(a.lr_part, a.diag_part + b)


@add.extend(Woodbury, Woodbury)
def add(a, b): return Woodbury(a.lr_part + b.lr_part,
                               a.diag_part + b.diag_part)
