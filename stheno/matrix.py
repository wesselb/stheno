# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import logging
from itertools import product
from numbers import Number

from lab import B
from plum import Referentiable, Self, Dispatcher

from .field import Element, OneElement, ZeroElement, mul, add, get_field, \
    ScaledElement

__all__ = ['matrix', 'Dense', 'LowRank', 'Diagonal', 'UniformlyDiagonal', 'One',
           'Zero', 'dense', 'Woodbury']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


class Dense(Element, Referentiable):
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

    def __neg__(self):
        return mul(B.cast(-1, dtype=B.dtype(self)), self)

    @property
    def T(self):
        return B.transpose(self)


# Register field.
@get_field.extend(Dense)
def get_field(a): return Dense


class Diagonal(Dense, Referentiable):
    """Diagonal symmetric positive-definite matrix.

    Args:
        diag (vector): Diagonal of the matrix.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag):
        Dense.__init__(self, None)
        self.diag = diag

    def cholesky(self):
        return Diagonal(self.diag ** .5)

    def logdet(self):
        return B.sum(B.log(self.diag))

    def root(self):
        return B.cholesky(self)


class UniformlyDiagonal(Diagonal, Referentiable):
    """Uniformly diagonal symmetric positive-definite matrix.

    Args:
        diag_scal (scalar): Scale of the diagonal of the matrix.
        n (int): Size of the the diagonal.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag_scale, n):
        diag = diag_scale * B.ones([n], dtype=B.dtype(diag_scale))
        Diagonal.__init__(self, diag)


class LowRank(Dense, Referentiable):
    """Low-rank symmetric positive-definite matrix.

    The low-rank matrix is constructed via `left diag(scales) transpose(right)`.

    Args:
        left (tensor): Left part of the matrix.
        right (tensor, optional): Right part of the matrix. Defaults to `left`.
        scales (tensor, optional): Scaling of the outer products. Defaults to
            ones.
        psd (bool, optional): Indicator for positive definiteness. Defaults to
            `False`.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, left, right=None, scales=None, psd=False):
        Dense.__init__(self, None)
        self._left = left
        self._right = left if right is None else right
        self._scales = scales
        self.psd = psd

    @property
    def scales(self):
        if self._scales is None:
            self._scales = B.ones([B.shape_int(self.left)[1]],
                                  dtype=B.dtype(self.left))
        return self._scales

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    def logdet(self):
        raise RuntimeError('Matrix is singular.')

    def split(self):
        zero = B.cast(0, dtype=B.dtype(self))
        pos_scales = B.maximum(self.scales, zero)
        neg_scales = -B.minimum(self.scales, zero)
        return LowRank(left=self.left, right=self.right,
                       scales=pos_scales, psd=True), \
               LowRank(left=self.left, right=self.right,
                       scales=neg_scales, psd=True)

    def symmetrise(self):
        if self.left is self.right:
            return self
        inner = B.concat([self.left + self.right,
                          self.left,
                          self.right], axis=1)
        scales = B.concat([.5 * self.scales,
                           -.5 * self.scales,
                           -.5 * self.scales],
                          axis=0)
        return LowRank(left=inner, right=inner, scales=scales)


class Constant(LowRank, Referentiable):
    """Constant symmetric positive-definite matrix.

    Args:
        constant (tensor): Constant of the matrix.
        num_rows (int): Number of rows.
        num_cols (int, optional): Number of columns. Defaults to the number of
            rows.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, object, [object])
    def __init__(self, constant, num_rows, num_cols=None):
        LowRank.__init__(self, None, psd=True)
        self.constant = constant
        self.num_rows = num_rows
        self.num_cols = num_cols

    @_dispatch(object, {Dense, B.Numeric})
    def __init__(self, constant, reference):
        shape = B.shape(reference)
        Constant.__init__(self, constant, shape[0], shape[1])

    @property
    def scales(self):
        if self._scales is None:
            self._scales = B.expand_dims(self.constant, axis=0)
        return self._scales

    @property
    def left(self):
        if self._left is None:
            self._left = B.ones([self.num_rows, 1],
                                dtype=B.dtype(self.constant))
        return self._left

    @property
    def right(self):
        if self.num_cols is None:
            return self.left
        else:
            if self._right is None:
                self._right = B.ones([self.num_cols, 1],
                                     dtype=B.dtype(self.constant))
            return self._right


class One(Constant, OneElement, Referentiable):
    """Dense matrix full of ones.

    Args:
        dtype (dtype): Data type.
        num_rows (int): Number of rows.
        num_cols (int, optional): Number of columns. Defaults to the number of
            rows.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.DType, object, [object])
    def __init__(self, dtype, num_rows, num_cols=None):
        Constant.__init__(self, B.cast(1, dtype=dtype), num_rows, num_cols)

    @_dispatch({Dense, B.Numeric})
    def __init__(self, reference):
        shape = B.shape(reference)
        One.__init__(self, B.dtype(reference), shape[0], shape[1])


class Zero(Constant, ZeroElement, Referentiable):
    """Dense matrix full of zeros.

    Args:
        dtype (dtype): Data type.
        num_rows (int): Number of rows.
        num_cols (int, optional): Number of columns. Defaults to the number of
            rows.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.DType, object, [object])
    def __init__(self, dtype, num_rows, num_cols=None):
        Constant.__init__(self, B.cast(0, dtype=dtype), num_rows, num_cols)

    @_dispatch({Dense, B.Numeric})
    def __init__(self, reference):
        shape = B.shape(reference)
        Zero.__init__(self, B.dtype(reference), shape[0], shape[1])


class Woodbury(Dense, Referentiable):
    """Sum of a low-rank and diagonal symmetric positive-definite matrix.

    Args:
        lr (:class:`.matrix.LowRank`): Low-rank part.
        diag (:class:`.matrix.Diagonal`): Diagonal part.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(LowRank, Diagonal)
    def __init__(self, lr, diag):
        Dense.__init__(self, None)
        self.lr_part = lr
        self.diag_part = diag

        # Store stuff related to the (outer) Schur complement.
        self._schur_inner = None
        self._schur = None
        self._schur_left = None
        self._schur_right = None

    def schur_complement(self):
        if self._schur is None:
            if self.lr_part.psd:
                self._schur_inner = self.diag_part
                lr = self.lr_part
            else:
                # Low-rank part not guaranteed to be PD. Split it up.
                lr_pos, lr_neg = self.lr_part.symmetrise().split()
                self._schur_inner = self.diag_part + lr_pos
                lr = lr_neg

            # Take the scales together with the left and right part.
            sqrt_scales = lr.scales[None, :] ** .5
            self._schur_left = lr.right * sqrt_scales
            self._schur_right = lr.left * sqrt_scales

            # Compute Schur complement.
            prod = B.qf(self._schur_inner, self._schur_left, self._schur_right)

            if self.lr_part.psd:
                self._schur = matrix(B.add(B.eye_from(prod), prod))
            else:
                # Low-rank part was split up. Instead subtract.
                self._schur = matrix(B.subtract(B.eye_from(prod), prod))

        return self._schur_inner, \
               self._schur, \
               self._schur_left, \
               self._schur_right

    def logdet(self):
        inner, schur, left, right = self.schur_complement()
        return B.logdet(schur) + B.logdet(inner)


# Conveniently make identity matrices.

@B.eye_from.extend({Dense, B.Numeric})
def eye_from(a): return Diagonal(B.ones([B.shape(a)[0]], dtype=B.dtype(a)))


# Conversion between `Matrix`s and dense matrices:

@_dispatch(Dense)
def matrix(a):
    """Matrix as `Matrix`.

    Args:
        a (tensor): Matrix to type.

    Returns:
        :class:`.matrix.Dense`: Matrix as `Matrix`.
    """
    return a


@_dispatch(B.Numeric)
def matrix(a): return Dense(a)


@_dispatch(Dense)
def dense(a):
    """`Matrix` as matrix.

    Args:
        a (:class:`.matrix.Dense`): `Matrix` to unwrap.

    Returns:
        tensor: `Matrix` as matrix.
    """
    return a.mat


@_dispatch(Diagonal)
def dense(a): return B.diag(a.diag)


@_dispatch(LowRank)
def dense(a): return B.matmul(a.left * a.scales[None, :], a.right, tr_b=True)


@_dispatch(Constant)
def dense(a): return a.constant * B.ones(B.shape(a), dtype=B.dtype(a))


@_dispatch(Woodbury)
def dense(a): return dense(a.lr_part) + dense(a.diag_part)


@_dispatch(B.Numeric)
def dense(a): return a


# Get data type of `Matrix`s.

@B.dtype.extend(Dense)
def dtype(a): return B.dtype(dense(a))


@B.dtype.extend(Diagonal)
def dtype(a): return B.dtype(a.diag)


@B.dtype.extend(LowRank)
def dtype(a): return B.dtype(a.left)


@B.dtype.extend(Constant)
def dtype(a): return B.dtype(a.constant)


@B.dtype.extend(Woodbury)
def dtype(a): return B.dtype(a.lr_part)


# Get diagonals of `Matrix`s.

@B.diag.extend(Dense)
def diag(a): return B.diag(dense(a))


@B.diag.extend(Diagonal)
def diag(a): return a.diag


@B.diag.extend(LowRank)
def diag(a):
    # TODO: Correctly handle shape here.
    return B.sum(a.left * a.right * a.scales[None, :], 1)


@B.diag.extend(Constant)
def diag(a):
    # TODO: Correctly handle shape here.
    return a.constant * B.ones([a.num_rows], dtype=B.dtype(a))


@B.diag.extend(Woodbury)
def diag(a): return B.diag(a.lr_part) + B.diag(a.diag_part)


# Cholesky decompose `Matrix`s.

@B.cholesky.extend(Dense)
def cholesky(a): return a.cholesky()


# Some efficient matrix multiplications:

@B.matmul.extend(Diagonal, {B.Numeric, Dense})
def matmul(a, b, tr_a=False, tr_b=False):
    b = B.transpose(b) if tr_b else b
    return B.diag(a)[:, None] * dense(b)


@B.matmul.extend({B.Numeric, Dense}, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    return dense(a) * B.diag(b)[None, :]


@B.matmul.extend(Diagonal, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    return Diagonal(B.diag(a) * B.diag(b))


@B.matmul.extend(Diagonal, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    b = B.transpose(b) if tr_b else b
    return LowRank(left=B.diag(a)[:, None] * b.left,
                   right=b.right,
                   scales=b.scales)


@B.matmul.extend(LowRank, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    return LowRank(left=a.left,
                   right=a.right * B.diag(b)[:, None],
                   scales=a.scales)


@B.matmul.extend(LowRank, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    inner = B.matmul(a.right, b.left, tr_a=True)
    U, S, V = B.svd(inner)
    left = B.matmul(a.left * a.scales[None, :], U)
    right = B.matmul(b.right * b.scales[None, :], V)
    return LowRank(left=left, right=right, scales=S)


@B.matmul.extend({B.Numeric, Dense}, Woodbury, precedence=1)
def matmul(a, b, tr_a=False, tr_b=False):
    return B.add(B.matmul(a, b.lr_part, tr_a=tr_a, tr_b=tr_b),
                 B.matmul(a, b.diag_part, tr_a=tr_a, tr_b=tr_b))


@B.matmul.extend(Woodbury, {B.Numeric, Dense}, precedence=1)
def matmul(a, b, tr_a=False, tr_b=False):
    return B.add(B.matmul(a.lr_part, b, tr_a=tr_a, tr_b=tr_b),
                 B.matmul(a.diag_part, b, tr_a=tr_a, tr_b=tr_b))


@B.matmul.extend(Woodbury, Woodbury, precedence=1)
def matmul(a, b, tr_a=False, tr_b=False):
    return add(add(B.matmul(a.lr_part, b.lr_part, tr_a=tr_a, tr_b=tr_b),
                   B.matmul(a.lr_part, b.diag_part, tr_a=tr_a, tr_b=tr_b)),
               add(B.matmul(a.diag_part, b.lr_part, tr_a=tr_a, tr_b=tr_b),
                   B.matmul(a.diag_part, b.diag_part, tr_a=tr_a, tr_b=tr_b)))


@B.matmul.extend(LowRank, B.Numeric)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    return B.matmul(a.left, B.matmul(a.scales[None, :] * a.right, b, tr_a=True))


@B.matmul.extend(B.Numeric, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    return B.matmul(B.matmul(a, b.scales[None, :] * b.left), b.right, tr_b=True)


@B.trimatmul.extend(object, object, object)
def trimatmul(a, b, c, tr_a=False, tr_b=False, tr_c=False):
    return B.matmul(B.matmul(a, b, tr_a=tr_a, tr_b=tr_b), c, tr_b=tr_c)


@B.trimatmul.extend(Woodbury, object, Woodbury)
def trimatmul(a, b, c, tr_a=False, tr_b=False, tr_c=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    c = B.transpose(c) if tr_c else c
    return B.trimatmul(a.lr_part, b, c.lr_part) + \
           B.trimatmul(a.diag_part, b, c.lr_part) + \
           B.trimatmul(a.lr_part, b, c.diag_part) + \
           B.trimatmul(a.diag_part, b, c.diag_part)


@B.trimatmul.extend(LowRank, LowRank, LowRank)
def trimatmul(a, b, c, tr_a=False, tr_b=False, tr_c=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    c = B.transpose(c) if tr_c else c
    left_block = B.matmul(a.right, b.left, tr_a=True)
    right_block = B.matmul(c.left, b.right, tr_a=True)
    left = B.matmul(a.left * a.scales[None, :], left_block)
    right = B.matmul(c.right * c.scales[None, :], right_block)
    return LowRank(left=left, right=right, scales=b.scales)


# Some efficient inverses:

@B.inverse.extend(Diagonal)
def inverse(a): return Diagonal(1 / B.diag(a))


# Compute products with inverses of `Matrix`s.

@B.inv_prod.extend(object, object)
def inv_prod(a, b):
    """Compute `inv(a) b`.

    Args:
        a (tensor): Matrix to invert.
        b (tensor): Matrix to multiply with.

    Returns:
        tensor: Product.
    """
    return B.qf(a, B.eye_from(a), b)


@B.inv_prod.extend(Diagonal, object)
def inv_prod(a, b):
    return b / a.diag[:, None]


@B.inv_prod.extend(LowRank, object)
def inv_prod(a, b):
    raise RuntimeError('Matrix is singular.')


# Compute the log-determinant of `Matrix`s.


@B.logdet.extend(B.Numeric)
def logdet(a): return B.logdet(matrix(a))


@B.logdet.extend(Dense)
def logdet(a): return a.logdet()


# Compute roots of `Matrix`s.

@B.root.extend(Dense)
def root(a): return a.root()


# Compute quadratic forms and diagonals thereof.

@B.qf.extend(object, object)
def qf(a, b):
    """Compute the quadratic form `transpose(b) inv(a) c`.

    Args:
        a (tensor): Covariance matrix.
        b (tensor): `b`.
        c (tensor, optional): `c`. Defaults to `b`.

    Returns:
        :class:`.matrix.Dense`: Quadratic form.
    """
    prod = B.trisolve(B.cholesky(a), b)
    return matrix(B.matmul(prod, prod, tr_a=True))


@B.qf.extend(object, object, object)
def qf(a, b, c):
    if b is c:
        return B.qf(a, b)
    left = B.trisolve(B.cholesky(a), b)
    right = B.trisolve(B.cholesky(a), c)
    return B.matmul(left, right, tr_a=True)


@B.qf.extend(Diagonal, object)
def qf(a, b):
    return B.qf(a, b, b)


@B.qf.extend(Diagonal, object, object)
def qf(a, b, c):
    return B.trimatmul(b, B.inverse(a), c, tr_a=True)


@B.qf.extend_multi((LowRank, object), (LowRank, object, object))
def qf(a, b, c=None):
    raise RuntimeError('Matrix is singular.')


@B.qf.extend(Woodbury, object)
def qf(a, b):
    return B.qf(a, b, b)


@B.qf.extend(Woodbury, object, object)
def qf(a, b, c):
    schur_inner, schur, schur_left, schur_right = a.schur_complement()
    chol = B.cholesky(schur)

    # Compute the component of the low-rank part.
    left = B.transpose(B.trisolve(chol, B.qf(schur_inner, schur_left, b)))
    right = B.transpose(B.trisolve(chol, B.qf(schur_inner, schur_right, c)))

    # Complete the Woodbury matrix identity.
    return matrix(B.qf(schur_inner, b, c)) - LowRank(left, right)


@B.qf_diag.extend(object, object)
def qf_diag(a, b):
    """Compute the diagonal of `transpose(b) inv(a) c`.

    Args:
        a (:class:`.matrix.Dense`): Covariance matrix.
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


@B.qf_diag.extend(Woodbury, object)
def qf_diag(a, b):
    # TODO: Optimise this!
    return B.qf_diag(a, b, b)


@B.qf_diag.extend(Woodbury, object, object)
def qf_diag(a, b, c):
    # TODO: Optimise this!
    return B.diag(B.qf(a, b, c))


@B.qf_diag.extend_multi((LowRank, object), (LowRank, object, object))
def qf_diag(a, b, c=None):
    raise RuntimeError('Matrix is singular.')


# Compute ratios between `Matrix`s.

@B.ratio.extend(object, object)
def ratio(a, b):
    """Compute the ratio between two positive-definite matrices.

    Args:
        a (tensor): Numerator.
        b (tensor): Denominator.

    Returns:
        tensor: Ratio.
    """
    return B.sum(B.qf_diag(b, B.cholesky(a)))


@B.ratio.extend(LowRank, Dense)
def ratio(a, b):
    # TODO: Check that `b` is indeed PSD.
    return B.sum(B.qf_diag(b, a.left * a.scales[None, :] ** .5))


@B.ratio.extend(Diagonal, Diagonal)
def ratio(a, b):
    return B.sum(a.diag / b.diag)


# Transpose `Matrix`s.

@B.transpose.extend(Dense)
def transpose(a): return matrix(B.transpose(dense(a)))


@B.transpose.extend(Diagonal)
def transpose(a): return a


@B.transpose.extend(LowRank)
def transpose(a): return LowRank(left=a.right, right=a.left, scales=a.scales)


@B.transpose.extend(Woodbury)
def transpose(a):
    return Woodbury(B.transpose(a.lr_part), B.transpose(a.diag_part))


# Extend LAB to the field of `Matrix`s.

B.add.extend(Dense, object)(add)
B.add.extend(object, Dense)(add)
B.add.extend(Dense, Dense)(add)

B.multiply.extend(Dense, object)(mul)
B.multiply.extend(object, Dense)(mul)
B.multiply.extend(Dense, Dense)(mul)


@B.subtract.extend(object, object)
def subtract(a, b): return B.add(a, -b)


@B.divide.extend(object, object)
def divide(a, b): return B.multiply(a, 1 / b)


# Get shapes of `Matrix`s.

@B.shape.extend(Dense)
def shape(a): return B.shape(dense(a))


@B.shape.extend(Diagonal)
def shape(a): return B.shape(a.diag)[0], B.shape(a.diag)[0]


@B.shape.extend(LowRank)
def shape(a): return B.shape(a.left)[0], B.shape(a.right)[0]


@B.shape.extend(Constant)
def shape(a):
    cols = a.num_rows if a.num_cols is None else a.num_cols
    return a.num_rows, cols


@B.shape.extend(Woodbury)
def shape(a): return B.shape(a.lr_part)


# Setup promotion and conversion of `Matrix`s as a fallback mechanism.

B.add_promotion_rule(B.Numeric, Dense, B.Numeric)
B.convert.extend(Dense, B.Numeric)(lambda x, _: dense(x))


# Simplify addiction and multiplication between `Matrix`s.

@mul.extend(Dense, Dense)
def mul(a, b): return Dense(dense(a) * dense(b))


@mul.extend(Dense, Diagonal)
def mul(a, b): return Diagonal(B.diag(a) * b.diag)


@mul.extend(Diagonal, Dense)
def mul(a, b): return Diagonal(a.diag * B.diag(b))


@mul.extend(Diagonal, Diagonal)
def mul(a, b): return Diagonal(a.diag * b.diag)


@mul.extend(LowRank, LowRank)
def mul(a, b):
    def cartesian_mul(xs, ys):
        return [x * y for x, y in product(xs, ys)]

    left = B.stack(cartesian_mul(B.unstack(a.left, axis=1),
                                 B.unstack(b.left, axis=1)), axis=1)
    right = B.stack(cartesian_mul(B.unstack(a.right, axis=1),
                                  B.unstack(b.right, axis=1)), axis=1)
    scales = B.stack(cartesian_mul(B.unstack(a.scales, axis=0),
                                   B.unstack(b.scales, axis=0)), axis=0)
    return LowRank(left=left, right=right, scales=scales)


@add.extend(Dense, Dense)
def add(a, b): return Dense(dense(a) + dense(b))


@add.extend(Diagonal, Diagonal)
def add(a, b): return Diagonal(a.diag + b.diag)


@add.extend(LowRank, LowRank)
def add(a, b): return LowRank(left=B.concat([a.left, b.left], axis=1),
                              right=B.concat([a.right, b.right], axis=1),
                              scales=B.concat([a.scales, b.scales], axis=0))


# Simplify addiction and multiplication between `Matrix`s and other objects. We
# immediately resolve scaled elements.

@mul.extend(object, Dense)
def mul(a, b): return mul(b, a)


@mul.extend(Dense, object)
def mul(a, b): return mul(a, mul(One(a), b))


@add.extend(object, Dense)
def add(a, b): return add(b, a)


@add.extend(Dense, object)
def add(a, b): return add(a, mul(One(a), b))


@mul.extend(object, One)
def mul(a, b): return mul(b, a)


@mul.extend(One, object)
def mul(a, b):
    if B.rank(b) == 0:
        if isinstance(b, Number) and b == 0:
            return Zero(a)
        elif isinstance(b, Number) and b == 1:
            return a
        else:
            return Constant(b, a)
    else:
        return matrix(dense(a) * b)


# Take over construction of ones: construction requires arguments.

@add.extend(object, Zero)
def add(a, b): return add(b, a)


@add.extend(Zero, object)
def add(a, b): return mul(b, One(a))


@add.extend(Zero, Zero)
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


# Expand out multiplication with Woodbury matrices. Higher precedence is used to
# have this happen immediately.

@mul.extend(Dense, Woodbury, precedence=1)
def mul(a, b): return add(mul(a, b.lr_part), mul(a, b.diag_part))


@mul.extend(Woodbury, Dense, precedence=1)
def mul(a, b): return add(mul(a.lr_part, b), mul(a.diag_part, b))


@mul.extend(Woodbury, Woodbury, precedence=1)
def mul(a, b): return add(add(mul(a.lr_part, b.lr_part),
                              mul(a.lr_part, b.diag_part)),
                          add(mul(a.diag_part, b.lr_part),
                              mul(a.diag_part, b.diag_part)))
