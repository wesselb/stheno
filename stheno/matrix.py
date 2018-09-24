# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import logging
from numbers import Number
import numpy as np

from lab import B
from plum import Referentiable, Self, Dispatcher

from .field import Element, OneElement, ZeroElement, mul, add, get_field

__all__ = ['matrix', 'Dense', 'LowRank', 'Diagonal', 'UniformlyDiagonal', 'One',
           'Zero', 'dense', 'Woodbury', 'Constant']

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

        # Caching:
        self.cholesky = None
        self.logdet = None
        self.root = None
        self.inverse = None

    def __neg__(self):
        return mul(B.cast(-1, dtype=B.dtype(self)), self)

    def __div__(self, other):
        return B.divide(self, other)

    def __truediv__(self, other):
        return B.divide(self, other)

    def __getitem__(self, item):
        return dense(self)[item]

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
        rows (scalar, optional): Number of rows. Defaults to the length of the
            diagonal.
        cols (scalar, optional): Number of columns. Defaults to the number of
            rows.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag, rows=None, cols=None):
        Dense.__init__(self, None)
        self.diag = diag
        self.rows = B.shape(diag)[0] if rows is None else rows
        self.cols = self.rows if cols is None else cols


class UniformlyDiagonal(Diagonal, Referentiable):
    """Uniformly diagonal symmetric positive-definite matrix.

    Args:
        diag_scale (scalar): Scale of the diagonal of the matrix.
        n (int): Length of the diagonal.
        rows (scalar, optional): Number of rows. Defaults to the length of the
            diagonal.
        cols (scalar, optional): Number of columns. Defaults to the number of
            rows.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag_scale, n, rows=None, cols=None):
        diag = diag_scale * B.ones([n], dtype=B.dtype(diag_scale))
        Diagonal.__init__(self, diag=diag, rows=rows, cols=cols)


class LowRank(Dense, Referentiable):
    """Low-rank symmetric positive-definite matrix.

    The low-rank matrix is constructed via `left diag(scales) transpose(right)`.

    Args:
        left (tensor): Left part of the matrix.
        right (tensor, optional): Right part of the matrix. Defaults to `left`.
        scales (tensor, optional): Scaling of the outer products. Defaults to
            ones.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, left, right=None, middle=None):
        Dense.__init__(self, None)
        self.left = left
        self.right = left if right is None else right
        if middle is None:
            self.middle = B.eye(B.shape(self.left)[1], dtype=B.dtype(self.left))
        else:
            self.middle = middle

        # Shorthands:
        self.l = self.left
        self.r = self.right
        self.m = self.middle


class Constant(LowRank, Referentiable):
    """Constant symmetric positive-definite matrix.

    Args:
        constant (scalar): Constant of the matrix.
        rows (scalar): Number of rows.
        cols (scalar, optional): Number of columns. Defaults to the number of
            rows.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, constant, rows, cols=None):
        self.constant = constant
        self.rows = rows
        self.cols = rows if cols is None else cols

        # Construct and initialise the low-rank representation.
        left = B.ones([self.rows, 1], dtype=B.dtype(self.constant))
        if self.rows is self.cols:
            right = left
        else:
            right = B.ones([self.cols, 1], dtype=B.dtype(self.constant))
        middle = B.expand_dims(B.expand_dims(self.constant, axis=0), axis=0)
        LowRank.__init__(self, left=left, right=right, middle=middle)

    @classmethod
    def from_(cls, constant, ref):
        return cls(B.cast(constant, dtype=B.dtype(ref)), *B.shape(ref))


class One(Constant, OneElement, Referentiable):
    """Dense matrix full of ones.

    Args:
        dtype (dtype): Data type.
        rows (scalar): Number of rows.
        cols (scalar, optional): Number of columns. Defaults to the number of
            rows.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, dtype, rows, cols=None):
        Constant.__init__(self, B.cast(1, dtype=dtype), rows=rows, cols=cols)

    @classmethod
    def from_(cls, ref):
        return cls(B.dtype(ref), *B.shape(ref))


class Zero(Constant, ZeroElement, Referentiable):
    """Dense matrix full of zeros.

    Args:
        dtype (dtype): Data type.
        rows (scalar): Number of rows.
        cols (scalar, optional): Number of columns. Defaults to the number of
            rows.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.DType, [object])
    def __init__(self, dtype, rows, cols=None):
        Constant.__init__(self, B.cast(0, dtype=dtype), rows=rows, cols=cols)

    @classmethod
    def from_(cls, ref):
        return cls(B.dtype(ref), *B.shape(ref))


class Woodbury(Dense, Referentiable):
    """Sum of a low-rank and diagonal symmetric positive-definite matrix.

    Args:
        lr (:class:`.matrix.LowRank`): Low-rank part.
        diag (:class:`.matrix.Diagonal`): Diagonal part.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, lr, diag):
        Dense.__init__(self, None)
        self.lr = lr
        self.diag = diag

        # Caching:
        self.schur = None


# Conveniently make identity matrices.

@B.eye_from.extend({Dense, B.Numeric})
def eye_from(a): return B.eye(B.shape(a)[0], B.shape(a)[1], dtype=B.dtype(a))


# Conversion between matrices and dense matrices:

@_dispatch(Dense)
def matrix(a):
    """Matrix as `Dense`.

    Args:
        a (tensor): Matrix to type.

    Returns:
        :class:`.matrix.Dense`: Matrix as `Dense`.
    """
    return a


@_dispatch(B.Numeric)
def matrix(a): return Dense(a)


@_dispatch(Dense)
def dense(a):
    """`Dense` as matrix.

    Args:
        a (:class:`.matrix.Dense`): `Dense` to unwrap.

    Returns:
        tensor: `Dense` as matrix.
    """
    return a.mat


@_dispatch(B.Numeric)
def dense(a): return a


@_dispatch(Diagonal)
def dense(a): return B.diag(B.diag(a), *B.shape(a))


@_dispatch(LowRank)
def dense(a): return B.matmul(a.left, a.middle, a.right, tr_c=True)


@_dispatch(Woodbury)
def dense(a): return dense(a.lr) + dense(a.diag)


# Get data type of matrices.

@B.dtype.extend(Dense)
def dtype(a): return B.dtype(dense(a))


@B.dtype.extend(Diagonal)
def dtype(a): return B.dtype(a.diag)


@B.dtype.extend(LowRank)
def dtype(a): return B.dtype(a.left)


@B.dtype.extend(Constant)
def dtype(a): return B.dtype(a.constant)


@B.dtype.extend(Woodbury)
def dtype(a): return B.dtype(a.lr)


# Sum over matrices.

@B.sum.extend(Dense, [object])
def sum(a, axis=None): return B.sum(dense(a), axis=axis)


@B.sum.extend(Diagonal, [object])
def sum(a, axis=None):
    # Efficiently handle a number of common cases.
    if axis is None:
        return B.sum(B.diag(a))
    elif axis is 0:
        return B.concat([B.diag(a), B.zeros([B.shape(a)[1] - B.diag_len(a)],
                                            dtype=B.dtype(a))], axis=0)
    elif axis is 1:
        return B.concat([B.diag(a), B.zeros([B.shape(a)[0] - B.diag_len(a)],
                                            dtype=B.dtype(a))], axis=0)
    else:
        # Fall back to generic implementation.
        return B.sum.invoke(Dense)(a, axis=axis)


@B.sum.extend(LowRank, [object])
def sum(a, axis=None):
    # Efficiently handle a number of common cases.
    if axis is None:
        left = B.expand_dims(B.sum(a.left, axis=0), axis=0)
        right = B.expand_dims(B.sum(a.right, axis=0), axis=1)
        return B.matmul(left, a.middle, right)[0, 0]
    elif axis is 0:
        left = B.expand_dims(B.sum(a.left, axis=0), axis=0)
        return B.matmul(left, a.middle, a.right, tr_c=True)[0, :]
    elif axis is 1:
        right = B.expand_dims(B.sum(a.right, axis=0), axis=1)
        return B.matmul(a.left, a.middle, right)[:, 0]
    else:
        # Fall back to generic implementation.
        return B.sum.invoke(Dense)(a, axis=axis)


@B.sum.extend(Woodbury, [object])
def sum(a, axis=None): return B.sum(a.lr, axis=axis) + B.sum(a.diag, axis=axis)


# Take traces of matrices.

@B.trace.extend(Dense)
def trace(a): return B.trace(dense(a))


# Get the length of the diagonal of a matrix.

@B.diag_len.extend({B.Numeric, Dense})
def diag_len(a):
    """Get the length of the diagonal of a matrix.

    Args:
        a (tensor): Matrix of which to get the length of the diagonal.

    Returns:
        tensor: Length of the diagonal of `a`.
    """
    return B.minimum(*B.shape(a))


# Get diagonals of matrices or create diagonal matrices.

@B.diag.extend(Dense)
def diag(a): return B.diag(dense(a))


@B.diag.extend(Diagonal)
def diag(a):
    # Append zeros or remove elements as necessary.
    diag_len = B.diag_len(a)
    extra_zeros = B.maximum(diag_len - B.shape(a.diag)[0], 0)
    return B.concat([a.diag[:diag_len],
                     B.zeros([extra_zeros], dtype=B.dtype(a))], axis=0)


@B.diag.extend(LowRank)
def diag(a):
    # The matrix might be non-square, so handle that.
    diag_len = B.diag_len(a)
    return B.sum(B.matmul(a.left, a.middle)[:diag_len, :] *
                 a.right[:diag_len, :], 1)


@B.diag.extend(Constant)
def diag(a): return a.constant * B.ones([B.diag_len(a)], dtype=B.dtype(a))


@B.diag.extend(Woodbury)
def diag(a): return B.diag(a.lr) + B.diag(a.diag)


@B.diag.extend(B.Numeric, B.Numeric, [B.Numeric])
def diag(diag, rows, cols=None):
    cols = rows if cols is None else cols

    # Cut the diagonal to accommodate the size.
    diag = diag[:B.minimum(rows, cols)]

    # Pad with extra zeros as specified.
    diag_len, dtype = B.shape(diag)[0], B.dtype(diag)
    extra_rows, extra_cols = rows - diag_len, cols - diag_len
    return B.concat2d([B.diag(diag),
                       B.zeros([diag_len, extra_cols], dtype=dtype)],
                      [B.zeros([extra_rows, diag_len], dtype=dtype),
                       B.zeros([extra_rows, extra_cols], dtype=dtype)])


# Cholesky decompose matrices.

@B.cholesky.extend(Dense)
def cholesky(a):
    if a.cholesky is None:
        a.cholesky = B.cholesky(B.reg(dense(a)))
    return a.cholesky


@B.cholesky.extend(Diagonal)
def cholesky(a):
    # NOTE: Assumes that the matrix is symmetric.
    if a.cholesky is None:
        a.cholesky = Diagonal(a.diag ** .5)
    return a.cholesky


@B.cholesky.extend(LowRank)
def cholesky(a):
    # NOTE: Assumes that the matrix is symmetric.
    if a.cholesky is None:
        m = matrix(a.middle)
        m = 0.5 * (m + m.T)
        a.cholesky = B.matmul(a.left, B.cholesky(m))
    return a.cholesky


# Matrix multiplications:

@B.matmul.extend(Dense, Dense, precedence=-1)
def matmul(a, b, tr_a=False, tr_b=False):
    # This is the fallback implementation. Any other specialised implementation
    # should be preferred over this.
    return B.matmul(dense(a), dense(b), tr_a=tr_a, tr_b=tr_b)


@B.matmul.extend(Diagonal, {B.Numeric, Dense})
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b

    # Get shape of `a`.
    a_rows, a_cols = B.shape(a)

    # If `a` is square, don't do complicated things.
    if a_rows is a_cols:
        return B.diag(a)[:, None] * dense(b)

    # Compute the core part.
    rows = B.minimum(a_rows, B.shape(b)[0])
    core = B.diag(a)[:rows, None] * dense(b)[:rows, :]

    # Compute extra zeros to be appended.
    extra_rows = a_rows - rows
    extra_zeros = B.zeros([extra_rows, B.shape(b)[1]], dtype=B.dtype(b))
    return B.concat([core, extra_zeros], axis=0)


@B.matmul.extend({B.Numeric, Dense}, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b

    # Get shape of `b`.
    b_rows, b_cols = B.shape(b)

    # If `b` is square, don't do complicated things.
    if b_rows is b_cols:
        return dense(a) * B.diag(b)[None, :]

    # Compute the core part.
    cols = B.minimum(B.shape(a)[1], b_cols)
    core = dense(a)[:, :cols] * B.diag(b)[None, :cols]

    # Compute extra zeros to be appended.
    extra_cols = b_cols - cols
    extra_zeros = B.zeros([B.shape(a)[0], extra_cols], dtype=B.dtype(b))
    return B.concat([core, extra_zeros], axis=1)


@B.matmul.extend(Diagonal, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    diag_len = B.minimum(B.diag_len(a), B.diag_len(b))
    return Diagonal(B.diag(a)[:diag_len] * B.diag(b)[:diag_len],
                    rows=B.shape(a)[0],
                    cols=B.shape(b)[1])


@B.matmul.extend(LowRank, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    middle_inner = B.matmul(a.right, b.left, tr_a=True)
    middle = B.matmul(a.middle, middle_inner, b.middle)
    return LowRank(left=a.left, right=b.right, middle=middle)


@B.matmul.extend(LowRank, {B.Numeric, Dense})
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    return LowRank(left=a.left,
                   right=B.matmul(b, a.right, tr_a=True),
                   middle=a.middle)


@B.matmul.extend({B.Numeric, Dense}, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b
    return LowRank(left=B.matmul(a, b.left),
                   right=b.right,
                   middle=b.middle)


@B.matmul.extend(Diagonal, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    # TODO: Refactor this to use ambiguous method resolution.
    return B.matmul.invoke(Dense, LowRank)(a, b, tr_a, tr_b)


@B.matmul.extend(LowRank, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    # TODO: Refactor this to use ambiguous method resolution.
    return B.matmul.invoke(LowRank, Dense)(a, b, tr_a, tr_b)


@B.matmul.extend(object, object, object)
def matmul(a, b, c, tr_a=False, tr_b=False, tr_c=False):
    return B.matmul(B.matmul(a, b, tr_a=tr_a, tr_b=tr_b), c, tr_b=tr_c)


@B.matmul.extend({B.Numeric, Dense}, Woodbury, precedence=2)
def matmul(a, b, tr_a=False, tr_b=False):
    # Prioritise expanding out the Woodbury matrix. Give this one even higher
    # precedence to resolve ambiguity in the case of two Woodbury matrices.
    return B.add(B.matmul(a, b.lr, tr_a=tr_a, tr_b=tr_b),
                 B.matmul(a, b.diag, tr_a=tr_a, tr_b=tr_b))


@B.matmul.extend(Woodbury, {B.Numeric, Dense}, precedence=1)
def matmul(a, b, tr_a=False, tr_b=False):
    # Prioritise expanding out the Woodbury matrix.
    return B.add(B.matmul(a.lr, b, tr_a=tr_a, tr_b=tr_b),
                 B.matmul(a.diag, b, tr_a=tr_a, tr_b=tr_b))


# Matrix inversion:

@B.inverse.extend({B.Numeric, Dense})
def inverse(a):
    # Assume that `a` is PD.
    a = matrix(a)
    if a.inverse is None:
        inv_prod = B.trisolve(B.cholesky(a), B.eye_from(a))
        a.inverse = B.matmul(inv_prod, inv_prod, tr_a=True)
    return a.inverse


@B.inverse.extend(Diagonal)
def inverse(a): return Diagonal(1 / B.diag(a), *reversed(B.shape(a)))


@B.inverse.extend(Woodbury)
def inverse(a):
    # Use the Woodbury matrix identity.
    if a.inverse is None:
        inv_diag = B.inverse(a.diag)
        a.inverse = inv_diag - \
                    LowRank(left=B.matmul(inv_diag, a.lr.left),
                            right=B.matmul(inv_diag, a.lr.right),
                            middle=B.inverse(B.schur(a)))
    return a.inverse


@B.inverse2.extend(Woodbury)
def inverse2(a):
    # Use the Woodbury matrix identity.
    if a.inverse is None:
        inv_diag = B.inverse(a.diag)
        a.inverse = inv_diag + \
                    LowRank(left=B.matmul(inv_diag, a.lr.left),
                            right=B.matmul(inv_diag, a.lr.right),
                            middle=B.inverse(B.schur2(a)))
    return a.inverse


@B.lr_diff.extend(LowRank, LowRank)
def lr_diff(a, b):
    # chol_a = B.cholesky(matrix(a.m))
    # chol_b = B.cholesky(matrix(b.m))
    # a = LowRank(left=B.matmul(a.l, chol_a), right=B.matmul(a.r, chol_a))
    # b = LowRank(left=B.matmul(b.l, chol_b), right=B.matmul(b.r, chol_b))
    print('LRDIFF:')
    print('a', B.shape(a.l), B.shape(a.m), B.shape(a.r))
    print('b', B.shape(b.l), B.shape(b.m), B.shape(b.r))
    diff = a - b
    n = B.shape(a.left)[1]
    print('diff', B.shape(diff.l), B.shape(diff.m), B.shape(diff.r))

    # Left:
    u, s, v = B.svd(diff.left)
    ul, sl, vl = u[:, :n], s[:n], v[:, :n]
    ul, sl, vl = u, s, v

    # Right:
    u, s, v = B.svd(diff.right)
    ur, sr, vr = u[:, :n], s[:n], v[:, :n]
    ur, sr, vr = u, s, v

    m = B.matmul(Diagonal(sl),
                 B.matmul(vl, diff.m, vr, tr_a=True),
                 Diagonal(sr))
    print('result for m', B.shape(m))
    print('DONE')
    return LowRank(left=ul, right=ur, middle=m)


@B.inverse.extend(LowRank)
def inverse(a): raise RuntimeError('Matrix is singular.')


# Compute the log-determinant of matrices.

@B.logdet.extend({B.Numeric, Dense})
def logdet(a):
    a = matrix(a)
    if a.logdet is None:
        a.logdet = 2 * B.sum(B.log(B.diag(B.cholesky(a))))
    return a.logdet


@B.logdet.extend(Diagonal)
def logdet(a): return B.sum(B.log(a.diag))


@B.logdet.extend(Woodbury)
def logdet(a):
    return B.logdet(B.schur(a)) + \
           B.logdet(a.diag) + \
           B.logdet(a.lr.middle)


@B.logdet.extend(LowRank)
def logdet(a): raise RuntimeError('Matrix is singular.')


# Compute roots of matrices.

@B.root.extend({B.Numeric, Dense})
def root(a):
    a = matrix(a)
    if a.root is None:
        vals, vecs = B.eig(B.reg(dense(a)))
        a.root = B.matmul(vecs * vals[None, :] ** .5, vecs, tr_b=True)
    return a.root


@B.root.extend(Diagonal)
def root(a):
    return Diagonal(a.diag ** .5)


# Sample from covariance matrices.

@B.sample.extend({B.Numeric, Dense}, [object])
def sample(a, num=1):
    """Sample from covariance matrices.

    Args:
        a (tensor): Covariance matrix to sample from.
        num (int): Number of samples.

    Returns:
        tensor: Samples as rank 2 column vectors.
    """
    # Convert integer data types to floats.
    dtype = float if B.issubdtype(B.dtype(a), np.integer) else B.dtype(a)

    # Perform sampling operation.
    chol = B.cholesky(a)
    return B.matmul(chol, B.randn([B.shape(chol)[1], num], dtype=dtype))


@B.sample.extend(Woodbury, [object])
def sample(a, num=1): return B.sample(a.diag, num) + B.sample(a.lr, num)


# Compute Schur complements.

@B.schur.extend(object, object, object, object)
def schur(a, b, c, d):
    """Compute the Schur complement `a - transpose(b) inv(c) d`.

    Args:
        a (tensor): `a`.
        b (tensor): `b`.
        c (tensor): `c`.
        d (tensor): `d`.

    Returns:
        :class:`.matrix.Dense`: Schur complement.
    """
    return B.subtract(a, B.qf(c, b, d))


@B.schur.extend(object, object, Woodbury, object)
def schur(a, b, c, d):
    # The subtraction will yield a negative-definite Woodbury matrix. Convert
    # to dense to avoid issues with positive definiteness later on. This is
    # suboptimal, but more complicated abstractions are required to exploit
    # this.
    return dense(B.subtract(a, B.qf(c, b, d)))


@B.schur.extend(Woodbury)
def schur(a):
    if a.schur is None:
        a.schur = B.inverse(a.lr.middle) + \
                  B.matmul(a.lr.right, B.inverse(a.diag), a.lr.left, tr_a=True)
    return a.schur


@B.schur2.extend(Woodbury)
def schur2(a):
    if a.schur is None:
        a.schur = B.inverse(-a.lr.middle) - \
                  B.matmul(a.lr.right, B.inverse(a.diag), a.lr.left, tr_a=True)
    return a.schur


# @B.schur.extend(Woodbury, Woodbury, Woodbury, Woodbury)
# def schur(a, b, c, d):
#     # if B.mean(B.abs(dense(b) - dense(d))) > 1e-6:
#     #     print('There we go cheat... with this:')
#     #     for x in [a, b, c, d]:
#     #         print(B.shape(x.lr.l), B.shape(x.lr.m), B.shape(x.lr.r))
#     #     res = a - B.matmul(b, B.inverse(c), d, tr_a=True)
#     #     print(res, B.shape(res.lr.l), B.shape(res.lr.m), B.shape(res.lr.r))
#     #     return res
#     # Assumptions:
#     #   - b.lr.middle = d.lr.middle
#     #   - b.lr.l = b.lr.r
#     print('There we go... with this:')
#     for x in [a, b, c, d]:
#         print(B.shape(x.lr.l), B.shape(x.lr.m), B.shape(x.lr.r))
#     print('All good? Then go!')
#
#     # Split `c` and `a`.
#     y, m = b.lr.l, b.lr.middle
#     lr_a = LowRank(left=b.lr.r, right=d.lr.r, middle=m)
#     lr_c = LowRank(left=b.lr.l, right=d.lr.l, middle=m)
#     print('lr_a', B.shape(lr_a.l), B.shape(lr_a.m), B.shape(lr_a.r))
#     k1 = a.diag + B.lr_diff(a.lr, lr_a)
#     k2 = c.diag + B.lr_diff(c.lr, lr_c)
#     # print('k1 eigs:', B.eig(k1.lr.m)[0])
#     # print('k2 eigs:', B.eig(k2.lr.m)[0])
#     #
#     # if np.any(B.eig(k1.lr.m)[0] < -1e-2):
#     #     raise RuntimeError
#
#     # Precompute quantities that we'll reuse.
#     ik2 = B.inverse(k2)
#     print(m)
#     print('m before inversion eigs:', B.eig(m)[0])
#     icore = B.inverse(m) + B.matmul(y, ik2, y, tr_a=True)
#     core = B.inverse(icore)
#     p = B.matmul(y, ik2, y, tr_a=True)
#
#     # Construct blocks of the middle of the low-rank part.
#     m11 = m - B.matmul(m, B.matmul(y, B.inverse(c), y, tr_a=True), m)
#     m12 = B.matmul(core, p, m) - m
#     m22 = core
#
#     # Assemble the middle of the low-rank part.
#     middle = B.concat2d([m11, m12], [B.transpose(m12), m22])
#     eigs = B.eig(middle)[0]
#     print('middle eigs:', eigs)
#
#     # Compute left and right of the low-rank part.
#     left = B.concat([lr_a.l, dense(B.matmul(b.diag.T, ik2, y))], axis=1)
#     right = B.concat([lr_a.r, dense(B.matmul(d.diag.T, ik2, y))], axis=1)
#
#     # Assemble result.
#     res = k1 - \
#           B.matmul(b.diag.T, ik2, d.diag) + \
#           LowRank(left=left, right=right, middle=middle)
#
#     print('!!!!!!!!!!! done with this:')
#     print(res, B.shape(res.lr.l), B.shape(res.lr.m), B.shape(res.lr.r))
#
#     return res


@B.schur.extend(Woodbury, Woodbury, Woodbury, Woodbury)
def schur(a, b, c, d):
    ic = B.inverse(c)

    part1 = a
    part2 = B.matmul(b.diag.T, ic, d.diag)
    print('p1 shape:', B.shape(part1))
    print('p2 shape:', B.shape(part2))

    left = B.concat([b.lr.r, dense(B.matmul(b.diag.T, ic, d.lr.l))], axis=1)
    right = B.concat([d.lr.r, dense(B.matmul(d.diag.T, ic, b.lr.l))], axis=1)

    m11 = B.matmul(B.matmul(b.lr.l, b.lr.m), ic, B.matmul(d.lr.l, d.lr.m),
                   tr_a=True)
    m12 = B.transpose(b.lr.m)
    m21 = d.lr.m
    m22 = B.zeros([B.shape(m21)[0], B.shape(m12)[1]], dtype=B.dtype(m12))

    middle = B.concat2d([m11, m12], [m21, m22])

    lr = LowRank(left=left, right=right, middle=middle)


    res_lr = B.lr_diff(B.lr_diff(part1.lr, part2.lr), lr)


    res_diag = part1.diag + part2.diag
    res = res_lr + res_diag

    if B.shape(res)[0] == B.shape(res)[1]:
        min_eig = B.min(B.real(B.eig(res.lr.m)[0]))
        print('eigs res lr:', min_eig)
        if min_eig < -1e-2:
            raise RuntimeError
    return res



@B.schur2.extend(Woodbury, Woodbury, Woodbury, Woodbury)
def schur(a, b, c, d):
    # IMPORTANT: Compatibility is assumed here, which may or may not be true in
    # most cases. This needs to be further investigated.

    # The inverse of `c` has bits that we can use and reuse.
    ic = B.inverse(c)
    nicm = -ic.lr.m  # _N_egative _i_nverse `c` _m_iddle.

    # Compute blocks of the middle of the low-rank part.
    m11 = a.lr.m - \
          B.matmul(b.lr.m, B.matmul(b.lr.l, ic, d.lr.l, tr_a=True), d.lr.m)
    m12 = B.matmul(b.lr.m, B.matmul(b.lr.l, ic.lr.l, nicm, tr_a=True)) - \
          b.lr.m
    m21 = B.matmul(B.matmul(nicm, ic.lr.r, d.lr.l, tr_b=True), d.lr.m) - \
          d.lr.m
    m22 = nicm

    # Assemble the middle of the low-rank part.
    middle = B.concat2d([m11, m12], [m21, m22])

    # Compute left and right of the low-rank part.
    left = B.concat([a.lr.l, B.matmul(b.diag.T, ic.diag, d.lr.l)], axis=1)
    right = B.concat([a.lr.r, B.matmul(d.diag.T, ic.diag, b.lr.l)], axis=1)

    # Assemble result.
    diag = a.diag - B.matmul(b.diag.T, ic.diag, d.diag)
    lr = LowRank(left=left, right=right, middle=middle)

    return diag + lr


@B.convert.extend(LowRank, Woodbury)
def convert(a, _):
    diag = Diagonal(B.zeros([B.diag_len(a)], dtype=B.dtype(a)), *B.shape(a))
    return Woodbury(lr=a, diag=diag)


@B.schur.extend({Woodbury, LowRank},
                {Woodbury, LowRank},
                Woodbury,
                {Woodbury, LowRank})
def schur(a, b, c, d):
    return B.schur(B.convert(a, Woodbury),
                   B.convert(b, Woodbury),
                   B.convert(c, Woodbury),
                   B.convert(d, Woodbury))


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
    chol = B.cholesky(a)
    return B.matmul(B.trisolve(chol, b), B.trisolve(chol, c), tr_a=True)


@B.qf.extend({Diagonal, Woodbury}, object)
def qf(a, b): return B.qf(a, b, b)


@B.qf.extend({Diagonal, Woodbury}, object, object)
def qf(a, b, c): return B.matmul(b, B.inverse(a), c, tr_a=True)


@B.qf.extend_multi((LowRank, object), (LowRank, object, object))
def qf(a, b, c=None): raise RuntimeError('Matrix is singular.')


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


@B.qf_diag.extend({Diagonal, Woodbury}, object)
def qf_diag(a, b): return B.qf_diag(a, b, b)


@B.qf_diag.extend(Woodbury, object, object)
def qf_diag(a, b, c): return B.diag(B.qf(a, b, c))


@B.qf_diag.extend(Diagonal, object, object)
def qf_diag(a, b, c): return B.sum(B.matmul(B.inverse(a), b) * c, axis=0)


@B.qf_diag.extend_multi((LowRank, object), (LowRank, object, object))
def qf_diag(a, b, c=None): raise RuntimeError('Matrix is singular.')


# Compute ratios between matrices.

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
def ratio(a, b): return B.sum(B.qf_diag(b, a.right, B.matmul(a.left, a.middle)))


@B.ratio.extend(Diagonal, Diagonal)
def ratio(a, b): return B.sum(a.diag / b.diag)


# Transpose matrices.

@B.transpose.extend(Dense)
def transpose(a): return matrix(B.transpose(dense(a)))


@B.transpose.extend(Diagonal)
def transpose(a): return Diagonal(B.diag(a), *reversed(B.shape(a)))


@B.transpose.extend(LowRank)
def transpose(a): return LowRank(left=a.right,
                                 right=a.left,
                                 middle=B.transpose(a.middle))


@B.transpose.extend(Constant)
def transpose(a): return Constant(a.constant, *reversed(B.shape(a)))


@B.transpose.extend(Woodbury)
def transpose(a): return Woodbury(lr=B.transpose(a.lr),
                                  diag=B.transpose(a.diag))


# Extend LAB to the field of matrices.

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


# Get shapes of matrices.

@B.shape.extend(Dense)
def shape(a): return B.shape(dense(a))


#   Don't use a union here to prevent ambiguity errors with the implementation
#     for `LowRank`.
@B.shape.extend_multi((Diagonal,), (Constant,))
def shape(a): return a.rows, a.cols


@B.shape.extend(LowRank)
def shape(a): return B.shape(a.left)[0], B.shape(a.right)[0]


@B.shape.extend(Woodbury)
def shape(a): return B.shape(a.lr)


# Setup promotion and conversion of matrices as a fallback mechanism.

B.add_promotion_rule(B.Numeric, Dense, B.Numeric)
B.convert.extend(Dense, B.Numeric)(lambda x, _: dense(x))


# Simplify addiction and multiplication between matrices.

@mul.extend(Dense, Dense)
def mul(a, b): return Dense(dense(a) * dense(b))


@mul.extend(Dense, Diagonal)
def mul(a, b): return Diagonal(B.diag(a) * b.diag, *B.shape(a))


@mul.extend(Diagonal, Dense)
def mul(a, b): return Diagonal(a.diag * B.diag(b), *B.shape(a))


@mul.extend(Diagonal, Diagonal)
def mul(a, b): return Diagonal(a.diag * b.diag, *B.shape(a))


@mul.extend(LowRank, LowRank)
def mul(a, b):
    # Pick apart the matrices.
    al, ar = B.unstack(a.left, axis=1), B.unstack(a.right, axis=1)
    bl, br = B.unstack(b.left, axis=1), B.unstack(b.right, axis=1)
    am = [B.unstack(x, axis=0) for x in B.unstack(a.middle, axis=0)]
    bm = [B.unstack(x, axis=0) for x in B.unstack(b.middle, axis=0)]

    # Construct all parts of the product.
    left = B.stack([ali * blk
                    for ali in al
                    for blk in bl], axis=1)
    right = B.stack([arj * brl
                     for arj in ar
                     for brl in br], axis=1)
    middle = B.stack([B.stack([amij * bmkl
                               for amij in ami
                               for bmkl in bmk], axis=0)
                      for ami in am
                      for bmk in bm], axis=0)

    # And assemble the result.
    return LowRank(left=left, right=right, middle=middle)


@mul.extend(Dense, Woodbury, precedence=2)
def mul(a, b):
    # Prioritise expanding out the Woodbury matrix. Give this one even higher
    # precedence to resolve ambiguity in the case of two Woodbury matrices.
    return add(mul(a, b.lr), mul(a, b.diag))


@mul.extend(Woodbury, Dense, precedence=1)
def mul(a, b):
    # Prioritise expanding out the Woodbury matrix.
    return add(mul(a.lr, b), mul(a.diag, b))


@add.extend(Dense, Dense)
def add(a, b): return Dense(dense(a) + dense(b))


@add.extend(Diagonal, Diagonal)
def add(a, b): return Diagonal(B.diag(a) + B.diag(b), *B.shape(a))


@add.extend(LowRank, LowRank)
def add(a, b):
    shape_a, shape_b, dtype = B.shape(a.middle), B.shape(b.middle), B.dtype(a)
    middle = B.concat2d(
        [a.middle, B.zeros([shape_a[0], shape_b[1]], dtype=dtype)],
        [B.zeros([shape_b[0], shape_a[1]], dtype=dtype), b.middle]
    )
    return LowRank(left=B.concat([a.left, b.left], axis=1),
                   right=B.concat([a.right, b.right], axis=1),
                   middle=middle)


#   Woodbury matrices:

@add.extend(LowRank, Diagonal)
def add(a, b): return Woodbury(a, b)


@add.extend(Diagonal, LowRank)
def add(a, b): return Woodbury(b, a)


@add.extend(LowRank, Woodbury)
def add(a, b): return Woodbury(a + b.lr, b.diag)


@add.extend(Woodbury, LowRank)
def add(a, b): return Woodbury(a.lr + b, a.diag)


@add.extend(Diagonal, Woodbury)
def add(a, b): return Woodbury(b.lr, a + b.diag)


@add.extend(Woodbury, Diagonal)
def add(a, b): return Woodbury(a.lr, a.diag + b)


@add.extend(Woodbury, Woodbury)
def add(a, b): return Woodbury(a.lr + b.lr,
                               a.diag + b.diag)


# Multiplication between matrices and other elements.

@mul.extend(LowRank, Constant)
def mul(a, b): return LowRank(left=a.left,
                              right=a.right,
                              middle=a.middle * b.constant)


@mul.extend(Diagonal, Constant)
def mul(a, b): return Diagonal(B.diag(a) * b.constant, *B.shape(a))


@mul.extend(Constant, {LowRank, Diagonal})
def mul(a, b): return mul(b, a)


# Simplify addiction and multiplication between matrices and other objects. We
# immediately resolve scaled elements.

@mul.extend(object, Dense)
def mul(a, b): return mul(b, a)


@mul.extend(Dense, object)
def mul(a, b): return mul(a, mul(One.from_(a), b))


@add.extend(object, Dense)
def add(a, b): return add(b, a)


@add.extend(Dense, object)
def add(a, b): return add(a, mul(One.from_(a), b))


@mul.extend(object, One)
def mul(a, b): return mul(b, a)


@mul.extend(One, object)
def mul(a, b):
    if B.rank(b) == 0:
        if b is 0:
            return Zero.from_(a)
        elif b is 1:
            return a
        else:
            return Constant.from_(b, a)
    else:
        return matrix(dense(a) * b)


# Take over construction of ones: construction requires arguments.

@add.extend(object, Zero)
def add(a, b): return add(b, a)


@add.extend(Zero, object)
def add(a, b): return mul(b, One.from_(a))


@add.extend(Zero, Zero)
def add(a, b): return a
