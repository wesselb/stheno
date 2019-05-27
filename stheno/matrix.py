# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

import logging
from numbers import Number
from string import ascii_lowercase

import numpy as np
from lab import B
from plum import Referentiable, Self, Dispatcher, add_conversion_method, \
    add_promotion_rule

from .field import Element, OneElement, ZeroElement, mul, add, get_field

__all__ = ['matrix',
           'Dense',
           'LowRank',
           'Diagonal',
           'UniformlyDiagonal',
           'One',
           'Zero',
           'dense',
           'Woodbury',
           'Constant']

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
        return mul(B.cast(B.dtype(self), -1), self)

    def __div__(self, other):
        return B.divide(self, other)

    def __truediv__(self, other):
        return B.divide(self, other)

    def __rdiv__(self, other):
        return B.divide(other, self)

    def __rtruediv__(self, other):
        return B.divide(other, self)

    def __getitem__(self, item):
        return dense(self)[item]

    @property
    def T(self):
        return B.transpose(self)

    def __matmul__(self, other):
        return B.matmul(self, other)

    @_dispatch(Self)
    def __eq__(self, other):
        return dense(self) == dense(other)


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

    @_dispatch(Self)
    def __eq__(self, other):
        return B.diag(self) == B.diag(other)


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
        diag = diag_scale * B.ones(B.dtype(diag_scale), n)
        Diagonal.__init__(self, diag=diag, rows=rows, cols=cols)

    @classmethod
    def from_(cls, diag_scale, ref):
        return cls(B.cast(B.dtype(ref), diag_scale),
                   B.diag_len(ref),
                   *B.shape(ref))


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
            self.middle = B.eye(B.dtype(self.left), B.shape(self.left)[1])
        else:
            self.middle = middle

        # Shorthands:
        self.l = self.left
        self.r = self.right
        self.m = self.middle

    @_dispatch(Self)
    def __eq__(self, other):
        return (self.left == other.left,
                self.middle == other.middle,
                self.right == other.right)


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
        left = B.ones(B.dtype(self.constant), self.rows, 1)
        if self.rows is self.cols:
            right = left
        else:
            right = B.ones(B.dtype(self.constant), self.cols, 1)
        middle = B.expand_dims(B.expand_dims(self.constant, axis=0), axis=0)
        LowRank.__init__(self, left=left, right=right, middle=middle)

    @classmethod
    def from_(cls, constant, ref):
        return cls(B.cast(B.dtype(ref), constant), *B.shape(ref))

    def __eq__(self, other):
        return B.shape(self) == B.shape(other) \
               and self.constant == other.constant


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
        Constant.__init__(self, B.cast(dtype, 1), rows=rows, cols=cols)

    @classmethod
    def from_(cls, ref):
        return cls(B.dtype(ref), *B.shape(ref))

    @_dispatch(Self)
    def __eq__(self, other):
        return B.shape(self) == B.shape(other)


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
        Constant.__init__(self, B.cast(dtype, 0), rows=rows, cols=cols)

    @classmethod
    def from_(cls, ref):
        return cls(B.dtype(ref), *B.shape(ref))

    @_dispatch(Self)
    def __eq__(self, other):
        return B.shape(self) == B.shape(other)


class Woodbury(Dense, Referentiable):
    """Sum of a low-rank and diagonal symmetric positive-definite matrix.

    Args:
        diag (:class:`.matrix.Diagonal`): Diagonal part.
        lr (:class:`.matrix.LowRank`): Low-rank part.
        lr_pd (bool, optional): Specify that the low-rank part is PD. Defaults
            to `True`.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, diag, lr, lr_pd=True):
        Dense.__init__(self, None)
        self.diag = diag
        self.lr = lr
        self.lr_pd = lr_pd

        # Caching:
        self.schur = None

    @_dispatch(Self)
    def __eq__(self, other):
        return self.lr == other.lr, self.diag == other.diag


# Conveniently make identity matrices.

@B.eye.extend(Dense)
def eye(a): return B.eye(B.dtype(a), *B.shape(a))


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


# Construct a matrix from blocks.

def block_matrix(*rows):
    """Construct a matrix from its blocks, preserving structure when possible.

    Args:
        *rows (list): Rows of the block matrix.

    Returns:
        :class:`.matrix.Dense`: Assembled matrix as structured as possible.
    """
    # If the blocks form a grid, then might know how to preserve structure.
    grid = True

    # Check that all shapes line up.
    for r in range(1, len(rows)):
        for c in range(1, len(rows[0])):
            block_shape = B.shape(rows[r][c])

            # The previous block in the row must have an equal
            # number of rows.
            if block_shape[0] != B.shape(rows[r][c - 1])[0]:
                grid = False

            # The previous block in the column must have an equal
            # number of columns.
            if block_shape[1] != B.shape(rows[r - 1][c])[1]:
                grid = False

    if grid:
        # We have a grid! First, determine the resulting shape.
        grid_rows = _builtin_sum([B.shape(row[0])[0] for row in rows])
        grid_cols = _builtin_sum([B.shape(K)[1] for K in rows[0]])

        # Check whether the result is just zeros.
        if all([all([isinstance(K, Zero) for K in row]) for row in rows]):
            return Zero(B.dtype(rows[0][0]),
                        rows=grid_rows,
                        cols=grid_cols)

        # Check whether the result is just ones.
        if all([all([isinstance(K, One) for K in row]) for row in rows]):
            return One(B.dtype(rows[0][0]),
                       rows=grid_rows,
                       cols=grid_cols)

        # Check whether the result is diagonal.
        diagonal = True
        diagonal_blocks = []

        for r in range(len(rows)):
            for c in range(len(rows[0])):
                block_shape = B.shape(rows[r][c])
                if r == c:
                    # Keep track of all the diagonal blocks.
                    diagonal_blocks.append(rows[r][c])

                    # All blocks on the diagonal must be diagonal or zero.
                    if not isinstance(rows[r][c], (Diagonal, Zero)):
                        diagonal = False

                    # All blocks on the diagonal must be square.
                    if not block_shape[0] == block_shape[1]:
                        diagonal = False

                # All blocks not on the diagonal must be zero.
                if r != c and not isinstance(rows[r][c], Zero):
                    diagonal = False

        if diagonal:
            return Diagonal(B.concat(*[B.diag(K) for K in diagonal_blocks],
                                     axis=0),
                            rows=grid_rows,
                            cols=grid_cols)

    # We could not preserve any structure. Simply concatenate them all
    # densely.
    return Dense(B.concat2d(*[[dense(K) for K in row] for row in rows]))


B.block_matrix = block_matrix  # Record in LAB.


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

_builtin_sum = sum  # Save built-in function.


@B.sum.extend(Dense, [object])
def sum(a, axis=None): return B.sum(dense(a), axis=axis)


@B.sum.extend(Diagonal, [object])
def sum(a, axis=None):
    # Efficiently handle a number of common cases.
    if axis is None:
        return B.sum(B.diag(a))
    elif axis is 0:
        return B.concat(B.diag(a),
                        B.zeros(B.dtype(a), B.shape(a)[1] - B.diag_len(a)),
                        axis=0)
    elif axis is 1:
        return B.concat(B.diag(a),
                        B.zeros(B.dtype(a), B.shape(a)[0] - B.diag_len(a)),
                        axis=0)
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

def diag_len(a):
    """Get the length of the diagonal of a matrix.

    Args:
        a (tensor): Matrix of which to get the length of the diagonal.

    Returns:
        tensor: Length of the diagonal of `a`.
    """
    return B.minimum(*B.shape(a))


B.diag_len = diag_len  # Record in LAB.


# Get diagonals of matrices or create diagonal matrices.

@B.diag.extend(Dense)
def diag(a): return B.diag(dense(a))


@B.diag.extend(Diagonal)
def diag(a):
    # Append zeros or remove elements as necessary.
    diag_len = B.diag_len(a)
    extra_zeros = B.maximum(diag_len - B.shape(a.diag)[0], 0)
    return B.concat(a.diag[:diag_len], B.zeros(B.dtype(a), extra_zeros), axis=0)


@B.diag.extend(LowRank)
def diag(a):
    # The matrix might be non-square, so handle that.
    diag_len = B.diag_len(a)
    return B.sum(B.matmul(a.left, a.middle)[:diag_len, :] *
                 a.right[:diag_len, :], axis=1)


@B.diag.extend(Constant)
def diag(a): return a.constant * B.ones(B.dtype(a), B.diag_len(a))


@B.diag.extend(Woodbury)
def diag(a): return B.diag(a.lr) + B.diag(a.diag)


@B.diag.extend(B.Numeric, B.Numeric, [B.Numeric])
def diag(diag, rows, cols=None):
    cols = rows if cols is None else cols

    # Cut the diagonal to accommodate the size.
    diag = diag[:B.minimum(rows, cols)]
    diag_len, dtype = B.shape(diag)[0], B.dtype(diag)

    # PyTorch incorrectly handles dimensions of size 0. Therefore, if the
    # numbers of extra columns and rows are `Number`s, which will be the case if
    # PyTorch is the backend, then perform a check to prevent appending tensors
    # with dimensions of size 0.

    # Start with just a diagonal matrix.
    res = B.diag(diag)

    # Pad extra columns if necessary.
    extra_cols = cols - diag_len
    if not (isinstance(extra_cols, Number) and extra_cols == 0):
        zeros = B.zeros(dtype, diag_len, extra_cols)
        res = B.concat(B.diag(diag), zeros, axis=1)

    # Pad extra rows if necessary.
    extra_rows = rows - diag_len
    if not (isinstance(extra_rows, Number) and extra_rows == 0):
        zeros = B.zeros(dtype, extra_rows, diag_len + extra_cols)
        res = B.concat(res, zeros, axis=0)

    return res


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
        a.cholesky = B.matmul(a.left, B.cholesky(matrix(a.middle)))
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
    if a_rows == a_cols and a_rows is not None:
        return B.diag(a)[:, None] * dense(b)

    # Compute the core part.
    rows = B.minimum(a_rows, B.shape(b)[0])
    core = B.diag(a)[:rows, None] * dense(b)[:rows, :]

    # Compute extra zeros to be appended.
    extra_rows = a_rows - rows
    extra_zeros = B.zeros(B.dtype(b), extra_rows, B.shape(b)[1])
    return B.concat(core, extra_zeros, axis=0)


@B.matmul.extend({B.Numeric, Dense}, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    a = B.transpose(a) if tr_a else a
    b = B.transpose(b) if tr_b else b

    # Get shape of `b`.
    b_rows, b_cols = B.shape(b)

    # If `b` is square, don't do complicated things.
    if b_rows == b_cols and b_rows is not None:
        return dense(a) * B.diag(b)[None, :]

    # Compute the core part.
    cols = B.minimum(B.shape(a)[1], b_cols)
    core = dense(a)[:, :cols] * B.diag(b)[None, :cols]

    # Compute extra zeros to be appended.
    extra_cols = b_cols - cols
    extra_zeros = B.zeros(B.dtype(b), B.shape(a)[0], extra_cols)
    return B.concat(core, extra_zeros, axis=1)


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


@B.matmul.extend({B.Numeric, Dense},
                 {B.Numeric, Dense},
                 {B.Numeric, Dense},
                 [{B.Numeric, Dense}])
def matmul(*xs, **trs):
    def tr(name):
        return trs['tr_' + name] if 'tr_' + name in trs else False

    # Compute the first product.
    res = B.matmul(xs[0], xs[1], tr_a=tr('a'), tr_b=tr('b'))

    # Compute the remaining products.
    for name, x in zip(ascii_lowercase[2:], xs[2:]):
        res = B.matmul(res, x, tr_b=tr(name))

    return res


# Matrix inversion:

@_dispatch({B.Numeric, Dense})
def inverse(a):
    # Assume that `a` is PD.
    a = matrix(a)
    if a.inverse is None:
        inv_prod = B.trisolve(B.cholesky(a), B.eye(a))
        a.inverse = B.matmul(inv_prod, inv_prod, tr_a=True)
    return a.inverse


B.inverse = inverse  # Record in LAB.


@B.inverse.extend(Diagonal)
def inverse(a): return Diagonal(1 / B.diag(a), *reversed(B.shape(a)))


@B.inverse.extend(Woodbury)
def inverse(a):
    # Use the Woodbury matrix identity.
    if a.inverse is None:
        inv_diag = B.inverse(a.diag)
        lr = LowRank(left=B.matmul(inv_diag, a.lr.left),
                     right=B.matmul(inv_diag, a.lr.right),
                     middle=B.inverse(B.schur(a)))
        if a.lr_pd:
            a.inverse = Woodbury(diag=inv_diag, lr=-lr, lr_pd=False)
        else:
            a.inverse = Woodbury(diag=inv_diag, lr=lr, lr_pd=True)
    return a.inverse


def lr_diff(a, b):
    """Subtract two low-rank matrices, forcing the resulting middle part to
    be positive definite if the result is so.

    Args:
        a (:class:`.matrix.LowRank`): `a`.
        b (:class:`.matrix.LowRank`): `b`.

    Returns:
        :class:`.matrix.LowRank`: Difference between `a` and `b`.
    """
    diff = a - b
    u_left, s_left, v_left = B.svd(diff.left)
    u_right, s_right, v_right = B.svd(diff.right)
    middle = B.matmul(Diagonal(s_left, *B.shape(diff.left)),
                      B.matmul(v_left, diff.middle, v_right, tr_a=True),
                      Diagonal(s_right, *B.shape(diff.right)).T)
    return LowRank(left=u_left,
                   right=u_right,
                   middle=middle)


B.lr_diff = lr_diff  # Record in LAB.


@B.inverse.extend(LowRank)
def inverse(a): raise RuntimeError('Matrix is singular.')


# Compute the log-determinant of matrices.

@_dispatch({B.Numeric, Dense})
def logdet(a):
    a = matrix(a)
    if a.logdet is None:
        a.logdet = 2 * B.sum(B.log(B.diag(B.cholesky(a))))
    return a.logdet


B.logdet = logdet  # Record in LAB.


@B.logdet.extend(Diagonal)
def logdet(a): return B.sum(B.log(a.diag))


@B.logdet.extend(Woodbury)
def logdet(a):
    return B.logdet(B.schur(a)) + \
           B.logdet(a.diag) + \
           B.logdet(a.lr.middle if a.lr_pd else -a.lr.middle)


@B.logdet.extend(LowRank)
def logdet(a): raise RuntimeError('Matrix is singular.')


# Compute roots of matrices.

@_dispatch({B.Numeric, Dense})
def root(a):
    # TODO: This assumed that `a` is PSD.
    a = matrix(a)
    if a.root is None:
        u, s, _ = B.svd(dense(a), compute_uv=True)
        a.root = B.matmul(u * s[None, :] ** .5, u, tr_b=True)
    return a.root


B.root = root  # Record in LAB.


@B.root.extend(Diagonal)
def root(a):
    return Diagonal(a.diag ** .5)


# Sample from covariance matrices.

@_dispatch({B.Numeric, Dense}, [object])
def sample(a, num=1):
    """Sample from covariance matrices.

    Args:
        a (tensor): Covariance matrix to sample from.
        num (int): Number of samples.

    Returns:
        tensor: Samples as rank 2 column vectors.
    """
    # # Convert integer data types to floats.
    dtype = float if B.issubdtype(B.dtype(a), np.integer) else B.dtype(a)

    # Perform sampling operation.
    chol = B.cholesky(matrix(a))
    return B.matmul(chol, B.randn(dtype, B.shape(chol)[1], num))


B.sample = sample  # Record in LAB.


@B.sample.extend(Woodbury, [object])
def sample(a, num=1): return B.sample(a.diag, num) + B.sample(a.lr, num)


# Compute Schur complements.

@_dispatch(object, object, object, object)
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


B.schur = schur  # Record in LAB.


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
        prod = B.matmul(a.lr.right, B.inverse(a.diag), a.lr.left, tr_a=True)
        if a.lr_pd:
            a.schur = B.inverse(a.lr.middle) + prod
        else:
            a.schur = B.inverse(-a.lr.middle) - prod
    return a.schur


# Compute quadratic forms and diagonals thereof.

@_dispatch(object, object)
def qf(a, b):
    """Compute the quadratic form `transpose(b) inv(a) c`.

    Args:
        a (tensor): Covariance matrix.
        b (tensor): `b`.
        c (tensor, optional): `c`. Defaults to `b`.

    Returns:
        :class:`.matrix.Dense`: Quadratic form.
    """
    prod = B.trisolve(B.cholesky(matrix(a)), b)
    return matrix(B.matmul(prod, prod, tr_a=True))


B.qf = qf  # Record in LAB.


@B.qf.extend(object, object, object)
def qf(a, b, c):
    if b is c:
        return B.qf(a, b)
    chol = B.cholesky(matrix(a))
    return B.matmul(B.trisolve(chol, b), B.trisolve(chol, c), tr_a=True)


@B.qf.extend({Diagonal, Woodbury}, object)
def qf(a, b): return B.qf(a, b, b)


@B.qf.extend({Diagonal, Woodbury}, object, object)
def qf(a, b, c): return B.matmul(b, B.inverse(a), c, tr_a=True)


@B.qf.extend_multi((LowRank, object), (LowRank, object, object))
def qf(a, b, c=None): raise RuntimeError('Matrix is singular.')


@_dispatch(object, object)
def qf_diag(a, b):
    """Compute the diagonal of `transpose(b) inv(a) c`.

    Args:
        a (:class:`.matrix.Dense`): Covariance matrix.
        b (tensor): `b`.
        c (tensor, optional): `c`. Defaults to `b`.

    Returns:
        tensor: Diagonal of the quadratic form.
    """
    prod = B.trisolve(B.cholesky(matrix(a)), b)
    return B.sum(prod ** 2, axis=0)


B.qf_diag = qf_diag  # Record in LAB.


@B.qf_diag.extend(object, object, object)
def qf_diag(a, b, c):
    a = matrix(a)
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

@_dispatch(object, object)
def ratio(a, b):
    """Compute the ratio between two positive-definite matrices.

    Args:
        a (tensor): Numerator.
        b (tensor): Denominator.

    Returns:
        tensor: Ratio.
    """
    return B.sum(B.qf_diag(b, B.cholesky(matrix(a))))


B.ratio = ratio  # Record in LAB.


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
def transpose(a): return Woodbury(diag=B.transpose(a.diag),
                                  lr=B.transpose(a.lr),
                                  lr_pd=a.lr_pd)


# Extend LAB to the field of matrices.

B.add.extend(Dense, object)(add)
B.add.extend(object, Dense)(add)
B.add.extend(Dense, Dense)(add)

B.multiply.extend(Dense, object)(mul)
B.multiply.extend(object, Dense)(mul)
B.multiply.extend(Dense, Dense)(mul)


@B.subtract.extend(Dense, Dense)
def subtract(a, b): return B.add(a, -b)


@B.divide.extend(Dense, Dense)
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

add_promotion_rule(B.Numeric, Dense, B.Numeric)
add_conversion_method(Dense, B.Numeric, dense)


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
    left = B.stack(*[ali * blk
                     for ali in al
                     for blk in bl], axis=1)
    right = B.stack(*[arj * brl
                      for arj in ar
                      for brl in br], axis=1)
    middle = B.stack(*[B.stack(*[amij * bmkl
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
    middle = B.concat2d([a.middle, B.zeros(dtype, shape_a[0], shape_b[1])],
                        [B.zeros(dtype, shape_b[0], shape_a[1]), b.middle])
    return LowRank(left=B.concat(a.left, b.left, axis=1),
                   right=B.concat(a.right, b.right, axis=1),
                   middle=middle)


# Woodbury matrices:

@add.extend(LowRank, Diagonal)
def add(a, b): return Woodbury(diag=b, lr=a)


@add.extend(Diagonal, LowRank)
def add(a, b): return Woodbury(diag=a, lr=b)


@add.extend(LowRank, Woodbury)
def add(a, b): return Woodbury(diag=b.diag, lr=a + b.lr)


@add.extend(Woodbury, LowRank)
def add(a, b): return Woodbury(diag=a.diag, lr=a.lr + b)


@add.extend(Diagonal, Woodbury)
def add(a, b): return Woodbury(diag=a + b.diag, lr=b.lr)


@add.extend(Woodbury, Diagonal)
def add(a, b): return Woodbury(diag=a.diag + b, lr=a.lr)


@add.extend(Woodbury, Woodbury)
def add(a, b): return Woodbury(diag=a.diag + b.diag, lr=a.lr + b.lr)


# Multiplication between matrices and constants.

@mul.extend(LowRank, Constant)
def mul(a, b): return LowRank(left=a.left,
                              right=a.right,
                              middle=a.middle * b.constant)


@mul.extend(Constant, LowRank)
def mul(a, b): return LowRank(left=b.left,
                              right=b.right,
                              middle=a.constant * b.middle)


@mul.extend(Diagonal, Constant)
def mul(a, b): return Diagonal(B.diag(a) * b.constant, *B.shape(a))


@mul.extend(Constant, Diagonal)
def mul(a, b): return Diagonal(a.constant * B.diag(b), *B.shape(a))


@mul.extend(Constant, Constant)
def mul(a, b): return Constant(a.constant * b.constant, *B.shape(a))


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
