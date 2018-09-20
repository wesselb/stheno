# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import product

import logging
import numpy as np
from lab import B
from stheno.matrix import Dense, Diagonal, UniformlyDiagonal, LowRank, dense, \
    One, Zero, Woodbury, Constant, matrix

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint, allclose


def test_dense_methods():
    a = Dense(np.random.randn(10, 5))

    yield allclose, dense(a.T), dense(a).T
    yield allclose, dense(-a), -dense(a)
    yield allclose, dense(a[5]), dense(a)[5]


def test_eye_form():
    a = Dense(np.random.randn(5, 10))
    yield allclose, dense(B.eye_from(a)), np.eye(5, 10)
    yield allclose, dense(B.eye_from(a.T)), np.eye(10, 5)


def test_matrix():
    a = np.random.randn(5, 3)
    yield eq, type(matrix(a)), Dense
    yield allclose, matrix(a).mat, a
    yield eq, type(matrix(matrix(a))), Dense
    yield allclose, matrix(matrix(a)).mat, a


def test_dense():
    a = np.random.randn(5, 3)
    yield allclose, dense(Dense(a)), a
    yield allclose, dense(a), a

    # Extensively test Diagonal.
    yield allclose, \
          dense(Diagonal([1, 2])), \
          np.array([[1, 0],
                    [0, 2]])
    yield allclose, \
          dense(Diagonal([1, 2], 3)), \
          np.array([[1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0]])
    yield allclose, \
          dense(Diagonal([1, 2], 1)), \
          np.array([[1]])
    yield allclose, \
          dense(Diagonal([1, 2], (3, 3))), \
          np.array([[1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0]])
    yield allclose, \
          dense(Diagonal([1, 2], (2, 3))), \
          np.array([[1, 0, 0],
                    [0, 2, 0]])
    yield allclose, \
          dense(Diagonal([1, 2], (3, 2))), \
          np.array([[1, 0],
                    [0, 2],
                    [0, 0]])
    yield allclose, \
          dense(Diagonal([1, 2], (1, 3))), \
          np.array([[1, 0, 0]])
    yield allclose, \
          dense(Diagonal([1, 2], (3, 1))), \
          np.array([[1],
                    [0],
                    [0]])

    # Test low-rank matrices.
    left = np.random.randn(5, 3)
    right = np.random.randn(10, 3)
    middle = np.random.randn(3, 3)
    lr = LowRank(left=left, right=right, middle=middle)
    yield allclose, dense(lr), left.dot(middle).dot(right.T)

    # Test Woodbury matrices.
    diag = Diagonal([1, 2, 3, 4], (5, 10))
    wb = Woodbury(lr, diag)
    yield allclose, dense(wb), dense(diag) + dense(lr)


def test_dtype():
    # Test `Dense`.
    yield eq, B.dtype(Dense(np.array([[1]]))), int
    yield eq, B.dtype(Dense(np.array([[1.0]]))), float

    # Test `Diagonal`.
    diag_int = Diagonal(np.array([1]))
    diag_float = Diagonal(np.array([1.0]))
    yield eq, B.dtype(diag_int), int
    yield eq, B.dtype(diag_float), float

    # Test `LowRank`.
    lr_int = LowRank(left=np.array([[1]]),
                     right=np.array([[2]]),
                     middle=np.array([[3]]))
    lr_float = LowRank(left=np.array([[1.0]]),
                       right=np.array([[2.0]]),
                       middle=np.array([[3.0]]))
    yield eq, B.dtype(lr_int), int
    yield eq, B.dtype(lr_float), float

    # Test `Constant`.
    yield eq, B.dtype(Constant(1, shape=1)), int
    yield eq, B.dtype(Constant(1.0, shape=1)), float

    # Test `Woodbury`.
    yield eq, B.dtype(Woodbury(lr_int, diag_int)), int
    yield eq, B.dtype(Woodbury(lr_float, diag_float)), float


def test_diag():
    # Test `Dense`.
    a = np.random.randn(5, 3)
    yield allclose, B.diag(Dense(a)), np.diag(a)

    # Test `Diagonal`.
    yield allclose, B.diag(Diagonal([1, 2, 3])), [1, 2, 3]
    yield allclose, B.diag(Diagonal([1, 2, 3], 2)), [1, 2]
    yield allclose, B.diag(Diagonal([1, 2, 3], 4)), [1, 2, 3, 0]

    # Test `LowRank`.
    b = np.random.randn(10, 3)
    yield allclose, B.diag(LowRank(left=a, right=a)), np.diag(a.dot(a.T))
    yield allclose, B.diag(LowRank(left=a, right=b)), np.diag(a.dot(b.T))
    yield allclose, B.diag(LowRank(left=b, right=b)), np.diag(b.dot(b.T))

    # Test `Constant`.
    yield allclose, B.diag(Constant(1, shape=(3, 5))), np.ones(3)

    # Test `Woodbury`.
    yield allclose, B.diag(Woodbury(LowRank(left=a, right=b),
                                    Diagonal([1, 2, 3], shape=(5, 10)))), \
          np.diag(a.dot(b.T) + np.concatenate((np.diag([1, 2, 3, 0, 0]),
                                               np.zeros((5, 5))), axis=1))


def test_cholesky():
    a = np.random.randn(5, 5)
    a = a.T.dot(a)

    # Test `Dense`.
    yield allclose, np.linalg.cholesky(a), B.cholesky(a)

    # Test `Diagonal`.
    d = Diagonal(np.diag(a))
    yield allclose, np.linalg.cholesky(dense(d)), B.cholesky(d)
    yield raises, RuntimeError, lambda: B.cholesky(Diagonal([1], shape=(2, 1)))


def test_matmul():
    diag_square = Diagonal([1, 2], 3)
    diag_tall = Diagonal([3, 4], (5, 3))
    diag_wide = Diagonal([5, 6], (2, 3))

    dense_square = Dense(np.random.randn(3, 3))
    dense_tall = Dense(np.random.randn(5, 3))
    dense_wide = Dense(np.random.randn(2, 3))

    lr = LowRank(left=np.random.randn(5, 2),
                 right=np.random.randn(3, 2),
                 middle=np.random.randn(2, 2))

    def compare_matmul(a, b, desc=None):
        allclose(B.matmul(a, b), B.matmul(dense(a), dense(b)))

    # Test `Dense`.
    yield compare_matmul, dense_wide, dense_tall.T, 'dense w x dense t'

    # Test `LowRank`.
    yield compare_matmul, lr, dense_tall.T, 'lr x dense t'
    yield compare_matmul, dense_wide, lr.T, 'dense w x lr'
    yield compare_matmul, lr, diag_tall.T, 'lr x diag t'
    yield compare_matmul, diag_wide, lr.T, 'diag w x lr'
    yield compare_matmul, lr, lr.T, 'lr x lr'
    yield compare_matmul, lr.T, lr, 'lr x lr (2)'

    # Test `Diagonal`.
    #   Test multiplication between diagonal matrices.
    yield compare_matmul, diag_square, diag_square.T, 'diag s x diag s'
    yield compare_matmul, diag_tall, diag_square.T, 'diag t x diag s'
    yield compare_matmul, diag_wide, diag_square.T, 'diag w x diag s'
    yield compare_matmul, diag_square, diag_tall.T, 'diag s x diag t'
    yield compare_matmul, diag_tall, diag_tall.T, 'diag t x diag t'
    yield compare_matmul, diag_wide, diag_tall.T, 'diag w x diag t'
    yield compare_matmul, diag_square, diag_wide.T, 'diag s x diag w'
    yield compare_matmul, diag_tall, diag_wide.T, 'diag t x diag w'
    yield compare_matmul, diag_wide, diag_wide.T, 'diag w x diag w'

    #   Test multiplication between diagonal and dense matrices.
    yield compare_matmul, diag_square, dense_square.T, 'diag s x dense s'
    yield compare_matmul, diag_square, dense_tall.T, 'diag s x dense t'
    yield compare_matmul, diag_square, dense_wide.T, 'diag s x dense w'
    yield compare_matmul, diag_tall, dense_square.T, 'diag t x dense s'
    yield compare_matmul, diag_tall, dense_tall.T, 'diag t x dense t'
    yield compare_matmul, diag_tall, dense_wide.T, 'diag t x dense w'
    yield compare_matmul, diag_wide, dense_square.T, 'diag w x dense s'
    yield compare_matmul, diag_wide, dense_tall.T, 'diag w x dense t'
    yield compare_matmul, diag_wide, dense_wide.T, 'diag w x dense w'

    yield compare_matmul, dense_square, diag_square.T, 'dense s x diag s'
    yield compare_matmul, dense_square, diag_tall.T, 'dense s x diag t'
    yield compare_matmul, dense_square, diag_wide.T, 'dense s x diag w'
    yield compare_matmul, dense_tall, diag_square.T, 'dense t x diag s'
    yield compare_matmul, dense_tall, diag_tall.T, 'dense t x diag t'
    yield compare_matmul, dense_tall, diag_wide.T, 'dense t x diag w'
    yield compare_matmul, dense_wide, diag_square.T, 'dense w x diag s'
    yield compare_matmul, dense_wide, diag_tall.T, 'dense w x diag t'
    yield compare_matmul, dense_wide, diag_wide.T, 'dense w x diag w'

    # Test `B.matmul` with three matrices simultaneously.
    yield allclose, \
          B.matmul(dense_tall, dense_square, dense_wide, tr_c=True), \
          dense(dense_tall).dot(dense(dense_square)).dot(dense(dense_wide).T)

    # Test `Woodbury`.
    wb = lr + dense_tall
    yield compare_matmul, wb, dense_square.T, 'wb x dense s'
    yield compare_matmul, dense_square, wb.T, 'dense s x wb'
    yield compare_matmul, wb, wb.T, 'wb x wb'
    yield compare_matmul, wb.T, wb, 'wb x wb (2)'


def test_inverse_and_logdet():
    # Test `Dense`.
    a = np.random.randn(5, 5)
    a = Dense(a.dot(a.T))
    yield allclose, B.matmul(a, B.inverse(a)), np.eye(5)
    yield allclose, B.matmul(B.inverse(a), a), np.eye(5)
    yield allclose, B.logdet(a), np.log(np.linalg.det(dense(a)))

    # Test `Diagonal`.
    d = Diagonal([1, 2, 3, 4, 5])
    yield allclose, B.matmul(d, B.inverse(d)), np.eye(5)
    yield allclose, B.matmul(B.inverse(d), d), np.eye(5)
    yield allclose, B.logdet(d), np.log(np.linalg.det(dense(d)))
    yield eq, B.shape(B.inverse(Diagonal([1, 2], shape=(2, 4)))), (4, 2)

    # Test `Woodbury`.
    a = np.random.randn(5, 3)
    b = np.random.randn(3, 3)
    wb = d + LowRank(left=a, right=a, middle=b.dot(b.T))
    yield allclose, B.matmul(wb, B.inverse(wb)), np.eye(5)
    yield allclose, B.matmul(B.inverse(wb), wb), np.eye(5)
    yield allclose, B.logdet(wb), np.log(np.linalg.det(dense(wb)))

    # Test `LowRank`.
    yield raises, RuntimeError, lambda: B.inverse(wb.lr)
    yield raises, RuntimeError, lambda: B.logdet(wb.lr)


def test_root():
    # Test `Dense`.
    a = np.random.randn(5, 5)
    a = Dense(a.dot(a.T))
    yield allclose, a, B.matmul(B.root(a), B.root(a))

    # Test `Diagonal`.
    d = Diagonal(np.array([1, 2, 3, 4, 5]))
    yield allclose, d, B.matmul(B.root(d), B.root(d))


def test_schur():
    # Test `Dense`.
    a = np.random.randn(5, 10)
    b = np.random.randn(3, 5)
    c = np.random.randn(3, 3)
    d = np.random.randn(3, 10)
    c = c.dot(c.T)

    yield allclose, B.schur(a, b, c, d), \
          a - np.linalg.solve(c.T, b).T.dot(d), 'np np np np'

    # Test `Woodbury`.
    #   The inverse of the Woodbury matrix already properly tests the method for
    #   Woodbury matrices.
    c = np.random.randn(2, 2)
    c = Diagonal(np.array([1, 2, 3])) + \
        LowRank(left=np.random.randn(3, 2), middle=c.dot(c.T))
    yield allclose, B.schur(a, b, c, d), \
          a - np.linalg.solve(dense(c).T, b).T.dot(d), 'np np wb np'

    #   Test all combinations of `Woodbury`, `LowRank`, and `Diagonal`.
    a = np.random.randn(2, 2)
    a = Diagonal(np.array([4, 5, 6, 7, 8]), shape=(5, 10)) + \
        LowRank(left=np.random.randn(5, 2),
                right=np.random.randn(10, 2),
                middle=a.dot(a.T))

    b = np.random.randn(2, 2)
    b = Diagonal(np.array([9, 10, 11]), shape=(3, 5)) + \
        LowRank(left=c.lr.left,
                right=a.lr.left,
                middle=b.dot(b.T))

    d = np.random.randn(2, 2)
    d = Diagonal(np.array([12, 13, 14]), shape=(3, 10)) + \
        LowRank(left=c.lr.right,
                right=a.lr.right,
                middle=d.dot(d.T))

    #     Loop over all combinations. Some of them should be efficient and
    #     representation preserving; all of them should be correct.
    for ai in [a, a.lr, a.diag]:
        for bi in [b, b.lr, b.diag]:
            for ci in [c, c.diag]:
                for di in [d, d.lr, d.diag]:
                    yield allclose, B.schur(ai, bi, ci, di), \
                          dense(ai) - \
                          np.linalg.solve(dense(ci).T,
                                          dense(bi)).T.dot(dense(di)), \
                          '{} {} {} {}'.format(ai, bi, ci, di)


def test_qf():
    a = np.random.randn(5, 5)
    a = Dense(a.dot(a.T))
    b, c = np.random.randn(5, 3), np.random.randn(5, 3)

    # Test `Dense`.
    yield allclose, B.qf(a, b), np.linalg.solve(dense(a), b).T.dot(b)
    yield allclose, B.qf(a, b, c), np.linalg.solve(dense(a), b).T.dot(c)
    yield allclose, B.qf_diag(a, b), \
          np.diag(np.linalg.solve(dense(a), b).T.dot(b))
    yield allclose, B.qf_diag(a, b, c), \
          np.diag(np.linalg.solve(dense(a), b).T.dot(c))

    # Test `Diagonal` (and `Woodbury`).
    d = Diagonal(B.diag(a))
    yield allclose, B.qf(d, b), np.linalg.solve(dense(d), b).T.dot(b)
    yield allclose, B.qf(d, b, c), np.linalg.solve(dense(d), b).T.dot(c)
    yield allclose, B.qf_diag(d, b), \
          np.diag(np.linalg.solve(dense(d), b).T.dot(b))
    yield allclose, B.qf_diag(d, b, c), \
          np.diag(np.linalg.solve(dense(d), b).T.dot(c))

    # Test `LowRank`.
    lr = LowRank(np.random.randn(5, 3))
    yield raises, RuntimeError, lambda: B.qf(lr, b)
    yield raises, RuntimeError, lambda: B.qf(lr, b, c)
    yield raises, RuntimeError, lambda: B.qf_diag(lr, b)
    yield raises, RuntimeError, lambda: B.qf_diag(lr, b, c)


def test_ratio():
    a, b = np.random.randn(5, 5), np.random.randn(5, 5)
    a, b = Dense(a.dot(a.T)), Dense(b.dot(b.T))
    d, e = Diagonal(B.diag(a)), Diagonal(B.diag(b))
    c = np.random.randn(3, 3)
    lr = LowRank(left=np.random.randn(5, 3), middle=c.dot(c.T))

    yield allclose, B.ratio(a, b), \
          np.trace(np.linalg.solve(dense(b), dense(a)))
    yield allclose, B.ratio(lr, b), \
          np.trace(np.linalg.solve(dense(b), dense(lr)))
    yield allclose, B.ratio(d, e), \
          np.trace(np.linalg.solve(dense(e), dense(d)))


def test_transposition():
    def compare_transposition(a):
        allclose(B.transpose(a), dense(a).T)

    d = Diagonal([1, 2, 3], shape=(5, 10))
    lr = LowRank(left=np.random.randn(5, 3), right=np.random.randn(10, 3))

    yield compare_transposition, Dense(np.random.randn(5, 5))
    yield compare_transposition, d
    yield compare_transposition, lr
    yield compare_transposition, d + lr
    yield compare_transposition, Constant(5, shape=(4, 2))


def test_arithmetic_and_shapes():
    a = Dense(np.random.randn(4, 3))
    d = Diagonal(np.array([1, 2, 3]), shape=(4, 3))
    lr = LowRank(left=np.random.randn(4, 2),
                 right=np.random.randn(3, 2),
                 middle=np.random.randn(2, 2))
    wb = d + lr

    # Check basic operations for all types: interaction with scalars.
    for m in [a, d, lr, wb]:
        yield allclose, 5 * m, 5 * dense(m)
        yield allclose, m * 5, dense(m) * 5
        yield allclose, 5 + m, 5 + dense(m)
        yield allclose, m + 5, dense(m) + 5
        yield allclose, 5 - m, 5 - dense(m)
        yield allclose, m - 5, dense(m) - 5
        yield allclose, m.__div__(5.0), dense(m) / 5.0
        yield allclose, m.__truediv__(5.0), dense(m) / 5.0

    # Check basic operations for all types: interaction with others.
    for m1 in [a, d, lr, wb]:
        for m2 in [a, d, lr, wb]:
            yield allclose, m1 * m2, dense(m1) * dense(m2)
            yield allclose, m1 + m2, dense(m1) + dense(m2)
            yield allclose, m1 - m2, dense(m1) - dense(m2)

    # Finally, check shapes.
    yield eq, B.shape(a), (4, 3)
    yield eq, B.shape(d), (4, 3)
    yield eq, B.shape(lr), (4, 3)
    yield eq, B.shape(wb), (4, 3)
