# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from lab import B

from stheno.matrix import Dense, Diagonal, LowRank, dense, \
    Woodbury, Constant, matrix
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint, allclose, \
    assert_allclose


def test_dense_methods():
    a = Dense(np.random.randn(10, 5))
    yield assert_allclose, dense(a.T), dense(a).T
    yield assert_allclose, dense(-a), -dense(a)
    yield assert_allclose, dense(a[5]), dense(a)[5]


def test_eye_form():
    a = Dense(np.random.randn(5, 10))
    yield assert_allclose, dense(B.eye_from(a)), np.eye(5, 10)
    yield assert_allclose, dense(B.eye_from(a.T)), np.eye(10, 5)


def test_matrix():
    a = np.random.randn(5, 3)
    yield eq, type(matrix(a)), Dense
    yield assert_allclose, matrix(a).mat, a
    yield eq, type(matrix(matrix(a))), Dense
    yield assert_allclose, matrix(matrix(a)).mat, a


def test_dense():
    a = np.random.randn(5, 3)
    yield assert_allclose, dense(Dense(a)), a
    yield assert_allclose, dense(a), a

    # Extensively test Diagonal.
    yield assert_allclose, \
          dense(Diagonal([1, 2])), \
          np.array([[1, 0],
                    [0, 2]])
    yield assert_allclose, \
          dense(Diagonal([1, 2], 3)), \
          np.array([[1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0]])
    yield assert_allclose, \
          dense(Diagonal([1, 2], 1)), \
          np.array([[1]])
    yield assert_allclose, \
          dense(Diagonal([1, 2], 3, 3)), \
          np.array([[1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0]])
    yield assert_allclose, \
          dense(Diagonal([1, 2], 2, 3)), \
          np.array([[1, 0, 0],
                    [0, 2, 0]])
    yield assert_allclose, \
          dense(Diagonal([1, 2], 3, 2)), \
          np.array([[1, 0],
                    [0, 2],
                    [0, 0]])
    yield assert_allclose, \
          dense(Diagonal([1, 2], 1, 3)), \
          np.array([[1, 0, 0]])
    yield assert_allclose, \
          dense(Diagonal([1, 2], 3, 1)), \
          np.array([[1],
                    [0],
                    [0]])

    # Test low-rank matrices.
    left = np.random.randn(5, 3)
    right = np.random.randn(10, 3)
    middle = np.random.randn(3, 3)
    lr = LowRank(left=left, right=right, middle=middle)
    yield assert_allclose, dense(lr), left.dot(middle).dot(right.T)

    # Test Woodbury matrices.
    diag = Diagonal([1, 2, 3, 4], 5, 10)
    wb = Woodbury(lr, diag)
    yield assert_allclose, dense(wb), dense(diag) + dense(lr)


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
    yield eq, B.dtype(Constant(1, rows=1)), int
    yield eq, B.dtype(Constant(1.0, rows=1)), float

    # Test `Woodbury`.
    yield eq, B.dtype(Woodbury(lr_int, diag_int)), int
    yield eq, B.dtype(Woodbury(lr_float, diag_float)), float


def test_sum():
    a = Dense(np.random.randn(10, 20))
    yield assert_allclose, B.sum(a, axis=0), np.sum(dense(a), axis=0)

    for x in [Diagonal(np.array([1, 2, 3]), rows=3, cols=5),
              LowRank(left=np.random.randn(5, 3),
                      right=np.random.randn(10, 3),
                      middle=np.random.randn(3, 3))]:
        yield assert_allclose, B.sum(x), np.sum(dense(x))
        yield assert_allclose, B.sum(x, axis=0), np.sum(dense(x), axis=0)
        yield assert_allclose, B.sum(x, axis=1), np.sum(dense(x), axis=1)
        yield assert_allclose, B.sum(x, axis=(0, 1)), np.sum(dense(x),
                                                             axis=(0, 1))


def test_trace():
    a = np.random.randn(10, 10)
    yield assert_allclose, B.trace(Dense(a)), np.trace(a)


def test_diag():
    # Test `Dense`.
    a = np.random.randn(5, 3)
    yield assert_allclose, B.diag(Dense(a)), np.diag(a)

    # Test `Diagonal`.
    yield assert_allclose, B.diag(Diagonal([1, 2, 3])), [1, 2, 3]
    yield assert_allclose, B.diag(Diagonal([1, 2, 3], 2)), [1, 2]
    yield assert_allclose, B.diag(Diagonal([1, 2, 3], 4)), [1, 2, 3, 0]

    # Test `LowRank`.
    b = np.random.randn(10, 3)
    yield assert_allclose, B.diag(LowRank(left=a, right=a)), np.diag(a.dot(a.T))
    yield assert_allclose, B.diag(LowRank(left=a, right=b)), np.diag(a.dot(b.T))
    yield assert_allclose, B.diag(LowRank(left=b, right=b)), np.diag(b.dot(b.T))

    # Test `Constant`.
    yield assert_allclose, B.diag(Constant(1, rows=3, cols=5)), np.ones(3)

    # Test `Woodbury`.
    yield assert_allclose, \
          B.diag(Woodbury(LowRank(left=a, right=b),
                          Diagonal([1, 2, 3], rows=5, cols=10))), \
          np.diag(a.dot(b.T) + np.concatenate((np.diag([1, 2, 3, 0, 0]),
                                               np.zeros((5, 5))), axis=1))

    # The method `B.diag(diag, rows, cols)` is tested in the tests for
    # `Diagonal` in `test_dense`.


def test_diag_len():
    yield eq, B.diag_len(np.ones((5, 5))), 5
    yield eq, B.diag_len(np.ones((10, 5))), 5
    yield eq, B.diag_len(np.ones((5, 10))), 5


def test_cholesky():
    a = np.random.randn(5, 5)
    a = a.T.dot(a)

    # Test `Dense`.
    yield assert_allclose, np.linalg.cholesky(a), B.cholesky(a)

    # Test `Diagonal`.
    d = Diagonal(np.diag(a))
    yield assert_allclose, np.linalg.cholesky(dense(d)), B.cholesky(d)
    yield raises, RuntimeError, \
          lambda: B.cholesky(Diagonal([1], rows=2, cols=1))


def test_matmul():
    diag_square = Diagonal([1, 2], 3)
    diag_tall = Diagonal([3, 4], 5, 3)
    diag_wide = Diagonal([5, 6], 2, 3)

    dense_square = Dense(np.random.randn(3, 3))
    dense_tall = Dense(np.random.randn(5, 3))
    dense_wide = Dense(np.random.randn(2, 3))

    lr = LowRank(left=np.random.randn(5, 2),
                 right=np.random.randn(3, 2),
                 middle=np.random.randn(2, 2))

    def compare(a, b):
        return allclose(B.matmul(a, b), B.matmul(dense(a), dense(b)))

    # Test `Dense`.
    yield ok, compare(dense_wide, dense_tall.T), 'dense w x dense t'

    # Test `LowRank`.
    yield ok, compare(lr, dense_tall.T), 'lr x dense t'
    yield ok, compare(dense_wide, lr.T), 'dense w x lr'
    yield ok, compare(lr, diag_tall.T), 'lr x diag t'
    yield ok, compare(diag_wide, lr.T), 'diag w x lr'
    yield ok, compare(lr, lr.T), 'lr x lr'
    yield ok, compare(lr.T, lr), 'lr x lr (2)'

    # Test `Diagonal`.
    #   Test multiplication between diagonal matrices.
    yield ok, compare(diag_square, diag_square.T), 'diag s x diag s'
    yield ok, compare(diag_tall, diag_square.T), 'diag t x diag s'
    yield ok, compare(diag_wide, diag_square.T), 'diag w x diag s'
    yield ok, compare(diag_square, diag_tall.T), 'diag s x diag t'
    yield ok, compare(diag_tall, diag_tall.T), 'diag t x diag t'
    yield ok, compare(diag_wide, diag_tall.T), 'diag w x diag t'
    yield ok, compare(diag_square, diag_wide.T), 'diag s x diag w'
    yield ok, compare(diag_tall, diag_wide.T), 'diag t x diag w'
    yield ok, compare(diag_wide, diag_wide.T), 'diag w x diag w'

    #   Test multiplication between diagonal and dense matrices.
    yield ok, compare(diag_square, dense_square.T), 'diag s x dense s'
    yield ok, compare(diag_square, dense_tall.T), 'diag s x dense t'
    yield ok, compare(diag_square, dense_wide.T), 'diag s x dense w'
    yield ok, compare(diag_tall, dense_square.T), 'diag t x dense s'
    yield ok, compare(diag_tall, dense_tall.T), 'diag t x dense t'
    yield ok, compare(diag_tall, dense_wide.T), 'diag t x dense w'
    yield ok, compare(diag_wide, dense_square.T), 'diag w x dense s'
    yield ok, compare(diag_wide, dense_tall.T), 'diag w x dense t'
    yield ok, compare(diag_wide, dense_wide.T), 'diag w x dense w'

    yield ok, compare(dense_square, diag_square.T), 'dense s x diag s'
    yield ok, compare(dense_square, diag_tall.T), 'dense s x diag t'
    yield ok, compare(dense_square, diag_wide.T), 'dense s x diag w'
    yield ok, compare(dense_tall, diag_square.T), 'dense t x diag s'
    yield ok, compare(dense_tall, diag_tall.T), 'dense t x diag t'
    yield ok, compare(dense_tall, diag_wide.T), 'dense t x diag w'
    yield ok, compare(dense_wide, diag_square.T), 'dense w x diag s'
    yield ok, compare(dense_wide, diag_tall.T), 'dense w x diag t'
    yield ok, compare(dense_wide, diag_wide.T), 'dense w x diag w'

    # Test `B.matmul` with three matrices simultaneously.
    yield assert_allclose, \
          B.matmul(dense_tall, dense_square, dense_wide, tr_c=True), \
          dense(dense_tall).dot(dense(dense_square)).dot(dense(dense_wide).T)

    # Test `Woodbury`.
    wb = lr + dense_tall
    yield ok, compare(wb, dense_square.T), 'wb x dense s'
    yield ok, compare(dense_square, wb.T), 'dense s x wb'
    yield ok, compare(wb, wb.T), 'wb x wb'
    yield ok, compare(wb.T, wb), 'wb x wb (2)'


def test_inverse_and_logdet():
    # Test `Dense`.
    a = np.random.randn(5, 5)
    a = Dense(a.dot(a.T))
    yield assert_allclose, B.matmul(a, B.inverse(a)), np.eye(5)
    yield assert_allclose, B.matmul(B.inverse(a), a), np.eye(5)
    yield assert_allclose, B.logdet(a), np.log(np.linalg.det(dense(a)))

    # Test `Diagonal`.
    d = Diagonal([1, 2, 3, 4, 5])
    yield assert_allclose, B.matmul(d, B.inverse(d)), np.eye(5)
    yield assert_allclose, B.matmul(B.inverse(d), d), np.eye(5)
    yield assert_allclose, B.logdet(d), np.log(np.linalg.det(dense(d)))
    yield eq, B.shape(B.inverse(Diagonal([1, 2], rows=2, cols=4))), (4, 2)

    # Test `Woodbury`.
    a = np.random.randn(5, 3)
    b = np.random.randn(3, 3)
    wb = d + LowRank(left=a, right=a, middle=b.dot(b.T))
    yield assert_allclose, B.matmul(wb, B.inverse(wb)), np.eye(5)
    yield assert_allclose, B.matmul(B.inverse(wb), wb), np.eye(5)
    yield assert_allclose, B.logdet(wb), np.log(np.linalg.det(dense(wb)))

    # Test `LowRank`.
    yield raises, RuntimeError, lambda: B.inverse(wb.lr)
    yield raises, RuntimeError, lambda: B.logdet(wb.lr)


def test_root():
    # Test `Dense`.
    a = np.random.randn(5, 5)
    a = Dense(a.dot(a.T))
    yield assert_allclose, a, B.matmul(B.root(a), B.root(a))

    # Test `Diagonal`.
    d = Diagonal(np.array([1, 2, 3, 4, 5]))
    yield assert_allclose, d, B.matmul(B.root(d), B.root(d))


def test_schur():
    # Test `Dense`.
    a = np.random.randn(5, 10)
    b = np.random.randn(3, 5)
    c = np.random.randn(3, 3)
    d = np.random.randn(3, 10)
    c = c.dot(c.T)

    yield ok, allclose(B.schur(a, b, c, d),
                       a - np.linalg.solve(c.T, b).T.dot(d)), 'n n n n'

    # Test `Woodbury`.
    #   The inverse of the Woodbury matrix already properly tests the method for
    #   Woodbury matrices.
    c = np.random.randn(2, 2)
    c = Diagonal(np.array([1, 2, 3])) + \
        LowRank(left=np.random.randn(3, 2), middle=c.dot(c.T))
    yield ok, allclose(B.schur(a, b, c, d),
                       a - np.linalg.solve(dense(c).T, b).T.dot(d)), 'n n w n'

    #   Test all combinations of `Woodbury`, `LowRank`, and `Diagonal`.
    a = np.random.randn(2, 2)
    a = Diagonal(np.array([4, 5, 6, 7, 8]), rows=5, cols=10) + \
        LowRank(left=np.random.randn(5, 2),
                right=np.random.randn(10, 2),
                middle=a.dot(a.T))

    b = np.random.randn(2, 2)
    b = Diagonal(np.array([9, 10, 11]), rows=3, cols=5) + \
        LowRank(left=c.lr.left,
                right=a.lr.left,
                middle=b.dot(b.T))

    d = np.random.randn(2, 2)
    d = Diagonal(np.array([12, 13, 14]), rows=3, cols=10) + \
        LowRank(left=c.lr.right,
                right=a.lr.right,
                middle=d.dot(d.T))

    #     Loop over all combinations. Some of them should be efficient and
    #     representation preserving; all of them should be correct.
    for ai in [a, a.lr, a.diag]:
        for bi in [b, b.lr, b.diag]:
            for ci in [c, c.diag]:
                for di in [d, d.lr, d.diag]:
                    yield ok, allclose(
                        B.schur(ai, bi, ci, di),
                        dense(ai) -
                        np.linalg.solve(dense(ci).T,
                                        dense(bi)).T.dot(dense(di))
                    ), '{} {} {} {}'.format(ai, bi, ci, di)


def test_qf():
    # Generate some test inputs.
    b, c = np.random.randn(5, 3), np.random.randn(5, 3)

    # Generate some matrices to test.
    a = np.random.randn(5, 5)
    a = Dense(a.dot(a.T))
    d = Diagonal(B.diag(a))
    e = np.random.randn(2, 2)
    wb = d + LowRank(left=np.random.randn(5, 2),
                     middle=e.dot(e.T))

    for x in [a, d, wb]:
        yield assert_allclose, B.qf(x, b), \
              np.linalg.solve(dense(x), b).T.dot(b)
        yield assert_allclose, B.qf(x, b, c), \
              np.linalg.solve(dense(x), b).T.dot(c)
        yield assert_allclose, B.qf_diag(x, b), \
              np.diag(np.linalg.solve(dense(x), b).T.dot(b))
        yield assert_allclose, B.qf_diag(x, b, c), \
              np.diag(np.linalg.solve(dense(x), b).T.dot(c))

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

    yield assert_allclose, B.ratio(a, b), \
          np.trace(np.linalg.solve(dense(b), dense(a)))
    yield assert_allclose, B.ratio(lr, b), \
          np.trace(np.linalg.solve(dense(b), dense(lr)))
    yield assert_allclose, B.ratio(d, e), \
          np.trace(np.linalg.solve(dense(e), dense(d)))


def test_transposition():
    def compare(a):
        assert_allclose(B.transpose(a), dense(a).T)

    d = Diagonal([1, 2, 3], rows=5, cols=10)
    lr = LowRank(left=np.random.randn(5, 3), right=np.random.randn(10, 3))

    yield compare, Dense(np.random.randn(5, 5))
    yield compare, d
    yield compare, lr
    yield compare, d + lr
    yield compare, Constant(5, rows=4, cols=2)


def test_arithmetic_and_shapes():
    a = Dense(np.random.randn(4, 3))
    d = Diagonal(np.array([1.0, 2.0, 3.0]), rows=4, cols=3)
    lr = LowRank(left=np.random.randn(4, 2),
                 right=np.random.randn(3, 2),
                 middle=np.random.randn(2, 2))
    wb = d + lr

    # Check basic operations for all types: interaction with scalars.
    for m in [a, d, lr, wb]:
        yield assert_allclose, 5 * m, 5 * dense(m)
        yield assert_allclose, m * 5, dense(m) * 5
        yield assert_allclose, 5 + m, 5 + dense(m)
        yield assert_allclose, m + 5, dense(m) + 5
        yield assert_allclose, 5 - m, 5 - dense(m)
        yield assert_allclose, m - 5, dense(m) - 5
        yield assert_allclose, m.__div__(5.0), dense(m) / 5.0
        yield assert_allclose, m.__truediv__(5.0), dense(m) / 5.0

    # Check basic operations for all types: interaction with others.
    for m1 in [a, d, lr, wb]:
        for m2 in [a, d, lr, wb]:
            yield assert_allclose, m1 * m2, dense(m1) * dense(m2)
            yield assert_allclose, m1 + m2, dense(m1) + dense(m2)
            yield assert_allclose, m1 - m2, dense(m1) - dense(m2)

    # Finally, check shapes.
    yield eq, B.shape(a), (4, 3)
    yield eq, B.shape(d), (4, 3)
    yield eq, B.shape(lr), (4, 3)
    yield eq, B.shape(wb), (4, 3)
