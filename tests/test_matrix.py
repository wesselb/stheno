# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from lab import B
from itertools import product

from stheno.matrix import Dense, Diagonal, LowRank, dense, \
    Woodbury, Constant, matrix, Zero, One, Constant
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok, allclose, assert_allclose


def test_equality():
    # Test `Dense.`
    a = Dense(np.random.randn(4, 2))
    yield assert_allclose, a == a, dense(a) == dense(a)

    # Test `Diagonal`.
    d = Diagonal(np.random.randn(4))
    yield assert_allclose, d == d, B.diag(d) == B.diag(d)

    # Test `LowRank`.
    lr = LowRank(left=np.random.randn(4, 2),
                 middle=np.random.randn(2, 2))
    yield assert_allclose, lr == lr, (lr.l == lr.l, lr.m == lr.m, lr.r == lr.r)

    # Test `Woodbury`.
    yield assert_allclose, (lr + d) == (lr + d), (lr == lr, d == d)

    # Test `Constant`.
    c1 = Constant.from_(1, a)
    c1_2 = Constant(1, 4, 3)
    c2 = Constant.from_(2, a)
    yield eq, c1, c1
    yield neq, c1, c1_2
    yield neq, c1, c2

    # Test `One`.
    one1 = One(np.float64, 4, 2)
    one2 = One(np.float64, 4, 3)
    yield eq, one1, one1
    yield neq, one1, one2

    # Test `Zero`.
    zero1 = Zero(np.float64, 4, 2)
    zero2 = Zero(np.float64, 4, 3)
    yield eq, zero1, zero1
    yield neq, zero1, zero2


def test_constant_zero_one():
    a = np.random.randn(4, 2)
    b = Dense(a)

    yield assert_allclose, Zero.from_(a), np.zeros((4, 2))
    yield assert_allclose, Zero.from_(b), np.zeros((4, 2))

    yield assert_allclose, One.from_(a), np.ones((4, 2))
    yield assert_allclose, One.from_(b), np.ones((4, 2))

    yield assert_allclose, Constant.from_(2, a), 2 * np.ones((4, 2))
    yield assert_allclose, Constant.from_(2, b), 2 * np.ones((4, 2))


def test_shorthands():
    a = Dense(np.random.randn(4, 4))
    yield assert_allclose, a.T, B.transpose(a)
    yield assert_allclose, a.__matmul__(a), B.matmul(a, a)


def test_dense_methods():
    a = Dense(np.random.randn(10, 5))
    yield assert_allclose, dense(a.T), dense(a).T
    yield assert_allclose, dense(-a), -dense(a)
    yield assert_allclose, dense(a[5]), dense(a)[5]


def test_eye_form():
    a = Dense(np.random.randn(5, 10))
    yield assert_allclose, dense(B.eye(a)), np.eye(5, 10)
    yield assert_allclose, dense(B.eye(a.T)), np.eye(10, 5)


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
          dense(Diagonal(np.array([1, 2]))), \
          np.array([[1, 0],
                    [0, 2]])
    yield assert_allclose, \
          dense(Diagonal(np.array([1, 2]), 3)), \
          np.array([[1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0]])
    yield assert_allclose, \
          dense(Diagonal(np.array([1, 2]), 1)), \
          np.array([[1]])
    yield assert_allclose, \
          dense(Diagonal(np.array([1, 2]), 3, 3)), \
          np.array([[1, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0]])
    yield assert_allclose, \
          dense(Diagonal(np.array([1, 2]), 2, 3)), \
          np.array([[1, 0, 0],
                    [0, 2, 0]])
    yield assert_allclose, \
          dense(Diagonal(np.array([1, 2]), 3, 2)), \
          np.array([[1, 0],
                    [0, 2],
                    [0, 0]])
    yield assert_allclose, \
          dense(Diagonal(np.array([1, 2]), 1, 3)), \
          np.array([[1, 0, 0]])
    yield assert_allclose, \
          dense(Diagonal(np.array([1, 2]), 3, 1)), \
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
    diag = Diagonal(np.array([1, 2, 3, 4]), 5, 10)
    wb = Woodbury(diag=diag, lr=lr)
    yield assert_allclose, dense(wb), dense(diag) + dense(lr)


def test_block_matrix():
    dt = np.float64

    # Check correctness.
    rows = [[np.random.randn(4, 3), np.random.randn(4, 5)],
            [np.random.randn(6, 3), np.random.randn(6, 5)]]
    yield assert_allclose, B.block_matrix(*rows), B.concat2d(rows)

    # Check that grid is checked correctly.
    yield eq, type(B.block_matrix([Zero(dt, 3, 7), Zero(dt, 3, 4)],
                                  [Zero(dt, 4, 5), Zero(dt, 4, 6)])), Dense
    yield raises, ValueError, \
          lambda: B.block_matrix([Zero(dt, 5, 5), Zero(dt, 3, 6)],
                                 [Zero(dt, 2, 5), Zero(dt, 4, 6)])

    # Test zeros.
    res = B.block_matrix([Zero(dt, 3, 5), Zero(dt, 3, 6)],
                         [Zero(dt, 4, 5), Zero(dt, 4, 6)])
    yield eq, type(res), Zero
    yield assert_allclose, res, Zero(dt, 7, 11)

    # Test ones.
    res = B.block_matrix([One(dt, 3, 5), One(dt, 3, 6)],
                         [One(dt, 4, 5), One(dt, 4, 6)])
    yield eq, type(res), One
    yield assert_allclose, res, One(dt, 7, 11)

    # Test diagonal.
    res = B.block_matrix([Diagonal(np.array([1, 2])), Zero(dt, 2, 3)],
                         [Zero(dt, 3, 2), Diagonal(np.array([3, 4, 5]))])
    yield eq, type(res), Diagonal
    yield assert_allclose, res, Diagonal(np.array([1, 2, 3, 4, 5]))
    # Check that all blocks on the diagonal must be diagonal or zero.
    yield eq, type(B.block_matrix([Diagonal(np.array([1, 2])), Zero(dt, 2, 3)],
                                  [Zero(dt, 3, 2), One(dt, 3)])), Dense
    yield eq, type(B.block_matrix([Diagonal(np.array([1, 2])), Zero(dt, 2, 3)],
                                  [Zero(dt, 3, 2), Zero(dt, 3)])), Diagonal
    # Check that all blocks on the diagonal must be square.
    yield eq, type(B.block_matrix([Diagonal(np.array([1, 2])), Zero(dt, 2, 4)],
                                  [Zero(dt, 3, 2), Zero(dt, 3, 4)])), Dense
    # Check that all other blocks must be zero.
    yield eq, type(B.block_matrix([Diagonal(np.array([1, 2])), One(dt, 2, 3)],
                                  [Zero(dt, 3, 2),
                                   Diagonal(np.array([3, 4, 5]))])), Dense


def test_dtype():
    # Test `Dense`.
    yield eq, B.dtype(Dense(np.array([[1]]))), np.int64
    yield eq, B.dtype(Dense(np.array([[1.0]]))), np.float64

    # Test `Diagonal`.
    diag_int = Diagonal(np.array([1]))
    diag_float = Diagonal(np.array([1.0]))
    yield eq, B.dtype(diag_int), np.int64
    yield eq, B.dtype(diag_float), np.float64

    # Test `LowRank`.
    lr_int = LowRank(left=np.array([[1]]),
                     right=np.array([[2]]),
                     middle=np.array([[3]]))
    lr_float = LowRank(left=np.array([[1.0]]),
                       right=np.array([[2.0]]),
                       middle=np.array([[3.0]]))
    yield eq, B.dtype(lr_int), np.int64
    yield eq, B.dtype(lr_float), np.float64

    # Test `Constant`.
    yield eq, B.dtype(Constant(1, rows=1)), int
    yield eq, B.dtype(Constant(1.0, rows=1)), float

    # Test `Woodbury`.
    yield eq, B.dtype(Woodbury(diag_int, lr_int)), np.int64
    yield eq, B.dtype(Woodbury(diag_float, lr_float)), np.float64


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
    yield assert_allclose, B.diag(Diagonal(np.array([1, 2, 3]))), [1, 2, 3]
    yield assert_allclose, B.diag(Diagonal(np.array([1, 2, 3]), 2)), [1, 2]
    yield assert_allclose, \
          B.diag(Diagonal(np.array([1, 2, 3]), 4)), [1, 2, 3, 0]

    # Test `LowRank`.
    b = np.random.randn(10, 3)
    yield assert_allclose, B.diag(LowRank(left=a, right=a)), np.diag(a.dot(a.T))
    yield assert_allclose, B.diag(LowRank(left=a, right=b)), np.diag(a.dot(b.T))
    yield assert_allclose, B.diag(LowRank(left=b, right=b)), np.diag(b.dot(b.T))

    # Test `Constant`.
    yield assert_allclose, B.diag(Constant(1, rows=3, cols=5)), np.ones(3)

    # Test `Woodbury`.
    yield assert_allclose, \
          B.diag(Woodbury(Diagonal(np.array([1, 2, 3]), rows=5, cols=10),
                          LowRank(left=a, right=b))), \
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

    # Test `LowRank`.
    a = np.random.randn(2, 2)
    lr = LowRank(left=np.random.randn(5, 2), middle=a.dot(a.T))
    chol = dense(B.cholesky(lr))
    #   The Cholesky here is not technically the Cholesky decomposition. Hence
    #   we test this slightly differently.
    yield assert_allclose, chol.dot(chol.T), lr


def test_matmul():
    diag_square = Diagonal(np.array([1, 2]), 3)
    diag_tall = Diagonal(np.array([3, 4]), 5, 3)
    diag_wide = Diagonal(np.array([5, 6]), 2, 3)

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
    a = np.random.randn(3, 3)
    a = Dense(a.dot(a.T))
    yield assert_allclose, B.matmul(a, B.inverse(a)), np.eye(3)
    yield assert_allclose, B.matmul(B.inverse(a), a), np.eye(3)
    yield assert_allclose, B.logdet(a), np.log(np.linalg.det(dense(a)))

    # Test `Diagonal`.
    d = Diagonal(np.array([1, 2, 3]))
    yield assert_allclose, B.matmul(d, B.inverse(d)), np.eye(3)
    yield assert_allclose, B.matmul(B.inverse(d), d), np.eye(3)
    yield assert_allclose, B.logdet(d), np.log(np.linalg.det(dense(d)))
    yield eq, \
          B.shape(B.inverse(Diagonal(np.array([1, 2]), rows=2, cols=4))), (4, 2)

    # Test `Woodbury`.
    a = np.random.randn(3, 2)
    b = np.random.randn(2, 2) + 1e-2 * np.eye(2)
    wb = d + LowRank(left=a, middle=b.dot(b.T))
    for _ in range(4):
        yield assert_allclose, B.matmul(wb, B.inverse(wb)), np.eye(3)
        yield assert_allclose, B.matmul(B.inverse(wb), wb), np.eye(3)
        yield assert_allclose, B.logdet(wb), np.log(np.linalg.det(dense(wb)))
        wb = B.inverse(wb)

    # Test `LowRank`.
    yield raises, RuntimeError, lambda: B.inverse(wb.lr)
    yield raises, RuntimeError, lambda: B.logdet(wb.lr)


def test_lr_diff():
    # First, test correctness.
    a = np.random.randn(3, 2)
    b = np.random.randn(2, 2)
    lr1 = LowRank(left=a, right=np.random.randn(3, 2), middle=b.dot(b.T))
    a = np.random.randn(3, 2)
    b = np.random.randn(2, 2)
    lr2 = LowRank(left=a, right=np.random.randn(3, 2), middle=b.dot(b.T))

    yield assert_allclose, B.lr_diff(lr1, lr1), B.zeros((3, 3))
    yield assert_allclose, B.lr_diff(lr1 + lr2, lr1), lr2
    yield assert_allclose, B.lr_diff(lr1 + lr2, lr2), lr1
    yield assert_allclose, B.lr_diff(lr1 + lr1 + lr2, lr1), lr1 + lr2
    yield assert_allclose, B.lr_diff(lr1 + lr1 + lr2, lr2), lr1 + lr1
    yield assert_allclose, B.lr_diff(lr1 + lr1 + lr2, lr1 + lr1), lr2
    yield assert_allclose, B.lr_diff(lr1 + lr1 + lr2, lr1 + lr2), lr1
    yield assert_allclose, \
          B.lr_diff(lr1 + lr1 + lr2, lr1 + lr1 + lr2), \
          B.zeros((3, 3))

    # Second, test positive definiteness.
    lr1 = LowRank(left=lr1.left, middle=lr1.middle)
    lr2 = LowRank(left=lr2.left, middle=lr2.middle)

    yield B.cholesky, B.lr_diff(lr1, 0.999 * lr1)
    yield B.cholesky, B.lr_diff(lr1 + lr2, lr1)
    yield B.cholesky, B.lr_diff(lr1 + lr2, lr2)
    yield B.cholesky, B.lr_diff(lr1 + lr1 + lr2, lr1)
    yield B.cholesky, B.lr_diff(lr1 + lr1 + lr2, lr1 + lr1)
    yield B.cholesky, B.lr_diff(lr1 + lr1 + lr2, lr1 + lr2)
    yield B.cholesky, B.lr_diff(lr1 + lr1 + lr2, lr1 + lr1 + 0.999 * lr2)


def test_root():
    # Test `Dense`.
    a = np.random.randn(5, 5)
    a = Dense(a.dot(a.T))
    yield assert_allclose, a, B.matmul(B.root(a), B.root(a))

    # Test `Diagonal`.
    d = Diagonal(np.array([1, 2, 3, 4, 5]))
    yield assert_allclose, d, B.matmul(B.root(d), B.root(d))


def test_sample():
    a = np.random.randn(3, 3)
    a = Dense(a.dot(a.T))
    b = np.random.randn(2, 2)
    wb = Diagonal(B.diag(a)) + \
         LowRank(left=np.random.randn(3, 2), middle=b.dot(b.T))

    # Test `Dense` and `Woodbury`.
    num_samps = 500000
    for cov in [a, wb]:
        samps = B.sample(cov, num_samps)
        cov_emp = B.matmul(samps, samps, tr_b=True) / num_samps
        yield le, np.mean(np.abs(dense(cov_emp) - dense(cov))), 5e-2


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
    a, b = np.random.randn(4, 4), np.random.randn(4, 4)
    a, b = Dense(a.dot(a.T)), Dense(b.dot(b.T))
    d, e = Diagonal(B.diag(a)), Diagonal(B.diag(b))
    c = np.random.randn(3, 3)
    lr = LowRank(left=np.random.randn(4, 3), middle=c.dot(c.T))

    yield assert_allclose, B.ratio(a, b), \
          np.trace(np.linalg.solve(dense(b), dense(a)))
    yield assert_allclose, B.ratio(lr, b), \
          np.trace(np.linalg.solve(dense(b), dense(lr)))
    yield assert_allclose, B.ratio(d, e), \
          np.trace(np.linalg.solve(dense(e), dense(d)))


def test_transposition():
    def compare(a):
        assert_allclose(B.transpose(a), dense(a).T)

    d = Diagonal(np.array([1, 2, 3]), rows=5, cols=10)
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
    zero = Zero.from_(a)
    one = One.from_(a)
    constant = Constant.from_(2.0, a)
    wb = d + lr

    # Aggregate all matrices.
    candidates = [a, d, lr, wb, constant, one, zero, 2, 1, 0, 2.0, 1.0, 0.0]

    # Check division.
    yield assert_allclose, a.__div__(5.0), dense(a) / 5.0
    yield assert_allclose, a.__truediv__(5.0), dense(a) / 5.0
    yield assert_allclose, B.divide(a, 5.0), dense(a) / 5.0

    # Check shapes.
    for m in candidates:
        yield eq, B.shape(a), (4, 3)

    # Check interactions.
    for m1, m2 in product(candidates, candidates):
        yield assert_allclose, m1 * m2, dense(m1) * dense(m2)
        yield assert_allclose, m1 + m2, dense(m1) + dense(m2)
        yield assert_allclose, m1 - m2, dense(m1) - dense(m2)
