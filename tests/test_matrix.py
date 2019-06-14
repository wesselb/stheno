# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import product

import numpy as np
import pytest
from lab import B
from stheno.matrix import (
    Dense,
    Diagonal,
    LowRank,
    Woodbury,
    matrix,
    Zero,
    One,
    Constant
)

from .util import allclose, to_np


def test_equality():
    # Test `Dense.`
    a = Dense(np.random.randn(4, 2))
    allclose(a == a, to_np(a) == to_np(a))

    # Test `Diagonal`.
    d = Diagonal(np.random.randn(4))
    allclose(d == d, B.diag(d) == B.diag(d))

    # Test `LowRank`.
    lr = LowRank(left=np.random.randn(4, 2), middle=np.random.randn(2, 2))
    allclose(lr == lr, (lr.l == lr.l, lr.m == lr.m, lr.r == lr.r))

    # Test `Woodbury`.
    allclose((lr + d) == (lr + d), (lr == lr, d == d))

    # Test `Constant`.
    c1 = Constant.from_(1, a)
    c1_2 = Constant(1, 4, 3)
    c2 = Constant.from_(2, a)
    assert c1 == c1
    assert c1 != c1_2
    assert c1 != c2

    # Test `One`.
    one1 = One(np.float64, 4, 2)
    one2 = One(np.float64, 4, 3)
    assert one1 == one1
    assert one1 != one2

    # Test `Zero`.
    zero1 = Zero(np.float64, 4, 2)
    zero2 = Zero(np.float64, 4, 3)
    assert zero1 == zero1
    assert zero1 != zero2


def test_constant_zero_one():
    a = np.random.randn(4, 2)
    b = Dense(a)

    allclose(Zero.from_(a), np.zeros((4, 2)))
    allclose(Zero.from_(b), np.zeros((4, 2)))

    allclose(One.from_(a), np.ones((4, 2)))
    allclose(One.from_(b), np.ones((4, 2)))

    allclose(Constant.from_(2, a), 2 * np.ones((4, 2)))
    allclose(Constant.from_(2, b), 2 * np.ones((4, 2)))


def test_shorthands():
    a = Dense(np.random.randn(4, 4))
    allclose(a.T, B.transpose(a))
    allclose(a.__matmul__(a), B.matmul(a, a))


def test_dense_methods():
    a = Dense(np.random.randn(10, 5))
    allclose(a.T, to_np(a).T)
    allclose(-a, -to_np(a))
    allclose(a[5], to_np(a)[5])


def test_eye():
    a = Dense(np.random.randn(5, 10))
    allclose(B.eye(a), np.eye(5, 10))
    allclose(B.eye(a.T), np.eye(10, 5))


def test_matrix():
    a = np.random.randn(5, 3)
    assert type(matrix(a)) == Dense
    allclose(matrix(a).mat, a)
    assert type(matrix(matrix(a))) == Dense
    allclose(matrix(matrix(a)).mat, a)


def test_dense():
    a = np.random.randn(5, 3)
    allclose(Dense(a), a)

    # Extensively test Diagonal.
    allclose(Diagonal(np.array([1, 2])), np.array([[1, 0],
                                                   [0, 2]]))
    allclose(Diagonal(np.array([1, 2]), 3), np.array([[1, 0, 0],
                                                      [0, 2, 0],
                                                      [0, 0, 0]]))
    allclose(Diagonal(np.array([1, 2]), 1), np.array([[1]]))
    allclose(Diagonal(np.array([1, 2]), 3, 3), np.array([[1, 0, 0],
                                                         [0, 2, 0],
                                                         [0, 0, 0]]))
    allclose(Diagonal(np.array([1, 2]), 2, 3), np.array([[1, 0, 0],
                                                         [0, 2, 0]]))
    allclose(Diagonal(np.array([1, 2]), 3, 2), np.array([[1, 0],
                                                         [0, 2],
                                                         [0, 0]]))
    allclose(Diagonal(np.array([1, 2]), 1, 3), np.array([[1, 0, 0]]))
    allclose(Diagonal(np.array([1, 2]), 3, 1), np.array([[1],
                                                         [0],
                                                         [0]]))

    # Test low-rank matrices.
    left = np.random.randn(5, 3)
    right = np.random.randn(10, 3)
    middle = np.random.randn(3, 3)
    lr = LowRank(left=left, right=right, middle=middle)
    allclose(lr, left.dot(middle).dot(right.T))

    # Test Woodbury matrices.
    diag = Diagonal(np.array([1, 2, 3, 4]), 5, 10)
    wb = Woodbury(diag=diag, lr=lr)
    allclose(wb, to_np(diag) + to_np(lr))


def test_block_matrix():
    dt = np.float64

    # Check correctness.
    rows = [[np.random.randn(4, 3), np.random.randn(4, 5)],
            [np.random.randn(6, 3), np.random.randn(6, 5)]]
    allclose(B.block_matrix(*rows), B.concat2d(*rows))

    # Check that grid is checked correctly.
    assert type(B.block_matrix([Zero(dt, 3, 7), Zero(dt, 3, 4)],
                               [Zero(dt, 4, 5), Zero(dt, 4, 6)])) == Dense
    with pytest.raises(ValueError):
        B.block_matrix([Zero(dt, 5, 5), Zero(dt, 3, 6)],
                       [Zero(dt, 2, 5), Zero(dt, 4, 6)])

    # Test zeros.
    res = B.block_matrix([Zero(dt, 3, 5), Zero(dt, 3, 6)],
                         [Zero(dt, 4, 5), Zero(dt, 4, 6)])
    assert type(res) == Zero
    allclose(res, Zero(dt, 7, 11))

    # Test ones.
    res = B.block_matrix([One(dt, 3, 5), One(dt, 3, 6)],
                         [One(dt, 4, 5), One(dt, 4, 6)])
    assert type(res) == One
    allclose(res, One(dt, 7, 11))

    # Test diagonal.
    res = B.block_matrix([Diagonal(np.array([1, 2])), Zero(dt, 2, 3)],
                         [Zero(dt, 3, 2), Diagonal(np.array([3, 4, 5]))])
    assert type(res) == Diagonal
    allclose(res, Diagonal(np.array([1, 2, 3, 4, 5])))
    # Check that all blocks on the diagonal must be diagonal or zero.
    assert type(B.block_matrix([Diagonal(np.array([1, 2])), Zero(dt, 2, 3)],
                               [Zero(dt, 3, 2), One(dt, 3)])) == Dense
    assert type(B.block_matrix([Diagonal(np.array([1, 2])), Zero(dt, 2, 3)],
                               [Zero(dt, 3, 2), Zero(dt, 3)])) == Diagonal
    # Check that all blocks on the diagonal must be square.
    assert type(B.block_matrix([Diagonal(np.array([1, 2])), Zero(dt, 2, 4)],
                               [Zero(dt, 3, 2), Zero(dt, 3, 4)])) == Dense
    # Check that all other blocks must be zero.
    assert type(B.block_matrix([Diagonal(np.array([1, 2])), One(dt, 2, 3)],
                               [Zero(dt, 3, 2),
                                Diagonal(np.array([3, 4, 5]))])) == Dense


def test_dtype():
    # Test `Dense`.
    assert B.dtype(Dense(np.array([[1]]))) == np.int64
    assert B.dtype(Dense(np.array([[1.0]]))) == np.float64

    # Test `Diagonal`.
    diag_int = Diagonal(np.array([1]))
    diag_float = Diagonal(np.array([1.0]))
    assert B.dtype(diag_int) == np.int64
    assert B.dtype(diag_float) == np.float64

    # Test `LowRank`.
    lr_int = LowRank(left=np.array([[1]]),
                     right=np.array([[2]]),
                     middle=np.array([[3]]))
    lr_float = LowRank(left=np.array([[1.0]]),
                       right=np.array([[2.0]]),
                       middle=np.array([[3.0]]))
    assert B.dtype(lr_int) == np.int64
    assert B.dtype(lr_float) == np.float64

    # Test `Constant`.
    assert B.dtype(Constant(1, rows=1)) == int
    assert B.dtype(Constant(1.0, rows=1)) == float

    # Test `Woodbury`.
    assert B.dtype(Woodbury(diag_int, lr_int)) == np.int64
    assert B.dtype(Woodbury(diag_float, lr_float)) == np.float64


def test_sum():
    a = Dense(np.random.randn(10, 20))
    allclose(B.sum(a, axis=0), np.sum(to_np(a), axis=0))

    for x in [Diagonal(np.array([1, 2, 3]), rows=3, cols=5),
              LowRank(left=np.random.randn(5, 3),
                      right=np.random.randn(10, 3),
                      middle=np.random.randn(3, 3))]:
        allclose(B.sum(x), np.sum(to_np(x)))
        allclose(B.sum(x, axis=0), np.sum(to_np(x), axis=0))
        allclose(B.sum(x, axis=1), np.sum(to_np(x), axis=1))
        allclose(B.sum(x, axis=(0, 1)), np.sum(to_np(x), axis=(0, 1)))


def test_trace():
    a = np.random.randn(10, 10)
    allclose(B.trace(Dense(a)), np.trace(a))


def test_diag():
    # Test `Dense`.
    a = np.random.randn(5, 3)
    allclose(B.diag(Dense(a)), np.diag(a))

    # Test `Diagonal`.
    allclose(B.diag(Diagonal(np.array([1, 2, 3]))), [1, 2, 3])
    allclose(B.diag(Diagonal(np.array([1, 2, 3]), 2)), [1, 2])
    allclose(B.diag(Diagonal(np.array([1, 2, 3]), 4)), [1, 2, 3, 0])

    # Test `LowRank`.
    b = np.random.randn(10, 3)
    allclose(B.diag(LowRank(left=a, right=a)), np.diag(a.dot(a.T)))
    allclose(B.diag(LowRank(left=a, right=b)), np.diag(a.dot(b.T)))
    allclose(B.diag(LowRank(left=b, right=b)), np.diag(b.dot(b.T)))

    # Test `Constant`.
    allclose(B.diag(Constant(1, rows=3, cols=5)), np.ones(3))

    # Test `Woodbury`.
    allclose(B.diag(Woodbury(Diagonal(np.array([1, 2, 3]), rows=5, cols=10),
                             LowRank(left=a, right=b))),
             np.diag(a.dot(b.T) + np.concatenate((np.diag([1, 2, 3, 0, 0]),
                                                  np.zeros((5, 5))), axis=1)))

    # The method `B.diag(diag, rows, cols)` is tested in the tests for
    # `Diagonal` in `test_dense`.


def test_diag_len():
    assert B.diag_len(np.ones((5, 5))) == 5
    assert B.diag_len(np.ones((10, 5))) == 5
    assert B.diag_len(np.ones((5, 10))) == 5


def test_cholesky():
    a = np.random.randn(5, 5)
    a = a.T.dot(a)

    # Test `Dense`.
    allclose(np.linalg.cholesky(a), B.cholesky(a))

    # Test `Diagonal`.
    d = Diagonal(np.diag(a))
    allclose(np.linalg.cholesky(to_np(d)), B.cholesky(d))

    # Test `LowRank`.
    a = np.random.randn(2, 2)
    lr = LowRank(left=np.random.randn(5, 2), middle=a.dot(a.T))
    chol = to_np(B.cholesky(lr))
    #   The Cholesky here is not technically the Cholesky decomposition. Hence
    #   we test this slightly differently.
    allclose(chol.dot(chol.T), lr)


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
        return np.allclose(to_np(B.matmul(a, b)),
                           B.matmul(to_np(a), to_np(b)))

    # Test `Dense`.
    assert compare(dense_wide, dense_tall.T), 'dense w x dense t'

    # Test `LowRank`.
    assert compare(lr, dense_tall.T), 'lr x dense t'
    assert compare(dense_wide, lr.T), 'dense w x lr'
    assert compare(lr, diag_tall.T), 'lr x diag t'
    assert compare(diag_wide, lr.T), 'diag w x lr'
    assert compare(lr, lr.T), 'lr x lr'
    assert compare(lr.T, lr), 'lr x lr (2)'

    # Test `Diagonal`.
    #   Test multiplication between diagonal matrices.
    assert compare(diag_square, diag_square.T), 'diag s x diag s'
    assert compare(diag_tall, diag_square.T), 'diag t x diag s'
    assert compare(diag_wide, diag_square.T), 'diag w x diag s'
    assert compare(diag_square, diag_tall.T), 'diag s x diag t'
    assert compare(diag_tall, diag_tall.T), 'diag t x diag t'
    assert compare(diag_wide, diag_tall.T), 'diag w x diag t'
    assert compare(diag_square, diag_wide.T), 'diag s x diag w'
    assert compare(diag_tall, diag_wide.T), 'diag t x diag w'
    assert compare(diag_wide, diag_wide.T), 'diag w x diag w'

    #   Test multiplication between diagonal and dense matrices.
    assert compare(diag_square, dense_square.T), 'diag s x dense s'
    assert compare(diag_square, dense_tall.T), 'diag s x dense t'
    assert compare(diag_square, dense_wide.T), 'diag s x dense w'
    assert compare(diag_tall, dense_square.T), 'diag t x dense s'
    assert compare(diag_tall, dense_tall.T), 'diag t x dense t'
    assert compare(diag_tall, dense_wide.T), 'diag t x dense w'
    assert compare(diag_wide, dense_square.T), 'diag w x dense s'
    assert compare(diag_wide, dense_tall.T), 'diag w x dense t'
    assert compare(diag_wide, dense_wide.T), 'diag w x dense w'

    assert compare(dense_square, diag_square.T), 'dense s x diag s'
    assert compare(dense_square, diag_tall.T), 'dense s x diag t'
    assert compare(dense_square, diag_wide.T), 'dense s x diag w'
    assert compare(dense_tall, diag_square.T), 'dense t x diag s'
    assert compare(dense_tall, diag_tall.T), 'dense t x diag t'
    assert compare(dense_tall, diag_wide.T), 'dense t x diag w'
    assert compare(dense_wide, diag_square.T), 'dense w x diag s'
    assert compare(dense_wide, diag_tall.T), 'dense w x diag t'
    assert compare(dense_wide, diag_wide.T), 'dense w x diag w'

    # Test `B.matmul` with three matrices simultaneously.
    allclose(B.matmul(dense_tall, dense_square, dense_wide, tr_c=True),
             (to_np(dense_tall)
              .dot(to_np(dense_square))
              .dot(to_np(dense_wide).T)))

    # Test `Woodbury`.
    wb = lr + dense_tall
    assert compare(wb, dense_square.T), 'wb x dense s'
    assert compare(dense_square, wb.T), 'dense s x wb'
    assert compare(wb, wb.T), 'wb x wb'
    assert compare(wb.T, wb), 'wb x wb (2)'


def test_inverse_and_logdet():
    # Test `Dense`.
    a = np.random.randn(3, 3)
    a = Dense(a.dot(a.T))
    allclose(B.matmul(a, B.inverse(a)), np.eye(3))
    allclose(B.matmul(B.inverse(a), a), np.eye(3))
    allclose(B.logdet(a), np.log(np.linalg.det(to_np(a))))

    # Test `Diagonal`.
    d = Diagonal(np.array([1, 2, 3]))
    allclose(B.matmul(d, B.inverse(d)), np.eye(3))
    allclose(B.matmul(B.inverse(d), d), np.eye(3))
    allclose(B.logdet(d), np.log(np.linalg.det(to_np(d))))
    assert B.shape(B.inverse(Diagonal(np.array([1, 2]),
                                      rows=2, cols=4))) == (4, 2)

    # Test `Woodbury`.
    a = np.random.randn(3, 2)
    b = np.random.randn(2, 2) + 1e-2 * np.eye(2)
    wb = d + LowRank(left=a, middle=b.dot(b.T))
    for _ in range(4):
        allclose(B.matmul(wb, B.inverse(wb)), np.eye(3))
        allclose(B.matmul(B.inverse(wb), wb), np.eye(3))
        allclose(B.logdet(wb), np.log(np.linalg.det(to_np(wb))))
        wb = B.inverse(wb)

    # Test `LowRank`.
    with pytest.raises(RuntimeError):
        B.inverse(wb.lr)
    with pytest.raises(RuntimeError):
        B.logdet(wb.lr)


def test_lr_diff():
    # First, test correctness.
    a = np.random.randn(3, 2)
    b = np.random.randn(2, 2)
    lr1 = LowRank(left=a, right=np.random.randn(3, 2), middle=b.dot(b.T))
    a = np.random.randn(3, 2)
    b = np.random.randn(2, 2)
    lr2 = LowRank(left=a, right=np.random.randn(3, 2), middle=b.dot(b.T))

    allclose(B.lr_diff(lr1, lr1), B.zeros(3, 3))
    allclose(B.lr_diff(lr1 + lr2, lr1), lr2)
    allclose(B.lr_diff(lr1 + lr2, lr2), lr1)
    allclose(B.lr_diff(lr1 + lr1 + lr2, lr1), lr1 + lr2)
    allclose(B.lr_diff(lr1 + lr1 + lr2, lr2), lr1 + lr1)
    allclose(B.lr_diff(lr1 + lr1 + lr2, lr1 + lr1), lr2)
    allclose(B.lr_diff(lr1 + lr1 + lr2, lr1 + lr2), lr1)
    allclose(B.lr_diff(lr1 + lr1 + lr2, lr1 + lr1 + lr2), B.zeros(3, 3))

    # Second, test positive definiteness.
    lr1 = LowRank(left=lr1.left, middle=lr1.middle)
    lr2 = LowRank(left=lr2.left, middle=lr2.middle)

    B.cholesky(B.lr_diff(lr1, 0.999 * lr1))
    B.cholesky(B.lr_diff(lr1 + lr2, lr1))
    B.cholesky(B.lr_diff(lr1 + lr2, lr2))
    B.cholesky(B.lr_diff(lr1 + lr1 + lr2, lr1))
    B.cholesky(B.lr_diff(lr1 + lr1 + lr2, lr1 + lr1))
    B.cholesky(B.lr_diff(lr1 + lr1 + lr2, lr1 + lr2))
    B.cholesky(B.lr_diff(lr1 + lr1 + lr2, lr1 + lr1 + 0.999 * lr2))


def test_root():
    # Test `Dense`.
    a = np.random.randn(5, 5)
    a = Dense(a.dot(a.T))
    allclose(a, B.matmul(B.root(a), B.root(a)))

    # Test `Diagonal`.
    d = Diagonal(np.array([1, 2, 3, 4, 5]))
    allclose(d, B.matmul(B.root(d), B.root(d)))


def test_sample():
    a = np.random.randn(3, 3)
    a = Dense(a.dot(a.T))
    b = np.random.randn(2, 2)
    wb = Diagonal(B.diag(a)) + LowRank(left=np.random.randn(3, 2),
                                       middle=b.dot(b.T))

    # Test `Dense` and `Woodbury`.
    num_samps = 500000
    for cov in [a, wb]:
        samps = B.sample(cov, num_samps)
        cov_emp = B.matmul(samps, samps, tr_b=True) / num_samps
        assert np.mean(np.abs(to_np(cov_emp) - to_np(cov))) <= 5e-2


def test_schur():
    # Test `Dense`.
    a = np.random.randn(5, 10)
    b = np.random.randn(3, 5)
    c = np.random.randn(3, 3)
    d = np.random.randn(3, 10)
    c = c.dot(c.T)

    allclose(B.schur(a, b, c, d), a - np.linalg.solve(c.T, b).T.dot(d),
             desc='n n n n')

    # Test `Woodbury`.
    #   The inverse of the Woodbury matrix already properly tests the method for
    #   Woodbury matrices.
    c = np.random.randn(2, 2)
    c = Diagonal(np.array([1, 2, 3])) + LowRank(left=np.random.randn(3, 2),
                                                middle=c.dot(c.T))
    allclose(B.schur(a, b, c, d), a - np.linalg.solve(to_np(c).T, b).T.dot(d),
             desc='n n w n')

    #   Test all combinations of `Woodbury`, `LowRank`, and `Diagonal`.
    a = np.random.randn(2, 2)
    a = Diagonal(np.array([4, 5, 6, 7, 8]), rows=5, cols=10) + LowRank(
        left=np.random.randn(5, 2),
        right=np.random.randn(10, 2),
        middle=a.dot(a.T))

    b = np.random.randn(2, 2)
    b = Diagonal(np.array([9, 10, 11]), rows=3, cols=5) + LowRank(
        left=c.lr.left,
        right=a.lr.left,
        middle=b.dot(b.T))

    d = np.random.randn(2, 2)
    d = Diagonal(np.array([12, 13, 14]), rows=3, cols=10) + LowRank(
        left=c.lr.right,
        right=a.lr.right,
        middle=d.dot(d.T))

    #     Loop over all combinations. Some of them should be efficient and
    #     representation preserving; all of them should be correct.
    for ai in [a, a.lr, a.diag]:
        for bi in [b, b.lr, b.diag]:
            for ci in [c, c.diag]:
                for di in [d, d.lr, d.diag]:
                    allclose(B.schur(ai, bi, ci, di),
                             to_np(ai) -
                             np.linalg.solve(to_np(ci).T,
                                             to_np(bi)).T.dot(to_np(di)),
                             desc='{} {} {} {}'.format(ai, bi, ci, di))


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
        allclose(B.qf(x, b), np.linalg.solve(to_np(x), b).T.dot(b))
        allclose(B.qf(x, b, b), B.qf(x, b))
        allclose(B.qf(x, b, c), np.linalg.solve(to_np(x), b).T.dot(c))
        allclose(B.qf_diag(x, b),
                 np.diag(np.linalg.solve(to_np(x), b).T.dot(b)))
        allclose(B.qf_diag(x, b, b), B.qf_diag(x, b, b))
        allclose(B.qf_diag(x, b, c),
                 np.diag(np.linalg.solve(to_np(x), b).T.dot(c)))

    # Test `LowRank`.
    lr = LowRank(np.random.randn(5, 3))
    with pytest.raises(RuntimeError):
        B.qf(lr, b)
    with pytest.raises(RuntimeError):
        B.qf(lr, b, c)
    with pytest.raises(RuntimeError):
        B.qf_diag(lr, b)
    with pytest.raises(RuntimeError):
        B.qf_diag(lr, b, c)


def test_ratio():
    a, b = np.random.randn(4, 4), np.random.randn(4, 4)
    a, b = Dense(a.dot(a.T)), Dense(b.dot(b.T))
    d, e = Diagonal(B.diag(a)), Diagonal(B.diag(b))
    c = np.random.randn(3, 3)
    lr = LowRank(left=np.random.randn(4, 3), middle=c.dot(c.T))

    allclose(B.ratio(a, b), np.trace(np.linalg.solve(to_np(b), to_np(a))))
    allclose(B.ratio(lr, b), np.trace(np.linalg.solve(to_np(b), to_np(lr))))
    allclose(B.ratio(d, e), np.trace(np.linalg.solve(to_np(e), to_np(d))))


def test_transposition():
    def compare(a):
        allclose(B.transpose(a), to_np(a).T)

    d = Diagonal(np.array([1, 2, 3]), rows=5, cols=10)
    lr = LowRank(left=np.random.randn(5, 3), right=np.random.randn(10, 3))

    compare(Dense(np.random.randn(5, 5)))
    compare(d)
    compare(lr)
    compare(d + lr)
    compare(Constant(5, rows=4, cols=2))


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
    allclose(a.__div__(5.0), to_np(a) / 5.0)
    allclose(a.__rdiv__(5.0), 5.0 / to_np(a))
    allclose(a.__truediv__(5.0), to_np(a) / 5.0)
    allclose(a.__rtruediv__(5.0), 5.0 / to_np(a))
    allclose(B.divide(a, 5.0), to_np(a) / 5.0)
    allclose(B.divide(a, a), B.ones(to_np(a)))

    # Check shapes.
    for m in candidates:
        assert B.shape(a) == (4, 3)

    # Check interactions.
    for m1, m2 in product(candidates, candidates):
        allclose(m1 * m2, to_np(m1) * to_np(m2))
        allclose(m1 + m2, to_np(m1) + to_np(m2))
        allclose(m1 - m2, to_np(m1) - to_np(m2))
