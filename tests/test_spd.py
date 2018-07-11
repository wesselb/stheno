# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import product

import numpy as np
from lab import B
from stheno.spd import SPD, Diagonal, UniformlyDiagonal, LowRank, dense, \
    OneSPD, ZeroSPD, Woodbury, ConstantSPD

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint, allclose


def compare(spd1, spd2, spd1_singular=False, spd2_singular=False):
    # Create a dummy.
    a = np.random.randn(*B.shape(spd1))
    dummy = SPD(a.dot(a.T))

    # Create some random matrices.
    a = np.random.randn(B.shape(spd1)[0], 10)
    b = np.random.randn(B.shape(spd1)[0], 10)
    A = np.random.randn(*B.shape(spd1))

    # Compare implementations.
    yield ok, allclose(dense(spd1), dense(spd2)), 'matrices'
    yield ok, allclose(B.diag(spd1), B.diag(spd2)), 'diagonals'
    yield ok, B.shape(spd1) == B.shape(spd2), 'shapes'
    yield ok, allclose(B.cholesky(spd1), B.cholesky(spd2)), 'cholesky'
    yield ok, allclose(B.root(spd1), B.root(spd2)), 'roots'
    yield ok, allclose(B.cholesky_mul(spd1, A),
                       B.cholesky_mul(spd2, A)), 'chol mul'
    yield ok, allclose(B.ratio(spd1, dummy), B.ratio(spd2, dummy)), 'ratio'

    if not spd1_singular and not spd2_singular:
        yield ok, allclose(B.mah_dist2(spd1, a), B.mah_dist2(spd2, a)), 'mah'
        yield ok, allclose(B.mah_dist2(spd1, a, b),
                           B.mah_dist2(spd2, a, b)), 'mah'
        yield ok, allclose(B.qf(spd1, a), B.qf(spd2, a)), 'qf'
        yield ok, allclose(B.qf(spd1, a, b), B.qf(spd2, a, b)), 'qf 2'
        yield ok, allclose(B.qf_diag(spd1, a), B.qf_diag(spd2, a)), 'qf diag'
        yield ok, allclose(B.qf_diag(spd1, a, b),
                           B.qf_diag(spd2, a, b)), 'qf diag 2'
        yield ok, allclose(B.ratio(spd1, spd1), B.ratio(spd2, spd1)), 'ratio 2'
        yield ok, allclose(B.ratio(spd1, spd1), B.ratio(spd2, spd2)), 'ratio 3'
        yield ok, allclose(B.ratio(spd1, spd2), B.ratio(spd2, spd1)), 'ratio 4'
        yield ok, allclose(B.ratio(spd1, spd2), B.ratio(spd2, spd2)), 'ratio 5'
        yield ok, allclose(B.inv_prod(spd1, A), B.inv_prod(spd2, A)), 'inv prod'
        yield ok, allclose(B.logdet(spd1), B.logdet(spd2)), 'logdets'

    for singular, spd in [(spd1_singular, spd1), (spd2_singular, spd2)]:
        if singular:
            yield raises, RuntimeError, lambda: B.mah_dist2(spd, a)
            yield raises, RuntimeError, lambda: B.mah_dist2(spd, a, a)
            yield raises, RuntimeError, lambda: B.qf(spd, a)
            yield raises, RuntimeError, lambda: B.qf(spd, a, a)
            yield raises, RuntimeError, lambda: B.qf_diag(spd, a)
            yield raises, RuntimeError, lambda: B.qf_diag(spd, a, a)
            yield raises, RuntimeError, lambda: B.inv_prod(spd, A)
            yield raises, RuntimeError, lambda: B.logdet(spd)


def test_spd():
    # Compare dense and diagonal.
    a = np.diag(np.random.randn(3) ** 2)
    for x in compare(SPD(a), Diagonal(np.diag(a))):
        yield x

    # Compare dense and uniformly diagonal.
    a = np.random.randn() ** 2
    for x in compare(SPD(np.eye(3) * a), UniformlyDiagonal(a, 3)):
        yield x

    # Compare dense and low-rank.
    a = np.random.randn(3, 2)
    for x in compare(SPD(a.dot(a.T)), LowRank(a), spd2_singular=True):
        yield x

    # Compare dense and Woodbury.
    a = np.random.randn(3, 2)
    b = np.random.randn(3) ** 2
    for x in compare(SPD(a.dot(a.T) + np.diag(b)), LowRank(a) + Diagonal(b)):
        yield x

    # Compare diagonal and uniformly diagonal.
    a = np.random.randn() ** 2
    for x in compare(Diagonal(np.ones(3) * a), UniformlyDiagonal(a, 3)):
        yield x

    # Compare diagonal and low-rank.
    a = np.diag(np.random.randn(3))
    for x in compare(Diagonal(np.diag(a ** 2)), LowRank(a), spd2_singular=True):
        yield x

    # Compare diagonal and Woodbury.
    a = np.diag(np.random.randn(3))
    b = np.random.randn(3) ** 2
    for x in compare(Diagonal(np.diag(a) ** 2 + b),
                     LowRank(a) + Diagonal(b)):
        yield x

    # Compare uniformly diagonal and low-rank.
    a = np.eye(3) * np.random.randn()
    for x in compare(UniformlyDiagonal(a[0, 0] ** 2, 3), LowRank(a),
                     spd2_singular=True):
        yield x

    # Compare uniformly diagonal and Woodbury.
    a = np.eye(3) * np.random.randn()
    b = np.ones(3) * np.random.randn() ** 2
    for x in compare(UniformlyDiagonal(a[0, 0] ** 2 + b[0], 3),
                     LowRank(a) + Diagonal(b)):
        yield x

    # Compare low-rank and Woodbury.
    a = np.random.randn(3, 2)
    b = np.zeros(3)
    for x in compare(LowRank(a), LowRank(a) + Diagonal(b), spd1_singular=True):
        yield x


def test_basic_spd_arithmetic():
    # Generate a bunch of SPDs.
    spd = SPD(np.eye(3))
    diag = Diagonal(np.random.randn(3) ** 2)
    lr = LowRank(np.random.randn(3, 2))
    woodbury = Woodbury(LowRank(np.random.randn(3, 2)),
                        Diagonal(np.random.randn(3) ** 2))
    one = OneSPD(spd)
    zero = ZeroSPD(spd)
    scalar = 5

    # Put them in a list.
    spds = [spd, diag, lr, woodbury, one, zero, scalar]

    # Verify that the right types are generated by addition.
    yield eq, type(spd + spd), SPD, 'add spd spd'
    yield eq, type(spd + diag), SPD, 'add spd diag'
    yield eq, type(spd + lr), SPD, 'add spd lr'
    yield eq, type(spd + woodbury), SPD, 'add spd Woodbury'
    yield eq, type(spd + one), SPD, 'add spd one'
    yield eq, type(spd + zero), SPD, 'add spd zero'
    yield eq, type(spd + scalar), SPD, 'add spd scalar'

    yield eq, type(diag + spd), SPD, 'add diag spd'
    yield eq, type(diag + diag), Diagonal, 'add diag diag'
    yield eq, type(diag + lr), Woodbury, 'add diag lr'
    yield eq, type(diag + woodbury), Woodbury, 'add diag Woodbury'
    yield eq, type(diag + one), Woodbury, 'add diag one'
    yield eq, type(diag + zero), Diagonal, 'add diag zero'
    yield eq, type(diag + scalar), Woodbury, 'add diag scalar'

    yield eq, type(lr + spd), SPD, 'add lr spd'
    yield eq, type(lr + diag), Woodbury, 'add lr diag'
    yield eq, type(lr + lr), LowRank, 'add lr lr'
    yield eq, type(lr + woodbury), Woodbury, 'add lr Woodbury'
    yield eq, type(lr + one), LowRank, 'add lr one'
    yield eq, type(lr + zero), LowRank, 'add lr zero'
    yield eq, type(lr + scalar), LowRank, 'add lr scalar'

    yield eq, type(woodbury + spd), SPD, 'add Woodbury spd'
    yield eq, type(woodbury + diag), Woodbury, 'add Woodbury diag'
    yield eq, type(woodbury + lr), Woodbury, 'add Woodbury lr'
    yield eq, type(woodbury + woodbury), Woodbury, 'add Woodbury Woodbury'
    yield eq, type(woodbury + one), Woodbury, 'add Woodbury one'
    yield eq, type(woodbury + zero), Woodbury, 'add Woodbury zero'
    yield eq, type(woodbury + scalar), Woodbury, 'add Woodbury scalar'

    yield eq, type(one + spd), SPD, 'add one spd'
    yield eq, type(one + diag), Woodbury, 'add one diag'
    yield eq, type(one + lr), LowRank, 'add one lr'
    yield eq, type(one + woodbury), Woodbury, 'add one Woodbury'
    yield eq, type(one + one), LowRank, 'add one one'
    yield eq, type(one + zero), OneSPD, 'add one zero'
    yield eq, type(one + scalar), LowRank, 'add one scalar'

    yield eq, type(zero + spd), SPD, 'add zero spd'
    yield eq, type(zero + diag), Diagonal, 'add zero diag'
    yield eq, type(zero + lr), LowRank, 'add zero lr'
    yield eq, type(zero + woodbury), Woodbury, 'add zero Woodbury'
    yield eq, type(zero + one), OneSPD, 'add zero one'
    yield eq, type(zero + zero), ZeroSPD, 'add zero zero'
    yield eq, type(zero + scalar), ConstantSPD, 'add zero scalar'

    yield eq, type(scalar + spd), SPD, 'add scalar spd'
    yield eq, type(scalar + diag), Woodbury, 'add scalar diag'
    yield eq, type(scalar + lr), LowRank, 'add scalar lr'
    yield eq, type(scalar + woodbury), Woodbury, 'add scalar Woodbury'
    yield eq, type(scalar + one), LowRank, 'add scalar one'
    yield eq, type(scalar + zero), ConstantSPD, 'add scalar zero'
    yield eq, type(scalar + scalar), int, 'add scalar scalar'

    # Verify that the right types are generated by multiplication.
    yield eq, type(spd * spd), SPD, 'mul spd spd'
    yield eq, type(spd * diag), Diagonal, 'mul spd diag'
    yield eq, type(spd * lr), SPD, 'mul spd lr'
    yield eq, type(spd * woodbury), SPD, 'mul spd Woodbury'
    yield eq, type(spd * one), SPD, 'mul spd one'
    yield eq, type(spd * zero), ZeroSPD, 'mul spd zero'
    yield eq, type(spd * scalar), SPD, 'mul spd scalar'

    yield eq, type(diag * spd), Diagonal, 'mul diag spd'
    yield eq, type(diag * diag), Diagonal, 'mul diag diag'
    yield eq, type(diag * lr), Diagonal, 'mul diag lr'
    yield eq, type(diag * woodbury), Diagonal, 'mul diag Woodbury'
    yield eq, type(diag * one), Diagonal, 'mul diag one'
    yield eq, type(diag * zero), ZeroSPD, 'mul diag zero'
    yield eq, type(diag * scalar), Diagonal, 'mul diag scalar'

    yield eq, type(lr * spd), SPD, 'mul lr spd'
    yield eq, type(lr * diag), Diagonal, 'mul lr diag'
    yield eq, type(lr * lr), LowRank, 'mul lr lr'
    yield eq, type(lr * woodbury), Woodbury, 'mul lr Woodbury'
    yield eq, type(lr * one), LowRank, 'mul lr one'
    yield eq, type(lr * zero), ZeroSPD, 'mul lr zero'
    yield eq, type(lr * scalar), LowRank, 'mul lr scalar'

    yield eq, type(woodbury * spd), SPD, 'mul Woodbury spd'
    yield eq, type(woodbury * diag), Diagonal, 'mul Woodbury diag'
    yield eq, type(woodbury * lr), Woodbury, 'mul Woodbury lr'
    yield eq, type(woodbury * woodbury), Woodbury, 'mul Woodbury Woodbury'
    yield eq, type(woodbury * one), Woodbury, 'mul Woodbury one'
    yield eq, type(woodbury * zero), ZeroSPD, 'mul Woodbury zero'
    yield eq, type(woodbury * scalar), Woodbury, 'mul Woodbury scalar'

    yield eq, type(one * spd), SPD, 'mul one spd'
    yield eq, type(one * diag), Diagonal, 'mul one diag'
    yield eq, type(one * lr), LowRank, 'mul one lr'
    yield eq, type(one * woodbury), Woodbury, 'mul one Woodbury'
    yield eq, type(one * one), OneSPD, 'mul one one'
    yield eq, type(one * zero), ZeroSPD, 'mul one zero'
    yield eq, type(one * scalar), ConstantSPD, 'mul one scalar'

    yield eq, type(zero * spd), ZeroSPD, 'mul zero spd'
    yield eq, type(zero * diag), ZeroSPD, 'mul zero diag'
    yield eq, type(zero * lr), ZeroSPD, 'mul zero lr'
    yield eq, type(zero * woodbury), ZeroSPD, 'mul zero Woodbury'
    yield eq, type(zero * one), ZeroSPD, 'mul zero one'
    yield eq, type(zero * zero), ZeroSPD, 'mul zero zero'
    yield eq, type(zero * scalar), ZeroSPD, 'mul zero scalar'

    yield eq, type(scalar * spd), SPD, 'mul scalar spd'
    yield eq, type(scalar * diag), Diagonal, 'mul scalar diag'
    yield eq, type(scalar * lr), LowRank, 'mul scalar lr'
    yield eq, type(scalar * woodbury), Woodbury, 'mul scalar Woodbury'
    yield eq, type(scalar * one), ConstantSPD, 'mul scalar one'
    yield eq, type(scalar * zero), ZeroSPD, 'mul scalar zero'
    yield eq, type(scalar * scalar), int, 'mul scalar scalar'

    # Check that the algebra is correct by comparing the dense matrices.
    for x, y in product(spds, spds):
        yield ok, allclose(dense(x) * dense(y), dense(x * y)), 'mul', x, y
        yield ok, allclose(dense(x) + dense(y), dense(x + y)), 'add', x, y
