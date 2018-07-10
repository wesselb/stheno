# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from plum import Dispatcher
from lab import B

from stheno.spd import SPD, Diagonal, UniformDiagonal, LowRank, dense
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint, allclose

dispatch = Dispatcher()


def test_spd():
    a = np.random.randn(3, 3)
    dummy = SPD(a.dot(a.T))

    def compare(ref, spd, invertible=True):
        a = np.random.randn(3, 10)
        b = np.random.randn(3, 10)
        A = np.random.randn(3, 3)

        # Compare implementations.
        yield ok, allclose(dense(ref), dense(spd)), 'matrices'
        yield ok, allclose(B.diag(ref), B.diag(spd)), 'diagonals'
        yield ok, B.shape(ref) == B.shape(spd), 'shapes'
        yield ok, allclose(B.cholesky(ref), B.cholesky(spd)), 'cholesky'
        yield ok, allclose(B.root(ref), B.root(spd)), 'roots'
        yield ok, allclose(B.cholesky_mul(ref, A),
                           B.cholesky_mul(spd, A)), 'chol mul'
        yield ok, allclose(B.ratio(ref, dummy), B.ratio(spd, dummy)), 'ratio'

        if invertible:
            yield ok, allclose(B.mah_dist2(ref, a), B.mah_dist2(spd, a)), 'mah'
            yield ok, allclose(B.mah_dist2(ref, a, b),
                               B.mah_dist2(spd, a, b)), 'mah'
            yield ok, allclose(B.qf(ref, a), B.qf(spd, a)), 'qf'
            yield ok, allclose(B.qf(ref, a, b), B.qf(spd, a, b)), 'qf 2'
            yield ok, allclose(B.qf_diag(ref, a), B.qf_diag(spd, a)), 'qf diag'
            yield ok, allclose(B.qf_diag(ref, a, b),
                               B.qf_diag(spd, a, b)), 'qf diag 2'
            yield ok, allclose(B.ratio(ref, ref), B.ratio(spd, ref)), 'ratio 2'
            yield ok, allclose(B.ratio(ref, ref), B.ratio(spd, spd)), 'ratio 3'
            yield ok, allclose(B.ratio(ref, spd), B.ratio(spd, ref)), 'ratio 4'
            yield ok, allclose(B.ratio(ref, spd), B.ratio(spd, spd)), 'ratio 5'
            yield ok, allclose(B.inv_prod(ref, A),
                               B.inv_prod(spd, A)), 'inv_prod'
            yield ok, allclose(B.logdet(ref), B.logdet(spd)), 'logdets'
        else:
            yield raises, RuntimeError, lambda: B.mah_dist2(spd, a)
            yield raises, RuntimeError, lambda: B.mah_dist2(spd, a, a)
            yield raises, RuntimeError, lambda: B.qf(spd, a)
            yield raises, RuntimeError, lambda: B.qf(spd, a, a)
            yield raises, RuntimeError, lambda: B.qf_diag(spd, a)
            yield raises, RuntimeError, lambda: B.qf_diag(spd, a, a)
            yield raises, RuntimeError, lambda: B.inv_prod(spd, A)
            yield raises, RuntimeError, lambda: B.logdet(spd)

    # Compare Dense and diagonal implementation.
    a = np.diag(np.random.randn(3) ** 2)
    spd = SPD(a)
    spd_diag = Diagonal(np.diag(a))
    for x in compare(spd, spd_diag):
        yield x

    # Compare Dense and uniform diagonal implementation.
    a = np.random.randn() ** 2
    spd = SPD(np.eye(3) * a)
    spd_diag_uniform = UniformDiagonal(a, 3)
    for x in compare(spd, spd_diag_uniform):
        yield x

    # Compare Dense and low-rank implementation.
    a = np.random.randn(3, 2)
    spd = SPD(a.dot(a.T))
    spd_low_rank = LowRank(a)
    for x in compare(spd, spd_low_rank, invertible=False):
        yield x


def test_spd_arithmetic():
    spd = SPD(np.eye(3))
    diag = Diagonal(np.ones(3))

    yield eq, type(spd + spd), SPD
    yield eq, type(spd + diag), SPD
    yield eq, type(diag + spd), SPD
    yield eq, type(diag + diag), Diagonal

    yield ok, allclose(dense(spd) + dense(diag), dense(spd + diag))

    yield eq, type(spd * spd), SPD
    yield eq, type(spd * diag), SPD
    yield eq, type(diag * spd), SPD
    yield eq, type(diag * diag), Diagonal

    yield ok, allclose(dense(spd) * dense(diag), dense(spd * diag))

    yield eq, type(5 * spd), SPD
    yield eq, type(5 * diag), Diagonal

    yield ok, allclose(5 * dense(spd), dense(5 * spd))
    yield ok, allclose(5 * dense(diag), dense(5 * diag))

    yield eq, type(5 + spd), SPD
    yield eq, type(5 + diag), SPD

    yield ok, allclose(5 + dense(spd), dense(5 + spd))
    yield ok, allclose(5 + dense(diag), dense(5 + diag))


def test_lab_interaction():
    diag = Diagonal(np.ones(3))

    yield eq, type(B.add(diag, diag)), Diagonal
    yield eq, type(B.add(5, diag)), SPD
    yield eq, type(B.add(diag, 5)), SPD

    yield eq, type(B.multiply(diag, diag)), Diagonal
    yield eq, type(B.multiply(5, diag)), Diagonal
    yield eq, type(B.multiply(diag, 5)), Diagonal
