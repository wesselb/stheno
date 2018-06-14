# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from plum import Dispatcher

from stheno.spd import SPD, Diagonal, UniformDiagonal
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint

dispatch = Dispatcher()


def test_corner_cases():
    yield raises, NotImplementedError, lambda: SPD(np.eye(3)) * np.eye(3)


def test_spd():
    def compare(spd1, spd2):
        a = np.random.randn(3, 10)
        b = np.random.randn(3, 10)
        A = np.random.randn(3, 3)

        yield ok, np.allclose(spd1.mat, spd2.mat), 'matrices'
        yield ok, np.allclose(spd1.diag, spd2.diag), 'diagonals'
        yield ok, spd1.shape == spd2.shape, 'shapes'
        yield ok, np.allclose(spd1.cholesky(),
                              spd2.cholesky()), 'cholesky'
        yield ok, np.allclose(spd1.root(), spd2.root()), 'roots'
        yield ok, np.allclose(spd1.log_det(), spd2.log_det()), 'log dets'
        yield ok, np.allclose(spd1.mah_dist2(a), spd2.mah_dist2(a)), 'mah'
        yield ok, np.allclose(spd1.mah_dist2(a, b),
                              spd2.mah_dist2(a, b)), 'mah 2'
        yield ok, np.allclose(spd1.quadratic_form(a),
                              spd2.quadratic_form(a)), 'qf'
        yield ok, np.allclose(spd1.quadratic_form(a, b),
                              spd2.quadratic_form(a, b)), 'qf 2'
        yield ok, np.allclose(spd1.ratio(spd1), spd2.ratio(spd1)), 'ratio'
        yield ok, np.allclose(spd1.ratio(spd1), spd2.ratio(spd2)), 'ratio 2'
        yield ok, np.allclose(spd1.ratio(spd2), spd2.ratio(spd1)), 'ratio 3'
        yield ok, np.allclose(spd1.ratio(spd2), spd2.ratio(spd2)), 'ratio 4'
        yield ok, np.allclose(spd1.inv_prod(A), spd2.inv_prod(A)), 'inv prod'
        yield ok, np.allclose(spd1.cholesky_mul(A),
                              spd2.cholesky_mul(A)), 'chol mul'

    # Compare SPD and diagonal implementation.
    a = np.diag(np.random.randn(3) ** 2)
    spd = SPD(a)
    spd_diag = Diagonal(np.diag(a))
    for x in compare(spd, spd_diag):
        yield x

    # Compare SPD and uniform diagonal implementation.
    a = np.random.randn() ** 2
    spd = SPD(np.eye(3) * a)
    spd_diag_uniform = UniformDiagonal(a, 3)
    for x in compare(spd, spd_diag_uniform):
        yield x


def test_spd_arithmetic():
    dense = SPD(np.eye(3))
    diag = Diagonal(np.ones(3))
    unif_diag = UniformDiagonal(1., 3)

    yield eq, type(dense + dense), SPD, 'add SPD'
    yield eq, type(dense + diag), SPD
    yield eq, type(dense + unif_diag), SPD
    yield eq, type(diag + dense), SPD
    yield eq, type(diag + diag), Diagonal
    yield eq, type(diag + unif_diag), Diagonal
    yield eq, type(unif_diag + dense), SPD
    yield eq, type(unif_diag + diag), Diagonal
    yield eq, type(unif_diag + unif_diag), UniformDiagonal

    yield ok, np.allclose(dense.mat + diag.mat, (dense + diag).mat)
    yield ok, np.allclose(diag.mat + unif_diag.mat, (diag + unif_diag).mat)
    yield ok, np.allclose(unif_diag.mat + unif_diag.mat,
                          (unif_diag + unif_diag).mat)

    yield eq, type(dense * dense), SPD, 'mul SPD'
    yield eq, type(dense * diag), SPD
    yield eq, type(dense * unif_diag), SPD
    yield eq, type(diag * dense), SPD
    yield eq, type(diag * diag), Diagonal
    yield eq, type(diag * unif_diag), Diagonal
    yield eq, type(unif_diag * dense), SPD
    yield eq, type(unif_diag * diag), Diagonal
    yield eq, type(unif_diag * unif_diag), UniformDiagonal

    yield ok, np.allclose(dense.mat * diag.mat, (dense * diag).mat)
    yield ok, np.allclose(diag.mat * unif_diag.mat, (diag * unif_diag).mat)
    yield ok, np.allclose(unif_diag.mat * unif_diag.mat,
                          (unif_diag * unif_diag).mat)

    yield eq, type(5. * dense), SPD, 'mul scalar'
    yield eq, type(5. * diag), Diagonal
    yield eq, type(5. * unif_diag), UniformDiagonal

    yield ok, np.allclose(5. * dense.mat, (5. * dense).mat)
    yield ok, np.allclose(5. * diag.mat, (5. * diag).mat)
    yield ok, np.allclose(5. * unif_diag.mat, (5. * unif_diag).mat)

    yield eq, type(5. + dense), SPD, 'add non-scalar'
    yield eq, type(5. + diag), SPD
    yield eq, type(5. + unif_diag), SPD

    yield ok, np.allclose(5. + dense.mat, (5. + dense).mat)
    yield ok, np.allclose(5. + diag.mat, (5. + diag).mat)
    yield ok, np.allclose(5. + unif_diag.mat, (5. + unif_diag).mat)
