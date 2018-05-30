# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok

from stheno import Normal, Dense, Diagonal, UniformDiagonal
import numpy as np
from scipy.stats import multivariate_normal
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def test_normal():
    mean = np.random.randn(3, 1)
    chol = np.random.randn(3, 3)
    var = chol.dot(chol.T)

    dist = Normal(var, mean)
    dist_sp = multivariate_normal(mean[:, 0], var)

    # Test `log_pdf` and `entropy`.
    x = np.random.randn(3, 10)
    yield ok, np.allclose(dist.log_pdf(x), dist_sp.logpdf(x.T))
    yield ok, np.allclose(dist.entropy(), dist_sp.entropy())

    # Test KL with Monte Carlo estimate.
    mean2 = np.random.randn(3, 1)
    chol2 = np.random.randn(3, 3)
    var2 = chol2.dot(chol2.T)
    dist2 = Normal(var2, mean2)
    samples = dist.sample(50000)
    kl_est = np.mean(dist.log_pdf(samples)) - np.mean(dist2.log_pdf(samples))
    kl = dist.kl(dist2)
    yield ok, np.abs(kl_est - kl) / np.abs(kl) < 1e-2

    # Test transformations of a normal.
    A = np.random.randn(3, 3)
    a = np.random.randn(1, 3)
    b = 5.
    yield ok, np.allclose((dist.__mul__(a)).mean, dist.mean.dot(a))
    yield ok, np.allclose((dist.__mul__(a)).var, a.dot(dist.var).dot(a.T))
    yield ok, np.allclose((dist.__rmul__(A)).mean, A.dot(dist.mean))
    yield ok, np.allclose((dist.__rmul__(A)).var, A.dot(dist.var).dot(A.T))
    yield ok, np.allclose((dist.__mul__(b)).mean, dist.mean * b)
    yield ok, np.allclose((dist.__mul__(b)).var, dist.var * b ** 2)
    yield ok, np.allclose((dist.__rmul__(b)).mean, dist.mean * b)
    yield ok, np.allclose((dist.__rmul__(b)).var, dist.var * b ** 2)
    yield raises, NotImplementedError, lambda: dist.__mul__(dist)
    yield raises, NotImplementedError, lambda: dist.__rmul__(dist)
    yield ok, np.allclose((dist + dist2).mean, dist.mean + dist2.mean)
    yield ok, np.allclose((dist + dist2).var, dist.var + dist2.var)
    yield ok, np.allclose((dist.__add__(b)).mean, dist.mean + b)
    yield ok, np.allclose((dist.__radd__(b)).mean, dist.mean + b)

    # Check a diagonal normal and dense normal.
    mean = np.random.randn(3, 1)
    var_diag = np.random.randn(3) ** 2
    var = np.diag(var_diag)
    dist1 = Normal(var, mean)
    dist2 = Normal(Diagonal(var_diag), mean)
    samples = dist1.sample(100)
    yield ok, np.allclose(dist1.log_pdf(samples), dist2.log_pdf(samples))
    yield ok, np.allclose(dist1.entropy(), dist2.entropy())
    yield ok, np.allclose(dist1.kl(dist2), 0.)
    yield ok, np.allclose(dist2.kl(dist1), 0.)
    yield ok, dist1.w2(dist2) < 1e-5

    # Check a uniformly diagonal normal and dense normal.
    mean = np.random.randn(3, 1)
    var_diag_scale = np.random.randn() ** 2
    var = np.eye(3) * var_diag_scale
    dist1 = Normal(var, mean)
    dist2 = Normal(UniformDiagonal(var_diag_scale, 3), mean)
    samples = dist1.sample(100)
    yield ok, np.allclose(dist1.log_pdf(samples), dist2.log_pdf(samples))
    yield ok, np.allclose(dist1.entropy(), dist2.entropy())
    yield ok, np.allclose(dist1.kl(dist2), 0.)
    yield ok, np.allclose(dist2.kl(dist1), 0.)
    yield ok, dist1.w2(dist2) < 1e-5
    yield ok, dist2.w2(dist1) < 1e-5
    yield ok, dist2.w2(dist1) < 1e-5


def test_spd():
    def compare(spd1, spd2):
        a = np.random.randn(3, 10)
        b = np.random.randn(3, 10)
        A = np.random.randn(3, 3)

        yield ok, np.allclose(spd1.mat, spd2.mat), 'compare matrices'
        yield ok, np.allclose(spd1.diag, spd2.diag), 'compare diagonals'
        yield ok, spd1.shape == spd2.shape, 'compare shapes'
        yield ok, np.allclose(spd1.cholesky(), spd2.cholesky()), \
              'compare choleskies'
        yield ok, np.allclose(spd1.root(), spd2.root()), 'compare roots'
        yield ok, np.allclose(spd1.log_det(), spd2.log_det()), \
              'compare log dets'
        yield ok, np.allclose(spd1.mah_dist2(a), spd2.mah_dist2(a)), \
              'compare mah_dist2'
        yield ok, np.allclose(spd1.mah_dist2(a, b), spd2.mah_dist2(a, b)), \
              'compare mah_dist2 2'
        yield ok, np.allclose(spd1.quadratic_form(a), spd2.quadratic_form(a)), \
              'compare quadratic_form'
        yield ok, np.allclose(spd1.quadratic_form(a, b),
                              spd2.quadratic_form(a, b)), \
              'compare quadratic_form2 '
        yield ok, np.allclose(spd1.ratio(spd1), spd2.ratio(spd1)), \
              'compare ratio'
        yield ok, np.allclose(spd1.ratio(spd1), spd2.ratio(spd2)), \
              'compare ratio 2'
        yield ok, np.allclose(spd1.ratio(spd2), spd2.ratio(spd1)), \
              'compare ratio 3'
        yield ok, np.allclose(spd1.ratio(spd2), spd2.ratio(spd2)), \
              'compare ratio 4'
        yield ok, np.allclose(spd1.inv_prod(A), spd2.inv_prod(A)), \
              'compare inv_prod'

    # Compare dense and diagonal implementation.
    a = np.diag(np.random.randn(3) ** 2)
    spd = Dense(a)
    spd_diag = Diagonal(np.diag(a))
    for x in compare(spd, spd_diag):
        yield x

    # Compare dense and uniform diagonal implementation.
    a = np.random.randn() ** 2
    spd = Dense(np.eye(3) * a)
    spd_diag_uniform = UniformDiagonal(a, 3)
    for x in compare(spd, spd_diag_uniform):
        yield x
