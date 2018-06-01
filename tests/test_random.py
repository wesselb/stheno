# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from plum import Dispatcher
from scipy.stats import multivariate_normal

from stheno import Normal, Diagonal, UniformDiagonal, GP, EQ, RQ, \
    FunctionMean
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint

dispatch = Dispatcher()


def test_normal():
    mean = np.random.randn(3, 1)
    chol = np.random.randn(3, 3)
    var = chol.dot(chol.T)

    dist = Normal(var, mean)
    dist_sp = multivariate_normal(mean[:, 0], var)

    # Test second moment.
    yield ok, np.allclose(dist.m2(), var + mean.dot(mean.T))

    # Test `log_pdf` and `entropy`.
    x = np.random.randn(3, 10)
    yield ok, np.allclose(dist.log_pdf(x), dist_sp.logpdf(x.T)), 'log pdf'
    yield ok, np.allclose(dist.entropy(), dist_sp.entropy()), 'entropy'

    # Test KL with Monte Carlo estimate.
    mean2 = np.random.randn(3, 1)
    chol2 = np.random.randn(3, 3)
    var2 = chol2.dot(chol2.T)
    dist2 = Normal(var2, mean2)
    samples = dist.sample(50000)
    kl_est = np.mean(dist.log_pdf(samples)) - np.mean(dist2.log_pdf(samples))
    kl = dist.kl(dist2)
    yield ok, np.abs(kl_est - kl) / np.abs(kl) < 1e-2, 'kl samples'

    # Check a diagonal normal and SPD normal.
    mean = np.random.randn(3, 1)
    var_diag = np.random.randn(3) ** 2
    var = np.diag(var_diag)
    dist1 = Normal(var, mean)
    dist2 = Normal(Diagonal(var_diag), mean)
    samples = dist1.sample(100)
    yield ok, np.allclose(dist1.log_pdf(samples),
                          dist2.log_pdf(samples)), 'log pdf'
    yield ok, np.allclose(dist1.entropy(), dist2.entropy()), 'entropy'
    yield ok, np.allclose(dist1.kl(dist2), 0.), 'kl 1'
    yield ok, np.allclose(dist1.kl(dist1), 0.), 'kl 2'
    yield ok, np.allclose(dist2.kl(dist2), 0.), 'kl 3'
    yield ok, np.allclose(dist2.kl(dist1), 0.), 'kl 4'
    yield ok, dist1.w2(dist1) < 1e-4, 'w2 1'
    yield ok, dist1.w2(dist2) < 1e-4, 'w2 2'
    yield ok, dist2.w2(dist1) < 1e-4, 'w2 3'
    yield ok, dist2.w2(dist2) < 1e-4, 'w2 4'

    # Check a uniformly diagonal normal and SPD normal.
    mean = np.random.randn(3, 1)
    var_diag_scale = np.random.randn() ** 2
    var = np.eye(3) * var_diag_scale
    dist1 = Normal(var, mean)
    dist2 = Normal(UniformDiagonal(var_diag_scale, 3), mean)
    samples = dist1.sample(100)
    yield ok, np.allclose(dist1.log_pdf(samples),
                          dist2.log_pdf(samples)), 'log_pdf'
    yield ok, np.allclose(dist1.entropy(), dist2.entropy()), 'entropy'
    yield ok, np.allclose(dist1.kl(dist2), 0.), 'kl 1'
    yield ok, np.allclose(dist1.kl(dist1), 0.), 'kl 2'
    yield ok, np.allclose(dist2.kl(dist2), 0.), 'kl 3'
    yield ok, np.allclose(dist2.kl(dist1), 0.), 'kl 4'
    yield ok, dist1.w2(dist1) < 1e-4, 'w2 1'
    yield ok, dist1.w2(dist2) < 1e-4, 'w2 2'
    yield ok, dist2.w2(dist1) < 1e-4, 'w2 3'
    yield ok, dist2.w2(dist2) < 1e-4, 'w2 4'


def test_normal_arithmetic():
    mean = np.random.randn(3, 1)
    chol = np.random.randn(3, 3)
    var = chol.dot(chol.T)
    dist = Normal(var, mean)

    mean = np.random.randn(3, 1)
    chol = np.random.randn(3, 3)
    var = chol.dot(chol.T)
    dist2 = Normal(var, mean)

    A = np.random.randn(3, 3)
    a = np.random.randn(1, 3)
    b = 5.
    yield ok, np.allclose((dist.__mul__(a)).mean,
                          dist.mean.dot(a)), 'mean mul'
    yield ok, np.allclose((dist.__mul__(a)).var,
                          a.dot(dist.var).dot(a.T)), 'var mul'
    yield ok, np.allclose((dist.__rmul__(A)).mean,
                          A.dot(dist.mean)), 'mean rmul'
    yield ok, np.allclose((dist.__rmul__(A)).var,
                          A.dot(dist.var).dot(A.T)), 'var rmul'
    yield ok, np.allclose((dist.__mul__(b)).mean, dist.mean * b), 'mean mul 2'
    yield ok, np.allclose((dist.__mul__(b)).var, dist.var * b ** 2), 'var mul 2'
    yield ok, np.allclose((dist.__rmul__(b)).mean, dist.mean * b), 'mean rmul 2'
    yield ok, np.allclose((dist.__rmul__(b)).var,
                          dist.var * b ** 2), 'var rmul 2'
    yield raises, NotImplementedError, lambda: dist.__mul__(dist)
    yield raises, NotImplementedError, lambda: dist.__rmul__(dist)
    yield ok, np.allclose((dist + dist2).mean,
                          dist.mean + dist2.mean), 'mean sum'
    yield ok, np.allclose((dist + dist2).var,
                          dist.var + dist2.var), 'var sum'
    yield ok, np.allclose((dist.__add__(b)).mean, dist.mean + b), 'mean add'
    yield ok, np.allclose((dist.__radd__(b)).mean, dist.mean + b), 'mean radd'


@dispatch(Normal, Normal)
def close(n1, n2):
    return np.allclose(n1.mean, n2.mean) and np.allclose(n1.var, n2.var)


def test_gp():
    # Check finite-dimensional distribution construction.
    k = EQ()
    m = FunctionMean(lambda x: x ** 2)
    p = GP(k, m)
    x = np.random.randn(10, 1)

    yield ok, close(Normal(k(x), m(x)), p(x))

    # Check conditioning.
    sample = p(x).sample()
    post = p.condition(x, sample)

    mu, lower, upper = post.predict(x)
    yield ok, np.allclose(mu[:, None], sample), 'mean at known points'
    yield ok, np.all(upper - lower < 1e-5), 'sigma at known points'

    mu, lower, upper = post.predict(x + 15.)
    sig = (upper - lower) / 4
    yield ok, np.allclose(mu[:, None], m(x + 15.)), 'mean at unknown points'
    yield ok, np.allclose(sig ** 2,
                          np.diag(k(x + 15.))), 'variance at unknown points'


def test_gp_arithmetic():
    x = np.random.randn(10, 2)

    gp1 = GP(EQ())
    gp2 = GP(RQ(1e-1))

    yield raises, NotImplementedError, lambda: gp1 * gp2
    yield raises, NotImplementedError, lambda: gp1 + Normal(np.eye(3))
    yield ok, close((5. * gp1)(x), 5. * gp1(x)), 'mul'
    yield ok, close((gp1 + gp2)(x), gp1(x) + gp2(x)), 'add'
