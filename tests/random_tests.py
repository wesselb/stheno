# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok

from stheno import Normal
import numpy as np
from scipy.stats import multivariate_normal


def test_normal():
    mean = np.random.randn(3, 1)
    chol = np.random.randn(3, 3)
    var = chol.dot(chol.T)

    dist = Normal(var, mean)
    dist_sp = multivariate_normal(mean[:, 0], var)

    x = np.random.randn(3, 10)
    yield ok, np.allclose(dist.log_pdf(x), np.sum(dist_sp.logpdf(x.T)))
    yield ok, np.allclose(dist.entropy(), dist_sp.entropy())
