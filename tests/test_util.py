# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from lab import B
from numpy.testing import assert_approx_equal, assert_array_almost_equal

from stheno import Kernel
from stheno.util import  uprank
from stheno.input import Component
from stheno.kernel import ZeroKernel, OneKernel, EQ
from stheno.mean import ZeroMean, OneMean
from stheno.matrix import dense
from stheno.graph import GP, Graph
# noinspection PyUnresolvedReferences
from . import eq, neq, ok, raises, benchmark, le, assert_allclose, allclose


def test_uprank():
    yield assert_allclose, uprank(0), [[0]]
    yield assert_allclose, uprank(np.array([0])), [[0]]
    yield assert_allclose, uprank(np.array([[0]])), [[0]]
    yield eq, type(uprank(Component('test')(0))), Component('test')

    k = OneKernel()

    yield eq, B.shape(k(0, 0)), (1, 1)
    yield eq, B.shape(k(0, np.ones(5))), (1, 5)
    yield eq, B.shape(k(0, np.ones((5, 2)))), (1, 5)

    yield eq, B.shape(k(np.ones(5), 0)), (5, 1)
    yield eq, B.shape(k(np.ones(5), np.ones(5))), (5, 5)
    yield eq, B.shape(k(np.ones(5), np.ones((5, 2)))), (5, 5)

    yield eq, B.shape(k(np.ones((5, 2)), 0)), (5, 1)
    yield eq, B.shape(k(np.ones((5, 2)), np.ones(5))), (5, 5)
    yield eq, B.shape(k(np.ones((5, 2)), np.ones((5, 2)))), (5, 5)

    yield raises, ValueError, lambda: k(0, np.ones((5, 2, 1)))
    yield raises, ValueError, lambda: k(np.ones((5, 2, 1)))

    m = OneMean()

    yield eq, B.shape(m(0)), (1, 1)
    yield eq, B.shape(m(np.ones(5))), (5, 1)
    yield eq, B.shape(m(np.ones((5, 2)))), (5, 1)

    p = GP(EQ(), graph=Graph())
    x = np.linspace(0, 10, 10)

    yield assert_approx_equal, p.condition(1, 1)(1).mean, np.array([[1]])
    yield assert_array_almost_equal, p.condition(x, x)(x).mean, x[:, None]
    yield assert_array_almost_equal, p.condition(x, x[:, None])(x).mean, \
          x[:, None]
