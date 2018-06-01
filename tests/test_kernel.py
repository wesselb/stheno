# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from stheno import EQ, RQ, Matern12, Matern32, Matern52, Kronecker, Kernel, \
    Observed
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint


def test_corner_cases():
    yield raises, NotImplementedError, lambda: Kernel()(1.)
    x = np.random.randn(10, 2)
    yield ok, np.allclose(EQ()(x), EQ()(Observed(x)))


def test_kronecker_kernel():
    k = Kronecker()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(10, 2)

    yield ok, np.allclose(k(x1), np.eye(10)), 'same'
    yield ok, np.allclose(k(x1, x2), np.zeros((10, 10))), 'others'


def test_arithmetic():
    k1 = EQ()
    k2 = RQ(1e-1)
    k3 = Matern12()
    k4 = Matern32()
    k5 = Matern52()
    k6 = Kronecker()
    xs1 = np.random.randn(10, 2), np.random.randn(10, 2)
    xs2 = np.random.randn(), np.random.randn()

    yield ok, np.allclose(k6(xs1[0]), k6(xs1[0], xs1[0])), 'dispatch'
    yield ok, np.allclose((k1 * k2)(*xs1), k1(*xs1) * k2(*xs1)), 'prod'
    yield ok, np.allclose((k1 * k2)(*xs2), k1(*xs2) * k2(*xs2)), 'prod 2'
    yield ok, np.allclose((k3 + k4)(*xs1), k3(*xs1) + k4(*xs1)), 'sum'
    yield ok, np.allclose((k3 + k4)(*xs2), k3(*xs2) + k4(*xs2)), 'sum 2'
    yield ok, np.allclose((k3 - k4)(*xs1), k3(*xs1) - k4(*xs1)), 'sub'
    yield ok, np.allclose((k3 - k4)(*xs2), k3(*xs2) - k4(*xs2)), 'sub 2'
    yield ok, np.allclose((5. - k4)(*xs2), 5. - k4(*xs2)), 'sub 2'
    yield ok, np.allclose((-k4)(*xs1), -k4(*xs1)), 'neg'
    yield ok, np.allclose((-k4)(*xs2), -k4(*xs2)), 'neg 2'
    yield ok, np.allclose((5. * k5)(*xs1), 5. * k5(*xs1)), 'prod 3'
    yield ok, np.allclose((5. * k5)(*xs2), 5. * k5(*xs2)), 'prod 4'
    yield ok, np.allclose((5. + k5)(*xs1), 5. + k5(*xs1)), 'sum 3'
    yield ok, np.allclose((5. + k5)(*xs2), 5. + k5(*xs2)), 'sum 4'
    yield ok, np.allclose(k1.stretch(2.)(*xs1),
                          k1(xs1[0] / 2., xs1[1] / 2.)), 'stretch'
    yield ok, np.allclose(k1.stretch(2.)(*xs2),
                          k1(xs2[0] / 2., xs2[1] / 2.)), 'stretch 2'
    yield ok, np.allclose(k1.periodic(1.)(*xs1),
                          k1.periodic(1.)(xs1[0], xs1[1] + 5.)), 'periodic'
    yield ok, np.allclose(k1.periodic(1.)(*xs2),
                          k1.periodic(1.)(xs2[0], xs2[1] + 5.)), 'periodic 2'


def test_reverse():
    k = EQ()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    yield ok, np.allclose(k(x1, x2), reversed(k)(x2, x1).T)
