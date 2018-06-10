# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_allclose

from stheno.graph import Graph, GP, At
from stheno.kernel import Linear, EQ
from stheno.mean import FunctionMean
# noinspection PyUnresolvedReferences,
from . import eq, raises, ok, le


def test_corner_cases():
    m1 = Graph()
    m2 = Graph()
    p1 = GP(EQ(), graph=m1)
    p2 = GP(EQ(), graph=m2)
    yield raises, RuntimeError, lambda: p1 + p2
    yield eq, str(GP(EQ(), graph=m1)), 'GP(EQ(), 0)'


def test_sum_other():
    model = Graph()
    p1 = GP(EQ(), FunctionMean(lambda x: x ** 2), graph=model)
    p2 = p1 + 5.
    p3 = 5. + p1
    p4 = model.sum(5., p1)

    x = np.random.randn(5, 1)
    yield assert_allclose, p1.mean(x) + 5., p2.mean(x)
    yield assert_allclose, p1.mean(x) + 5., p3.mean(x)
    yield assert_allclose, p1.mean(x) + 5., p4.mean(x)
    yield assert_allclose, p1.kernel(x), p2.kernel(x)
    yield assert_allclose, p1.kernel(x), p3.kernel(x)
    yield assert_allclose, p1.kernel(x), p4.kernel(x)
    yield assert_allclose, p1.kernel(At(p2)(x), At(p3)(x)), \
          p1.kernel(x)
    yield assert_allclose, p1.kernel(At(p2)(x), At(p4)(x)), \
          p1.kernel(x)


def test_mul_other():
    model = Graph()
    p1 = GP(EQ(), FunctionMean(lambda x: x ** 2), graph=model)
    p2 = 5. * p1
    p3 = p1 * 5.

    yield raises, NotImplementedError, lambda: p1 * p2

    x = np.random.randn(5, 1)
    yield assert_allclose, 5. * p1.mean(x), p2.mean(x)
    yield assert_allclose, 5. * p1.mean(x), p3.mean(x)
    yield assert_allclose, 25. * p1.kernel(x), p2.kernel(x)
    yield assert_allclose, 25. * p1.kernel(x), p3.kernel(x)
    yield assert_allclose, p1.kernel(At(p2)(x), At(p3)(x)), \
          25. * p1.kernel(x)


def test_at_shorthand():
    model = Graph()
    p1 = GP(EQ(), graph=model)
    x = p1.__matmul__(1)

    yield eq, type(x), At(p1)
    yield eq, x.get(), 1


def test_properties():
    model = Graph()

    p1 = GP(EQ(), graph=model)
    p2 = GP(EQ().stretch(2), graph=model)
    p3 = GP(EQ().periodic(10), graph=model)

    p = p1 + 2 * p2

    yield eq, p.stationary, True, 'stationary'
    yield eq, p.var, 1 + 2 ** 2, 'var'
    yield assert_allclose, p.length_scale, \
          (1 + 2 * 2 ** 2) / (1 + 2 ** 2)
    yield eq, p.period, 0, 'period'

    yield eq, p3.period, 10, 'period'

    p = p3 + p

    yield eq, p.stationary, True, 'stationary 2'
    yield eq, p.var, 1 + 2 ** 2 + 1, 'var 2'
    yield eq, p.period, 0, 'period 2'

    p = p + GP(Linear(), graph=model)

    yield eq, p.stationary, False, 'stationary 3'


def test_case1():
    # Test summing the same GP with itself.
    model = Graph()
    p1 = GP(EQ(), graph=model)
    p2 = p1 + p1 + p1 + p1 + p1
    x = np.linspace(0, 10, 5)[:, None]

    yield assert_allclose, p2(x).var, 25 * p1(x).var
    yield assert_allclose, p2(x).mean, np.zeros((5, 1))

    y = np.random.randn(5, 1)
    model.condition(At(p1)(x), y)

    yield assert_allclose, p2(x).mean, 5 * y


def test_case2():
    # Test some additive model.

    model = Graph()
    p1 = GP(EQ(), graph=model)
    p2 = GP(EQ(), graph=model)
    p3 = p1 + p2

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y1 = p1(x).sample()
    y2 = p2(x).sample()

    # First, test independence:
    yield assert_allclose, p1.kernel(At(p2)(x), x), np.zeros((n, n))
    yield assert_allclose, p1.kernel(At(p2)(x), At(p1)(x)), np.zeros((n, n))
    yield assert_allclose, p1.kernel(x, At(p2)(x)), np.zeros((n, n))
    yield assert_allclose, p1.kernel(At(p1)(x), At(p2)(x)), np.zeros((n, n))

    # Now run through some test cases:

    model.condition(At(p1)(x), y1)
    model.condition(At(p2)(x), y2)
    yield assert_allclose, p3(x).mean, y1 + y2
    model.revert_prior()

    model.condition(At(p2)(x), y2)
    model.condition(At(p1)(x), y1)
    yield assert_allclose, p3(x).mean, y1 + y2
    model.revert_prior()

    model.condition(At(p1)(x), y1)
    model.condition(At(p3)(x), y1 + y2)
    yield assert_allclose, p2(x).mean, y2
    model.revert_prior()

    model.condition(At(p3)(x), y1 + y2)
    model.condition(At(p1)(x), y1)
    yield assert_allclose, p2(x).mean, y2
    model.revert_prior()

    yield assert_allclose, p3 \
        .condition(At(p1)(x), y1) \
        .condition(At(p2)(x), y2)(x) \
        .mean, y1 + y2
    p3.revert_prior()

    yield assert_allclose, p3 \
        .condition(At(p2)(x), y2) \
        .condition(At(p1)(x), y1)(x) \
        .mean, y1 + y2
    p3.revert_prior()

    yield assert_allclose, p2 \
        .condition(At(p3)(x), y1 + y2) \
        .condition(At(p1)(x), y1)(x) \
        .mean, y2
    p3.revert_prior()

    yield assert_allclose, p2 \
        .condition(At(p1)(x), y1) \
        .condition(At(p3)(x), y1 + y2)(x) \
        .mean, y2
    p3.revert_prior()

    yield assert_allclose, p3.condition(x, y1 + y2)(x).mean, y1 + y2


def test_shifting():
    model = Graph()

    p = GP(EQ(), graph=model)
    p2 = p.shift(5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    yield assert_allclose, p.condition(At(p2)(x), y)(x - 5).mean, y
    yield le, np.sum(np.abs(p(x - 5).spd.diag)), 1e-10
    p.revert_prior()
    yield assert_allclose, p2.condition(At(p)(x), y)(x + 5).mean, y
    yield le, np.sum(np.abs(p2(x + 5).spd.diag)), 1e-10
    p.revert_prior()


def test_stretching():
    model = Graph()

    p = GP(EQ(), graph=model)
    p2 = p.stretch(5)

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    y = p2(x).sample()

    yield assert_allclose, p.condition(At(p2)(x), y)(x / 5).mean, y
    yield le, np.sum(np.abs(p(x / 5).spd.diag)), 1e-10
    p.revert_prior()
    yield assert_allclose, p2.condition(At(p)(x), y)(x * 5).mean, y
    yield le, np.sum(np.abs(p2(x * 5).spd.diag)), 1e-10
    p.revert_prior()


def test_selection():
    model = Graph()

    p = GP(EQ(), graph=model)  # 1D
    p2 = p.select(0)  # 2D

    n = 5
    x = np.linspace(0, 10, n)[:, None]
    x1 = np.concatenate((x, np.random.randn(n, 1)), axis=1)
    x2 = np.concatenate((x, np.random.randn(n, 1)), axis=1)
    y = p2(x).sample()

    yield assert_allclose, p.condition(At(p2)(x1), y)(x).mean, y
    yield le, np.sum(np.abs(p(x).spd.diag)), 1e-10
    p.revert_prior()

    yield assert_allclose, p.condition(At(p2)(x2), y)(x).mean, y
    yield le, np.sum(np.abs(p(x).spd.diag)), 1e-10
    p.revert_prior()

    yield assert_allclose, p2.condition(At(p)(x), y)(x1).mean, y
    yield assert_allclose, p2(x2).mean, y
    yield le, np.sum(np.abs(p2(x1).spd.diag)), 1e-10
    yield le, np.sum(np.abs(p2(x2).spd.diag)), 1e-10
    p.revert_prior()
