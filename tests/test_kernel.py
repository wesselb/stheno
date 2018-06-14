# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_allclose
from lab import B
from plum import Dispatcher

from stheno.input import Observed
from stheno.kernel import EQ, RQ, Matern12, Matern32, Matern52, Delta, Kernel, \
    Linear, OneKernel, ZeroKernel, PosteriorCrossKernel, PosteriorKernel, \
    ShiftedKernel, FunctionKernel
from stheno.random import GPPrimitive
from stheno.spd import SPD
from stheno.cache import Cache
# noinspection PyUnresolvedReferences
from tests import ok
from . import eq, raises, ok


def test_corner_cases():
    yield raises, RuntimeError, lambda: Kernel()(1.)


def test_construction():
    k = EQ()

    x = np.random.randn(10, 1)
    c = Cache()

    yield k, x
    yield raises, RuntimeError, lambda: k(Observed(x))
    yield k, x, c
    yield raises, RuntimeError, lambda: k(Observed(x), c)
    yield k, x, x
    yield raises, RuntimeError, lambda: k(Observed(x), Observed(x))
    yield k, x, x, c
    yield raises, RuntimeError, lambda: k(Observed(x), Observed(x), c)


def test_basic_arithmetic():
    k1 = EQ()
    k2 = RQ(1e-1)
    k3 = Matern12()
    k4 = Matern32()
    k5 = Matern52()
    k6 = Delta()
    k7 = Linear()
    xs1 = np.random.randn(10, 2), np.random.randn(20, 2)
    xs2 = np.random.randn(), np.random.randn()

    yield ok, np.allclose(k6(xs1[0]), k6(xs1[0], xs1[0])), 'dispatch'
    yield ok, np.allclose((k1 * k2)(*xs1), k1(*xs1) * k2(*xs1)), 'prod'
    yield ok, np.allclose((k1 * k2)(*xs2), k1(*xs2) * k2(*xs2)), 'prod 2'
    yield ok, np.allclose((k3 + k4)(*xs1), k3(*xs1) + k4(*xs1)), 'sum'
    yield ok, np.allclose((k3 + k4)(*xs2), k3(*xs2) + k4(*xs2)), 'sum 2'
    yield ok, np.allclose((5. * k5)(*xs1), 5. * k5(*xs1)), 'prod 3'
    yield ok, np.allclose((5. * k5)(*xs2), 5. * k5(*xs2)), 'prod 4'
    yield ok, np.allclose((5. + k7)(*xs1), 5. + k7(*xs1)), 'sum 3'
    yield ok, np.allclose((5. + k7)(*xs2), 5. + k7(*xs2)), 'sum 4'
    yield ok, np.allclose(k1.stretch(2.)(*xs1),
                          k1(xs1[0] / 2., xs1[1] / 2.)), 'stretch'
    yield ok, np.allclose(k1.stretch(2.)(*xs2),
                          k1(xs2[0] / 2., xs2[1] / 2.)), 'stretch 2'
    yield ok, np.allclose(k1.periodic(1.)(*xs1),
                          k1.periodic(1.)(xs1[0], xs1[1] + 5.)), 'periodic'
    yield ok, np.allclose(k1.periodic(1.)(*xs2),
                          k1.periodic(1.)(xs2[0], xs2[1] + 5.)), 'periodic 2'


def test_reversal():
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)
    x3 = np.random.randn()

    # Test with a stationary and non-stationary kernel.
    for k in [EQ(), Linear()]:
        yield ok, np.allclose(k(x1), reversed(k)(x1))
        yield ok, np.allclose(k(x3), reversed(k)(x3))
        yield ok, np.allclose(k(x1, x2), reversed(k)(x1, x2))
        yield ok, np.allclose(k(x1, x2), reversed(k)(x2, x1).T)

        # Test double reversal does the right thing.
        yield ok, np.allclose(k(x1), reversed(reversed(k))(x1))
        yield ok, np.allclose(k(x3), reversed(reversed(k))(x3))
        yield ok, np.allclose(k(x1, x2), reversed(reversed(k))(x1, x2))
        yield ok, np.allclose(k(x1, x2), reversed(reversed(k))(x2, x1).T)

    # Verify that the kernel has the right properties.
    k = reversed(EQ())
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf

    # Verify that the kernel has the right properties.
    k = reversed(Linear())
    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.length_scale
    yield eq, k.period, np.inf
    yield eq, str(k), 'Reversed(Linear())'


def test_shifting():
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)
    k = Linear()
    yield ok, np.allclose(k.shift(5)(x1, x2), k(x1 - 5, x2 - 5))


def test_kernel_delta():
    k = Delta()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 0
    yield eq, k.period, np.inf
    yield eq, str(k), 'Delta()'

    yield ok, np.allclose(k(x1), np.eye(10)), 'same'
    yield ok, np.allclose(k(x1, x2), np.zeros((10, 5))), 'others'


def test_kernel_eq():
    k = EQ()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'EQ()'


def test_kernel_rq():
    k = RQ(1e-1)
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.alpha, 1e-1
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'RQ(0.1)'


def test_kernel_exp():
    k = Matern12()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'Exp()'


def test_kernel_mat32():
    k = Matern32()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'Matern32()'


def test_kernel_mat52():
    k = Matern52()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, str(k), 'Matern52()'


def test_kernel_one():
    k = OneKernel()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)
    yield ok, np.allclose(k(x1, x2), np.ones((10, 5)))

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 0
    yield eq, k.period, 0
    yield eq, str(k), '1'


def test_kernel_zero():
    k = ZeroKernel()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)
    yield ok, np.allclose(k(x1, x2), np.zeros((10, 5)))

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 0
    yield eq, k.length_scale, 0
    yield eq, k.period, 0
    yield eq, str(k), '0'


def test_kernel_linear():
    k = Linear()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.length_scale
    yield eq, k.period, np.inf
    yield eq, str(k), 'Linear()'


def test_kernel_posteriorcross():
    k = PosteriorCrossKernel(
        EQ(), EQ(), EQ(),
        np.random.randn(5, 2), SPD(EQ()(np.random.randn(5, 1)))
    )
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.period
    yield eq, str(k), 'PosteriorCrossKernel()'

    k = PosteriorKernel(GPPrimitive(EQ()), None, None)
    yield eq, str(k), 'PosteriorKernel()'


def test_properties_sum():
    k1 = EQ().stretch(2)
    k2 = 3 * RQ(1e-2).stretch(5)
    k = k1 + k2

    yield eq, k.stationary, True
    yield ok, np.allclose(k.length_scale, (1 * 2 + 3 * 5) / 4)
    yield eq, k.period, np.inf
    yield eq, k.var, 4

    yield ok, np.allclose((EQ() + EQ()).length_scale, 1)
    yield ok, np.allclose((EQ().stretch(2) + EQ().stretch(2)).length_scale, 2)


def test_properties_stretch():
    k = EQ().stretch(2)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 2
    yield eq, k.period, np.inf
    yield eq, k.var, 1

    k = EQ().stretch(1, 2)

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.period
    yield eq, k.var, 1


def test_properties_periodic():
    k = EQ().stretch(2).periodic(3)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 2
    yield eq, k.period, 3
    yield eq, k.var, 1

    k = 5 * k.stretch(5)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 10
    yield eq, k.period, 15
    yield eq, k.var, 5


def test_properties_scaled():
    k = 2 * EQ()

    yield eq, k.stationary, True
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, k.var, 2


def test_properties_shifted():
    k = ShiftedKernel(2 * EQ(), 5)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 1
    yield eq, k.period, np.inf
    yield eq, k.var, 2

    k = (2 * EQ()).shift(5, 6)

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield eq, k.period, np.inf
    yield eq, k.var, 2


def test_properties_product():
    k = (2 * EQ().stretch(10)) * (3 * RQ(1e-2).stretch(20))

    yield eq, k.stationary, True
    yield eq, k.length_scale, 10
    yield eq, k.period, np.inf
    yield eq, k.var, 6


def test_properties_selected():
    k = (2 * EQ().stretch(5)).select(0)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 5
    yield eq, k.period, np.inf
    yield eq, k.var, 2

    k = (2 * EQ().stretch(5)).select(2, 3)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 5
    yield eq, k.period, np.inf
    yield eq, k.var, 2

    k = (2 * EQ().stretch(np.array([1, 2, 3]))).select(0, 2)

    yield eq, k.stationary, True
    yield assert_allclose, k.length_scale, [1, 3]
    yield assert_allclose, k.period, [np.inf, np.inf]
    yield eq, k.var, 2

    k = (2 * EQ().periodic(np.array([1, 2, 3]))).select(1, 2)

    yield eq, k.stationary, True
    yield assert_allclose, k.length_scale, [1, 1]
    yield assert_allclose, k.period, [2, 3]
    yield eq, k.var, 2

    k = (2 * EQ().stretch(np.array([1, 2, 3]))).select((0, 2), (1, 2))

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.period
    yield eq, k.var, 2

    k = (2 * EQ().periodic(np.array([1, 2, 3]))).select((0, 2), (1, 2))

    yield eq, k.stationary, False
    yield eq, k.length_scale, 1
    yield raises, RuntimeError, lambda: k.period
    yield eq, k.var, 2


def test_properties_input_transform():
    k = Linear().transform(lambda x, c: x - 5)

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.period


def test_properties_derivative():
    k = EQ().diff(0)

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.period

    yield raises, RuntimeError, lambda: EQ().diff(None, None)(1)


def test_properties_function():
    k = FunctionKernel(lambda x: x ** 2)

    yield eq, k.stationary, False
    yield raises, RuntimeError, lambda: k.length_scale
    yield raises, RuntimeError, lambda: k.var
    yield raises, RuntimeError, lambda: k.period


def test_selection():
    # Test that computation is valid.
    k1 = EQ().select(1, 2)
    k2 = EQ()
    x = np.random.randn(10, 3)
    yield assert_allclose, k1(x), k2(x[:, [1, 2]])


def test_input_transform():
    k = Linear()
    x1, x2 = np.random.randn(10, 2), np.random.randn(10, 2)

    k2 = k.transform(lambda x, c: x ** 2)
    k3 = k.transform(lambda x, c: x ** 2, lambda x, c: x - 5)

    yield assert_allclose, k(x1 ** 2, x2 ** 2), k2(x1, x2)
    yield assert_allclose, k(x1 ** 2, x2 - 5), k3(x1, x2)


def test_derivative():
    B.backend_to_tf()
    s = B.Session()

    # Test derivative of kernel EQ.
    k = EQ()
    x1 = B.array(np.random.randn(10, 1))
    x2 = B.array(np.random.randn(5, 1))

    # Test derivative with respect to first input.
    ref = s.run(-k(x1, x2) * (x1 - B.transpose(x2)))
    yield assert_allclose, s.run(k.diff(0, None)(x1, x2)), ref
    ref = s.run(-k(x1) * (x1 - B.transpose(x1)))
    yield assert_allclose, s.run(k.diff(0, None)(x1)), ref

    # Test derivative with respect to second input.
    ref = s.run(-k(x1, x2) * (B.transpose(x2) - x1))
    yield assert_allclose, s.run(k.diff(None, 0)(x1, x2)), ref
    ref = s.run(-k(x1) * (B.transpose(x1) - x1))
    yield assert_allclose, s.run(k.diff(None, 0)(x1)), ref

    # Test derivative with respect to both inputs.
    ref = s.run(k(x1, x2) * (1 - (x1 - B.transpose(x2)) ** 2))
    yield assert_allclose, s.run(k.diff(0, 0)(x1, x2)), ref
    yield assert_allclose, s.run(k.diff(0)(x1, x2)), ref
    ref = s.run(k(x1) * (1 - (x1 - B.transpose(x1)) ** 2))
    yield assert_allclose, s.run(k.diff(0, 0)(x1)), ref
    yield assert_allclose, s.run(k.diff(0)(x1)), ref

    # Test derivative of kernel Linear.
    k = Linear()
    x1 = B.array(np.random.randn(10, 1))
    x2 = B.array(np.random.randn(5, 1))

    # Test derivative with respect to first input.
    ref = s.run(B.ones((10, 5), dtype=np.float64) * B.transpose(x2))
    yield assert_allclose, s.run(k.diff(0, None)(x1, x2)), ref
    ref = s.run(B.ones((10, 10), dtype=np.float64) * B.transpose(x1))
    yield assert_allclose, s.run(k.diff(0, None)(x1)), ref

    # Test derivative with respect to second input.
    ref = s.run(B.ones((10, 5), dtype=np.float64) * x1)
    yield assert_allclose, s.run(k.diff(None, 0)(x1, x2)), ref
    ref = s.run(B.ones((10, 10), dtype=np.float64) * x1)
    yield assert_allclose, s.run(k.diff(None, 0)(x1)), ref

    # Test derivative with respect to both inputs.
    ref = s.run(B.ones((10, 5), dtype=np.float64))
    yield assert_allclose, s.run(k.diff(0, 0)(x1, x2)), ref
    yield assert_allclose, s.run(k.diff(0)(x1, x2)), ref
    ref = s.run(B.ones((10, 10), dtype=np.float64))
    yield assert_allclose, s.run(k.diff(0, 0)(x1)), ref
    yield assert_allclose, s.run(k.diff(0)(x1)), ref

    s.close()
    B.backend_to_np()


def test_function():
    k = FunctionKernel(lambda x: x)
    x1 = np.linspace(0, 1, 100)[:, None]
    x2 = np.linspace(0, 1, 50)[:, None]

    yield assert_allclose, k(x1), x1 * x1.T
    yield assert_allclose, k(x1, x2), x1 * x2.T

    k = FunctionKernel(lambda x: x ** 2)

    yield assert_allclose, k(x1), x1 ** 2 * (x1 ** 2).T
    yield assert_allclose, k(x1, x2), (x1 ** 2) * (x2 ** 2).T
