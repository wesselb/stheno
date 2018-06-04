# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from stheno import EQ, RQ, Matern12, Matern32, Matern52, Delta, Kernel, \
    Observed, Linear, ConstantKernel, ZeroKernel, Exp, PosteriorCrossKernel, \
    SPD, KernelCache, ProductKernel
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint, lam


def test_corner_cases():
    yield raises, NotImplementedError, lambda: Kernel()(1.)
    x = np.random.randn(10, 2)
    yield ok, np.allclose(EQ()(x), EQ()(Observed(x)))


def test_arithmetic():
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


def test_reverse():
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
    yield eq, k.period, 0

    # Verify that the kernel has the right properties.
    k = reversed(Linear())
    yield eq, k.stationary, False
    yield ok, k.var is np.nan
    yield ok, k.length_scale is np.nan
    yield eq, k.period, 0


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
    yield eq, k.period, 0

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
    yield eq, k.period, 0


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
    yield eq, k.period, 0


def test_kernel_exp():
    k = Exp()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield eq, k.length_scale, 1
    yield eq, k.period, 0


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
    yield eq, k.period, 0


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
    yield eq, k.period, 0


def test_kernel_constant():
    k = ConstantKernel()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)
    yield ok, np.allclose(k(x1, x2), np.ones((10, 5)))

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, True
    yield eq, k.var, 1
    yield ok, k.length_scale is np.inf
    yield eq, k.period, 0


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


def test_kernel_linear():
    k = Linear()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn(5, 2)

    # Test that the kernel computes.
    yield ok, np.allclose(k(x1, x2), k(x2, x1).T)

    # Verify that the kernel has the right properties.
    yield eq, k.stationary, False
    yield ok, k.var is np.nan
    yield ok, k.length_scale is np.nan
    yield eq, k.period, 0


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
    yield ok, k.var is np.nan
    yield ok, k.length_scale is np.nan
    yield eq, k.period, 0


def test_properties_sum():
    k1 = EQ().stretch(2)
    k2 = 3 * RQ(1e-2).stretch(5)
    k = k1 + k2

    yield eq, k.stationary, True
    yield ok, np.allclose(k.length_scale, (1 * 2 + 3 * 5) / 4)
    yield eq, k.period, 0
    yield eq, k.var, 4

    yield ok, np.allclose((EQ() + EQ()).length_scale, 1)
    yield ok, np.allclose((EQ().stretch(2) + EQ().stretch(2)).length_scale, 2)


def test_properties_stretch():
    k = EQ().stretch(2)

    yield eq, k.stationary, True
    yield eq, k.length_scale, 2
    yield eq, k.period, 0
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
    yield eq, k.period, 0
    yield eq, k.var, 2


def test_properties_product():
    k = (2 * EQ().stretch(10)) * (3 * RQ(1e-2).stretch(20))

    yield eq, k.stationary, True
    yield eq, k.length_scale, 10
    yield eq, k.period, 0
    yield eq, k.var, 6


def test_cancellations_zero():
    # Sums:
    yield eq, str(EQ() + EQ()), '(EQ() + EQ())'
    yield eq, str(ZeroKernel() + EQ()), 'EQ()'
    yield eq, str(EQ() + ZeroKernel()), 'EQ()'
    yield eq, str(ZeroKernel() + ZeroKernel()), '0'

    # Products:
    yield eq, str(EQ() * EQ()), '(EQ() * EQ())'
    yield eq, str(ZeroKernel() * EQ()), '0'
    yield eq, str(EQ() * ZeroKernel()), '0'
    yield eq, str(ZeroKernel() * ZeroKernel()), '0'

    # Scales:
    yield eq, str(5 * ZeroKernel()), '0'
    yield eq, str(ZeroKernel() * 5), '0'
    yield eq, str(EQ() * 5), '(5 * EQ())'
    yield eq, str(5 * EQ()), '(5 * EQ())'

    # Stretches:
    yield eq, str(ZeroKernel().stretch(5)), '0'
    yield eq, str(EQ().stretch(5)), '(EQ() > 5)'

    # Periodicisations:
    yield eq, str(ZeroKernel().periodic(5)), '0'
    yield eq, str(EQ().periodic(5)), '(EQ() per 5)'

    # Reversals:
    yield eq, str(reversed(ZeroKernel())), '0'
    yield eq, str(reversed(EQ())), 'EQ()'
    yield eq, str(reversed(Linear())), 'Reversed(Linear())'

    # Integration:
    yield eq, str(EQ() * EQ() + ZeroKernel() * EQ()), '(EQ() * EQ())'
    yield eq, str(EQ() * ZeroKernel() + ZeroKernel() * EQ()), '0'


def test_cancellations_one():
    # Products:
    yield eq, str(EQ() * EQ()), '(EQ() * EQ())'
    yield eq, str(ConstantKernel() * EQ()), 'EQ()'
    yield eq, str(EQ() * ConstantKernel()), 'EQ()'
    yield eq, str(ConstantKernel() * ConstantKernel()), '1'


def test_grouping():
    # Scales:
    yield eq, str(5 * EQ()), '(5 * EQ())'
    yield eq, str(5 * (5 * EQ())), '(25 * EQ())'

    # Stretches:
    yield eq, str(EQ().stretch(5)), '(EQ() > 5)'
    yield eq, str(EQ().stretch(5).stretch(5)), '(EQ() > 25)'

    # Products:
    yield eq, str((5 * EQ()) * (5 * EQ())), '(25 * EQ())'
    yield eq, str((5 * (EQ() * EQ())) * (5 * (EQ() * EQ()))), \
          '(25 * (EQ() * EQ()))'
    yield eq, str((5 * RQ(1)) * (5 * RQ(2))), '((5 * RQ(1)) * (5 * RQ(2)))'

    # Sums:
    yield eq, str((5 * EQ()) + (5 * EQ())), '(10 * EQ())'
    yield eq, str((5 * (EQ() * EQ())) + (5 * (EQ() * EQ()))), \
          '(10 * (EQ() * EQ()))'
    yield eq, str((5 * RQ(1)) + (5 * RQ(2))), '((5 * RQ(1)) + (5 * RQ(2)))'

    # Reversal:
    yield eq, str(reversed(Linear() + Linear())), '(Reversed(Linear()) + ' \
                                                  'Reversed(Linear()))'
    yield eq, str(reversed(Linear() * Linear())), '(Reversed(Linear()) * ' \
                                                  'Reversed(Linear()))'


def test_distributive_property():
    k1 = RQ(1)
    k2 = RQ(2)
    k3 = RQ(3)
    k4 = RQ(4)
    yield eq, str(k1 * (k2 + k3)), '((RQ(1) * RQ(2)) + (RQ(1) * RQ(3)))'
    yield eq, str((k1 + k2) * k3), '((RQ(1) * RQ(3)) + (RQ(2) * RQ(3)))'
    yield eq, str((k1 + k2) * (k3 + k4)), \
          '((((RQ(1) * RQ(3)) + (RQ(1) * RQ(4))) + ' \
          '(RQ(2) * RQ(3))) + (RQ(2) * RQ(4)))'


def test_kernel_cache():
    c = KernelCache()

    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)
    x2 = np.random.randn(10, 10)

    yield eq, id(c.pw_dists(x1, x2)), id(c.pw_dists(x1, x2))
    yield neq, id(c.pw_dists(x1, x1)), id(c.pw_dists(x1, x2))
    yield eq, id(c.matmul(x1, x2, tr_a=True)), id(c.matmul(x1, x2, tr_a=True))
    yield neq, id(c.matmul(x1, x2, tr_a=True)), id(c.matmul(x1, x2))

    # Test that ones and zeros are cached.
    k = ZeroKernel()
    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)
    yield eq, id(k(x1, c)), id(k(x2, c))
    x1, x2 = np.random.randn(10, 10), np.random.randn(5, 10)
    yield neq, id(k(x1, c)), id(k(x2, c))
    yield eq, id(k(1, c)), id(k(1, c))

    k = ConstantKernel()
    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)
    yield eq, id(k(x1, c)), id(k(x2, c))
    x1, x2 = np.random.randn(10, 10), np.random.randn(5, 10)
    yield neq, id(k(x1, c)), id(k(x2, c))
    yield eq, id(k(1, c)), id(k(1, c))
