# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number
from abc import ABCMeta, abstractproperty
import numpy as np

from lab import B
from plum import Dispatcher, Self, Referentiable

from stheno import Input

__all__ = ['Kernel', 'ProductKernel', 'SumKernel', 'ConstantKernel',
           'ScaledKernel', 'StretchedKernel', 'PeriodicKernel',
           'EQ', 'RQ', 'Matern12', 'Exp', 'Matern32', 'Matern52',
           'Kronecker', 'Linear', 'PosteriorKernel', 'ZeroKernel',
           'PosteriorCrossKernel']


class Kernel(Referentiable):
    """Kernel function.

    Kernels can be added and multiplied.
    """
    __metaclass__ = ABCMeta
    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, object)
    def __call__(self, x, y):
        """Construct the kernel for design matrices of points.

        Args:
            x (design matrix): First argument.
            y (design matrix, optional): Second argument. Defaults to first
                argument.

        Returns:
            Kernel matrix.
        """
        raise NotImplementedError('Could not resolve kernel arguments.')

    @dispatch(Input, Input)
    def __call__(self, x, y):
        return self(x.get(), y.get())

    @dispatch(object)
    def __call__(self, x):
        return self(x, x)

    @dispatch(Self)
    def __add__(self, other):
        return SumKernel(self, other)

    @dispatch(object)
    def __add__(self, other):
        return SumKernel(self, other * ConstantKernel())

    def __radd__(self, other):
        return self + other

    @dispatch(Self)
    def __mul__(self, other):
        return ProductKernel(self, other)

    @dispatch(object)
    def __mul__(self, other):
        return ScaledKernel(self, other)

    def __rmul__(self, other):
        return self * other

    def stretch(self, stretch):
        """Stretch the kernel.

        Args:
            scale (tensor): Stretch.
        """
        return StretchedKernel(self, stretch)

    def periodic(self, period=1):
        """Map to a periodic space.

        Args:
            period (tensor): Period. Defaults to `1`.
        """
        return PeriodicKernel(self, period)

    def __reversed__(self):
        """Reverse the arguments of the kernel."""
        return ReversedKernel(self)

    @property
    def stationary(self):
        """Stationarity of the kernel"""
        return True

    @property
    def var(self):
        """Variance of the kernel. Returns `np.nan` if the variance is
        undefined or cannot be determined.
        """
        return 1

    @property
    def length_scale(self):
        """Approximation to the length scale of the kernel. Returns `np.nan`
        if the length scale is undefined or cannot be determined.
        """
        return 1

    @property
    def period(self):
        """Period of the kernel. Returns `np.nan` is the period is undefined
        or cannot be determined.
        """
        return 0


class SumKernel(Kernel, Referentiable):
    """Sum of kernels.

    Args:
        k1 (instance of :class:`.kernel.Kernel`): First kernel in sum.
        k2 (instance of :class:`.kernel.Kernel`): Second kernel in sum.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @dispatch(object, object)
    def __call__(self, x, y):
        return self.k1(x, y) + self.k2(x, y)

    @property
    def stationary(self):
        return self.k1.stationary and self.k2.stationary

    @property
    def var(self):
        return self.k1.var + self.k2.var

    @property
    def length_scale(self):
        return (self.k1.var * self.k1.length_scale +
                self.k2.var * self.k2.length_scale) / self.var


class ProductKernel(Kernel, Referentiable):
    """Product of two kernels.

    Args:
        k1 (instance of :class:`.kernel.Kernel`): First kernel in product.
        k2 (instance of :class:`.kernel.Kernel`): Second kernel in product.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @dispatch(object, object)
    def __call__(self, x, y):
        return self.k1(x, y) * self.k2(x, y)

    @property
    def stationary(self):
        return self.k1.stationary and self.k2.stationary

    @property
    def var(self):
        return self.k1.var * self.k2.var

    @property
    def length_scale(self):
        return B.minimum(self.k1.length_scale, self.k2.length_scale)


class StretchedKernel(Kernel, Referentiable):
    """Stretched kernel.

    Args:
        k (instance of :class:`.kernel.Kernel`): Kernel to stretch.
        stretch (tensor): Stretch.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k, stretch):
        self.k = k
        self.stretch = stretch

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        return self.k(x / self.stretch, y / self.stretch)

    @property
    def stationary(self):
        return self.k.stationary

    @property
    def var(self):
        return self.k.var

    @property
    def length_scale(self):
        return self.k.length_scale * self.stretch

    @property
    def period(self):
        return self.k.period * self.stretch


class ScaledKernel(Kernel, Referentiable):
    """Scaled kernel.

    Args:
        k (instance of :class:`.kernel.Kernel`): Kernel to scale.
        scale (tensor): Scale.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k, scale):
        self.k = k
        self.scale = scale

    @dispatch(object, object)
    def __call__(self, x, y):
        return self.scale * self.k(x, y)

    @property
    def stationary(self):
        return self.k.stationary

    @property
    def var(self):
        return self.scale * self.k.var

    @property
    def length_scale(self):
        return self.k.length_scale

    @property
    def period(self):
        return self.k.period


class PeriodicKernel(Kernel, Referentiable):
    """Periodic kernel.

    Args:
        k (instance of :class:`.kernel.Kernel`): Kernel to make periodic.
        scale (tensor): Period.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k, period):
        self.k = k
        self._period = period

    @dispatch(Number, Number)
    def __call__(self, x, y):
        def feat_map(z):
            z = z * 2 * B.pi / self.period
            return B.array([[B.sin(z), B.cos(z)]])

        return self.k(feat_map(x), feat_map(y))

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        def feat_map(z):
            z = z * 2 * B.pi / self.period
            return B.concat((B.sin(z), B.cos(z)), axis=1)

        return self.k(feat_map(x), feat_map(y))

    @property
    def stationary(self):
        return self.k.stationary

    @property
    def var(self):
        return self.k.var

    @property
    def length_scale(self):
        return self.k.length_scale

    @property
    def period(self):
        return self._period


class ConstantKernel(Kernel, Referentiable):
    """Constant kernel of `1`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number, Number)
    def __call__(self, x, y):
        return B.array([[1.]])

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        return B.ones((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

    @property
    def length_scale(self):
        return np.inf


class ZeroKernel(Kernel, Referentiable):
    """Constant kernel of `0`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number, Number)
    def __call__(self, x, y):
        return B.array([[0.]])

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        return B.zeros((B.shape(x)[0], B.shape(y)[0]), dtype=B.dtype(x))

    @property
    def var(self):
        return 0

    @property
    def length_scale(self):
        return 0


class EQ(Kernel, Referentiable):
    """Exponentiated quadratic kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        return B.exp(-.5 * B.pw_dists2(x, y))


class RQ(Kernel, Referentiable):
    """Rational quadratic kernel.

    Args:
        alpha (positive float): Shape of the prior over length scales.
            determines the weight of the tails of the kernel.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, alpha):
        self.alpha = alpha

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        return (1 + .5 * B.pw_dists2(x, y) / self.alpha) ** (-self.alpha)


class Exp(Kernel, Referentiable):
    """Exponential kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        return B.exp(-B.pw_dists(x, y))


Matern12 = Exp  #: Alias for the exponential kernel.


class Matern32(Kernel, Referentiable):
    """Matern--3/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        r = 3 ** .5 * B.pw_dists(x, y)
        return (1 + r) * B.exp(-r)


class Matern52(Kernel, Referentiable):
    """Matern--5/2 kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        r1 = 5 ** .5 * B.pw_dists(x, y)
        r2 = 5 * B.pw_dists2(x, y) / 3
        return (1 + r1 + r2) * B.exp(-r1)


class Kronecker(Kernel, Referentiable):
    """Kronecker kernel.

    Args:
        epsilon (float, optional): Tolerance for equality in squared distance.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        dists2 = B.pw_dists2(x, y)
        return B.cast(dists2 < self.epsilon, B.dtype(x))

    @property
    def length_scale(self):
        return 0


class Linear(Kernel, Referentiable):
    """Linear kernel."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(B.Numeric, B.Numeric)
    def __call__(self, x, y):
        return B.matmul(x, y, tr_b=True)

    @property
    def stationary(self):
        return False

    @property
    def length_scale(self):
        return np.nan

    @property
    def var(self):
        return np.nan


class PosteriorCrossKernel(Kernel, Referentiable):
    """Posterior cross kernel.

    Args:
        k_ij (instance of :class:`.kernel.Kernel`): Kernel between processes
            corresponding to the left input and the right input respectively.
        k_zi (instance of :class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the left input respectively.
        k_zj (instance of :class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the right input respectively.
        z (matrix): Locations of data.
        Kz (instance of :class:`.spd.SPD`): Kernel matrix of data.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k_ij, k_zi, k_zj, z, Kz):
        self.k_ij = k_ij
        self.k_zi = k_zi
        self.k_zj = k_zj
        self.z = z
        self.Kz = Kz

    @dispatch(object, object)
    def __call__(self, x, y):
        return (self.k_ij(x, y) - self.Kz.quadratic_form(self.k_zi(self.z, x),
                                                         self.k_zj(self.z, y)))

    @property
    def stationary(self):
        return False

    @property
    def length_scale(self):
        return np.nan

    @property
    def var(self):
        return np.nan


class PosteriorKernel(PosteriorCrossKernel, Referentiable):
    """Posterior kernel.

    Args:
        gp (instance of :class:`.random.GP`): Prior GP.
        z (matrix): Locations of data.
        Kz (instance of :class:`.spd.SPD`): Kernel matrix of data.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, gp, z, Kz):
        PosteriorCrossKernel.__init__(
            self, gp.kernel, gp.kernel, gp.kernel, z, Kz
        )


class ReversedKernel(Kernel, Referentiable):
    """Kernel that evaluates with its arguments reversed.

    Args:
        k (instance of :class:`.kernel.Kernel`): Kernel to evaluate.
    """

    def __init__(self, k):
        self.k = k

    def __call__(self, *args):
        # Nothing has to be done if the kernel is stationary, or the underlying
        # kernel also has its arguments reversed.
        if self.stationary or type(self.k) == ReversedKernel:
            return self.k(*args)
        return B.transpose(self.k(*reversed(args)))

    @property
    def stationary(self):
        return self.k.stationary

    @property
    def var(self):
        return self.k.var

    @property
    def length_scale(self):
        return self.k.length_scale

    @property
    def period(self):
        return self.k.period
