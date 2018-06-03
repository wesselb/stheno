# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number
from types import FunctionType

from lab import B
from plum import Dispatcher, Self, Referentiable

from .input import Input

__all__ = ['Mean', 'SumMean', 'ProductMean', 'ScaledMean', 'ConstantMean',
           'PosteriorMean', 'FunctionMean', 'ZeroMean', 'PosteriorCrossMean']


class Mean(Referentiable):
    """Mean function.

    Means can be added and multiplied.
    """

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object)
    def __call__(self, x):
        """Construct the mean for a design matrix.

        Args:
            x (design matrix): Points to construct the mean at.

        Returns:
            Mean vector.
        """
        raise NotImplementedError()

    @dispatch(Input)
    def __call__(self, x):
        return self(x.get())

    @dispatch(FunctionType)
    def __add__(self, other):
        return self + FunctionMean(other)

    @dispatch(Self)
    def __add__(self, other):
        return SumMean(self, other)

    @dispatch(object)
    def __add__(self, other):
        return SumMean(self, other * ConstantMean())

    def __radd__(self, other):
        return self + other

    @dispatch(Self)
    def __mul__(self, other):
        return ProductMean(self, other)

    @dispatch(object)
    def __mul__(self, other):
        return ScaledMean(self, other)

    def __rmul__(self, other):
        return self * other


class SumMean(Mean, Referentiable):
    """Sum of two means.

    Args:
        m1 (instance of :class:`.mean.Mean`): First mean in sum.
        m2 (instance of :class:`.mean.Mean`): Second mean in sum.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    @dispatch(object)
    def __call__(self, x):
        return self.m1(x) + self.m2(x)


class ProductMean(Mean, Referentiable):
    """Product of two means.

    Args:
        m1 (instance of :class:`.mean.Mean`): First mean in product.
        m2 (instance of :class:`.mean.Mean`): Second mean in product.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    @dispatch(object)
    def __call__(self, x):
        return self.m1(x) * self.m2(x)


class ScaledMean(Mean, Referentiable):
    """Scaled mean.

    Args:
        m (instance of :class:`.kernel.Mean`): Mean to scale.
        scale (tensor): Scale.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, m, scale):
        self.m = m
        self.scale = scale

    @dispatch(object)
    def __call__(self, x):
        return self.scale * self.m(x)


class ConstantMean(Mean, Referentiable):
    """Constant mean of `1`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number)
    def __call__(self, x):
        return B.array([[1.]])

    @dispatch(B.Numeric)
    def __call__(self, x):
        return B.ones((B.shape(x)[0], 1), dtype=B.dtype(x))


class ZeroMean(Mean, Referentiable):
    """Constant mean of `0`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number)
    def __call__(self, x):
        return B.array([[1.]])

    @dispatch(B.Numeric)
    def __call__(self, x):
        return B.zeros((B.shape(x)[0], 1), dtype=B.dtype(x))


class FunctionMean(Mean, Referentiable):
    """Mean parametrised by a function.

    Args:
        f (function): Function.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, f):
        self.f = f

    @dispatch(B.Numeric)
    def __call__(self, x):
        return self.f(x)


class PosteriorCrossMean(Mean, Referentiable):
    """Posterior mean.

    Args:
        m_i (instance of :class:`.mean.Mean`): Mean of process corresponding to
            the input.
        m_z (instance of :class:`.mean.Mean`): Mean of process corresponding to
            the data.
        k_zi (instance of :class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the input respectively.
        z (matrix): Locations of data.
        Kz (instance of :class:`.spd.SPD`): Kernel matrix of data.
        y (matrix): Observations to condition on.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, m_i, m_z, k_zi, z, Kz, y):
        self.m_i = m_i
        self.k_zi = k_zi
        self.m_z = m_z
        self._m_z_z = None
        self.z = z
        self.Kz = Kz
        self.y = y

    @property
    def m_z_z(self):
        if self._m_z_z is None:
            self._m_z_z = self.m_z(self.z)
        return self._m_z_z

    @dispatch(object)
    def __call__(self, x):
        return self.m_i(x) + \
               B.matmul(self.Kz.inv_prod(self.k_zi(self.z, x)),
                        self.y - self.m_z_z, tr_a=True)


class PosteriorMean(PosteriorCrossMean, Referentiable):
    """Posterior mean.

    Args:
        gp (instance of :class:`.random.GP`): Corresponding GP.
        z (matrix): Locations of data.
        Kz (instance of :class:`.spd.SPD`): Kernel matrix of data.
        y (matrix): Observations to condition on.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, gp, z, Kz, y):
        PosteriorCrossMean.__init__(self, gp.mean, gp.mean, gp.kernel, z, Kz, y)
