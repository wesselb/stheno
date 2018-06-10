# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number
from types import FunctionType

from lab import B
from plum import Dispatcher, Self, Referentiable

from .field import add, dispatch, Type, ZeroType, OneType, ScaledType, \
    ProductType, SumType, ShiftedType, SelectedType, InputTransformedType, \
    StretchedType
from .input import Input
from .cache import Cache, cache

__all__ = ['FunctionMean']


class Mean(Type, Referentiable):
    """Mean function.

    Means can be added and multiplied.
    """

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, Cache)
    def __call__(self, x, cache):
        """Construct the mean for a design matrix.

        Args:
            x (design matrix): Points to construct the mean at.
            cache (instance of :class:`.cache.Cache`): Cache.

        Returns:
            Mean vector.
        """
        raise NotImplementedError()

    @dispatch(object)
    def __call__(self, x):
        return self(x, Cache())

    @dispatch(Input)
    def __call__(self, x):
        return self(x, Cache())

    @dispatch(Input, Cache)
    def __call__(self, x, cache):
        # This should not have been reached. Attempt to unwrap.
        return self(x.get(), cache)


class SumMean(Mean, SumType, Referentiable):
    """Sum of two means."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, Cache)
    @cache
    def __call__(self, x, cache):
        return self[0](x, cache) + self[1](x, cache)


class ProductMean(Mean, ProductType, Referentiable):
    """Product of two means."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, Cache)
    @cache
    def __call__(self, x, cache):
        return self[0](x, cache) * self[1](x, cache)


class ScaledMean(Mean, ScaledType, Referentiable):
    """Scaled mean."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, Cache)
    @cache
    def __call__(self, x, cache):
        return self.scale * self[0](x, cache)


class StretchedMean(Mean, StretchedType, Referentiable):
    """Stretched mean."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, Cache)
    @cache
    def __call__(self, x, cache):
        return self[0](x / self.extent, cache)


class ShiftedMean(Mean, ShiftedType, Referentiable):
    """Shifted mean."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, Cache)
    @cache
    def __call__(self, x, cache):
        return self[0](x - self.amount, cache)


class SelectedMean(Mean, SelectedType, Referentiable):
    """Mean with particular input dimensions selected."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, Cache)
    @cache
    def __call__(self, x, cache):
        return self[0](B.take(x, self.dims, axis=1), cache)


class InputTransformedMean(Mean, InputTransformedType, Referentiable):
    """Input-transformed mean."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(object, Cache)
    @cache
    def __call__(self, x, cache):
        return self[0](self.fs[0](x), cache)


class OneMean(Mean, OneType, Referentiable):
    """Constant mean of `1`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number, Cache)
    @cache
    def __call__(self, x, cache):
        return cache.ones([1, 1], dtype=B.dtype(x))

    @dispatch(B.Numeric, Cache)
    @cache
    def __call__(self, x, cache):
        return cache.ones([B.shape(x)[0], 1], dtype=B.dtype(x))


class ZeroMean(Mean, ZeroType, Referentiable):
    """Constant mean of `0`."""

    dispatch = Dispatcher(in_class=Self)

    @dispatch(Number, Cache)
    @cache
    def __call__(self, x, cache):
        return cache.zeros([1, 1], dtype=B.dtype(x))

    @dispatch(B.Numeric, Cache)
    @cache
    def __call__(self, x, cache):
        return cache.zeros([B.shape(x)[0], 1], dtype=B.dtype(x))


class FunctionMean(Mean, Referentiable):
    """Mean parametrised by a function.

    Args:
        f (function): Function.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, f):
        self.f = f

    @dispatch(B.Numeric, Cache)
    @cache
    def __call__(self, x, cache):
        return self.f(x)

    def __str__(self):
        return self.f.__name__


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

    def m_z_z(self, cache):
        if self._m_z_z is None:
            self._m_z_z = self.m_z(self.z, cache)
        return self._m_z_z

    @dispatch(object, Cache)
    @cache
    def __call__(self, x, cache):
        return self.m_i(x, cache) + \
               B.matmul(self.Kz.inv_prod(self.k_zi(self.z, x, cache)),
                        self.y - self.m_z_z(cache), tr_a=True)

    def __str__(self):
        return 'PosteriorCrossMean()'


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

    def __str__(self):
        return 'PosteriorMean()'


# Add functions to means.

@dispatch(FunctionType, Mean)
def add(a, b): return add(FunctionMean(a), b)


@dispatch(FunctionType, ZeroMean)
def add(a, b): return a


@dispatch(Mean, FunctionType)
def add(a, b): return add(a, FunctionMean(b))


@dispatch(ZeroMean, FunctionType)
def add(a, b): return b
