# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from plum import Dispatcher, Self, Referentiable

from .cache import Cache, cache, uprank
from .field import Type, ZeroType, OneType, ScaledType, \
    ProductType, SumType, ShiftedType, SelectedType, InputTransformedType, \
    StretchedType, DerivativeType, FunctionType, apply_optional_arg
from .input import Input

__all__ = ['FunctionMean', 'DerivativeMean']

log = logging.getLogger(__name__)


class Mean(Type, Referentiable):
    """Mean function.

    Means can be added and multiplied.
    """

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    def __call__(self, x, cache):
        """Construct the mean for a design matrix.

        Args:
            x (input): Points to construct the mean at.
            cache (:class:`.cache.Cache`): Cache.

        Returns:
            tensor: Mean vector as a rank 2 column vector.
        """
        raise NotImplementedError()

    @_dispatch(object)
    def __call__(self, x):
        return self(x, Cache())

    @_dispatch(Input)
    def __call__(self, x):
        return self(x, Cache())


class SumMean(Mean, SumType, Referentiable):
    """Sum of two means."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return B.add(self[0](x, B), self[1](x, B))


class ProductMean(Mean, ProductType, Referentiable):
    """Product of two means."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return B.multiply(self[0](x, B), self[1](x, B))


class ScaledMean(Mean, ScaledType, Referentiable):
    """Scaled mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return B.multiply(self.scale, self[0](x, B))


class StretchedMean(Mean, StretchedType, Referentiable):
    """Stretched mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return self[0](B.divide(x, self.stretches[0]), B)


class ShiftedMean(Mean, ShiftedType, Referentiable):
    """Shifted mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return self[0](B.subtract(x, self.shifts[0]), B)


class SelectedMean(Mean, SelectedType, Referentiable):
    """Mean with particular input dimensions selected."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return self[0](B.take(x, self.dims[0], axis=1), B)


class InputTransformedMean(Mean, InputTransformedType, Referentiable):
    """Input-transformed mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        return self[0](apply_optional_arg(self.fs[0], x, B), B)


class OneMean(Mean, OneType, Referentiable):
    """Constant mean of `1`."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, B):
        return B.ones([B.shape(x)[0], 1], dtype=B.dtype(x))


class ZeroMean(Mean, ZeroType, Referentiable):
    """Constant mean of `0`."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, B):
        return B.zeros([B.shape(x)[0], 1], dtype=B.dtype(x))


class FunctionMean(Mean, FunctionType, Referentiable):
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, B):
        return apply_optional_arg(self.fs[0], x, B)


class DerivativeMean(Mean, DerivativeType, Referentiable):
    """Derivative of mean."""
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric, Cache)
    @cache
    @uprank
    def __call__(self, x, B):
        i = self.derivs[0]
        return B.gradients(self[0](x, B), x)[0][:, i:i + 1]


class PosteriorCrossMean(Mean, Referentiable):
    """Posterior mean.

    Args:
        m_i (:class:`.mean.Mean`): Mean of process corresponding to
            the input.
        m_z (:class:`.mean.Mean`): Mean of process corresponding to
            the data.
        k_zi (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the input respectively.
        z (input): Locations of data.
        Kz (:class:`.spd.SPD`): Kernel matrix of data.
        y (tensor): Observations to condition on.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, m_i, m_z, k_zi, z, Kz, y):
        self.m_i = m_i
        self.k_zi = k_zi
        self.m_z = m_z
        self.z = z
        self.Kz = Kz
        self.y = uprank(y)

    @_dispatch(object, Cache)
    @cache
    def __call__(self, x, B):
        diff = B.subtract(self.y, self.m_z(self.z, B))
        prod = self.Kz.inv_prod(self.k_zi(self.z, x))
        return B.add(self.m_i(x, B), B.matmul(prod, diff, tr_a=True))

    def __str__(self):
        return 'PosteriorCrossMean()'


class PosteriorMean(PosteriorCrossMean, Referentiable):
    """Posterior mean.

    Args:
        gp (:class:`.random.GP`): Corresponding GP.
        z (input): Locations of data.
        Kz (:class:`.spd.SPD`): Kernel matrix of data.
        y (tensor): Observations to condition on.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, gp, z, Kz, y):
        PosteriorCrossMean.__init__(self, gp.mean, gp.mean, gp.kernel, z, Kz, y)

    def __str__(self):
        return 'PosteriorMean()'
