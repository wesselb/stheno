# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import tensorflow as tf
from lab import B
from plum import Dispatcher, Self, Referentiable

from .field import get_field
from .function_field import (
    StretchedFunction,
    ShiftedFunction,
    SelectedFunction,
    InputTransformedFunction,
    DerivativeFunction,
    TensorProductFunction,
    ZeroFunction,
    OneFunction,
    ScaledFunction,
    ProductFunction,
    SumFunction,
    Function
)
from .input import Input
from .matrix import dense
from .util import uprank

__all__ = ['TensorProductMean', 'DerivativeMean']

log = logging.getLogger(__name__)


class Mean(Function, Referentiable):
    """Mean function.

    Means can be added and multiplied.
    """

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object)
    def __call__(self, x):
        """Construct the mean for a design matrix.

        Args:
            x (input): Points to construct the mean at.

        Returns:
            tensor: Mean vector as a rank 2 column vector.
        """
        raise NotImplementedError()

    @_dispatch(Input)
    def __call__(self, x):
        # This should not have been reached. Attempt to unwrap.
        return self(x.get())


# Register the field.
@get_field.extend(Mean)
def get_field(a): return Mean


class SumMean(Mean, SumFunction, Referentiable):
    """Sum of two means."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object)
    def __call__(self, x):
        return B.add(self[0](x), self[1](x))


class ProductMean(Mean, ProductFunction, Referentiable):
    """Product of two means."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object)
    def __call__(self, x):
        return B.multiply(self[0](x), self[1](x))


class ScaledMean(Mean, ScaledFunction, Referentiable):
    """Scaled mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object)
    def __call__(self, x):
        return B.multiply(self.scale, self[0](x))


class StretchedMean(Mean, StretchedFunction, Referentiable):
    """Stretched mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object)
    @uprank
    def __call__(self, x):
        return self[0](B.divide(x, self.stretches[0]))


class ShiftedMean(Mean, ShiftedFunction, Referentiable):
    """Shifted mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object)
    @uprank
    def __call__(self, x):
        return self[0](B.subtract(x, self.shifts[0]))


class SelectedMean(Mean, SelectedFunction, Referentiable):
    """Mean with particular input dimensions selected."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object)
    @uprank
    def __call__(self, x):
        return self[0](B.take(x, self.dims[0], axis=1))


class InputTransformedMean(Mean, InputTransformedFunction, Referentiable):
    """Input-transformed mean."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(object)
    def __call__(self, x):
        return self[0](uprank(self.fs[0](x)))


class OneMean(Mean, OneFunction, Referentiable):
    """Constant mean of `1`."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric)
    @uprank
    def __call__(self, x):
        return B.ones(B.dtype(x), B.shape(x)[0], 1)


class ZeroMean(Mean, ZeroFunction, Referentiable):
    """Constant mean of `0`."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric)
    @uprank
    def __call__(self, x):
        return B.zeros(B.dtype(x), B.shape(x)[0], 1)


class TensorProductMean(Mean, TensorProductFunction, Referentiable):
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric)
    @uprank
    def __call__(self, x):
        return uprank(self.fs[0](x))


class DerivativeMean(Mean, DerivativeFunction, Referentiable):
    """Derivative of mean."""
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(B.Numeric)
    @uprank
    def __call__(self, x):
        i = self.derivs[0]
        with tf.GradientTape() as t:
            xi = x[:, i:i + 1]
            t.watch(xi)
            x = B.concat(x[:, :i], xi, x[:, i + 1:], axis=1)
            out = dense(self[0](x))
            return t.gradient(out, xi, unconnected_gradients='zero')


class PosteriorMean(Mean, Referentiable):
    """Posterior mean.

    Args:
        m_i (:class:`.mean.Mean`): Mean of process corresponding to
            the input.
        m_z (:class:`.mean.Mean`): Mean of process corresponding to
            the data.
        k_zi (:class:`.kernel.Kernel`): Kernel between processes
            corresponding to the data and the input respectively.
        z (input): Locations of data.
        K_z (:class:`.matrix.Dense`): Kernel matrix of data.
        y (tensor): Observations to condition on.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, m_i, m_z, k_zi, z, K_z, y):
        self.m_i = m_i
        self.m_z = m_z
        self.k_zi = k_zi
        self.z = z
        self.K_z = K_z
        self.y = uprank(y)

    @_dispatch(object)
    def __call__(self, x):
        diff = B.subtract(self.y, self.m_z(self.z))
        return B.add(self.m_i(x),
                     B.qf(self.K_z, self.k_zi(self.z, x), diff))
