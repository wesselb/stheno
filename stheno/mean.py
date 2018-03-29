# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B
from plum import Dispatcher, Self, Referentiable

__all__ = ['Mean', 'ConstantMean', 'PosteriorMean', 'ZeroMean']


class Mean(Referentiable):
    """Mean function.

    TODO: Means can be added and multiplied.
    TODO: Multi-output means?

    Args:
        f (function): Function that implements the mean. It should take in
            one design matrices and return the resulting mean vector.
    """

    dispatch = Dispatcher(in_class=Self)

    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        """Construct the mean for a design matrix.

        Args:
            x (design matrix): Points to construct the mean at.

        Returns:
            Mean vector.
        """
        if B.rank(x) != 2:
            raise ValueError('Argument must have rank 2.')
        return self.f(x)

    @dispatch(Self)
    def __add__(self, other):
        return Mean(lambda x: self(x) + other(x))

    @dispatch(object)
    def __add__(self, other):
        return Mean(lambda x: self(x) + other)

    def __radd__(self, other):
        return self + other

    @dispatch(Self)
    def __mul__(self, other):
        return Mean(lambda x: self(x) * other(x))

    @dispatch(object)
    def __mul__(self, other):
        return Mean(lambda x: self(x) * other)

    def __rmul__(self, other):
        return self * other


class ConstantMean(Mean):
    """A constant mean of `1`."""

    def __init__(self):
        def f(x):
            n = B.shape(x)[1]
            return B.ones((n, 1), dtype=x.dtype)

        self.f = f


class ZeroMean(Mean):
    """A constant mean of `0`."""

    def __init__(self):
        def f(x):
            n = B.shape(x)[1]
            return B.zeros((n, 1), dtype=x.dtype)

        self.f = f


class PosteriorMean(Mean):
    def __init__(self, gp, z, Kz, y):
        def f(x):
            return gp.mean(x) + B.dot(Kz.inv_prod(gp.kernel(z, x)),
                                       y - gp.mean(z), tr_a=True)

        self.f = f
