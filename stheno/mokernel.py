# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from plum import Dispatcher, Self, Referentiable, type_parameter
from lab import B

from .cache import cache, Cache
from .input import Component
from .kernel import Kernel
from .graph import GP, At

__all__ = ['MultiOutputKernel']

log = logging.getLogger(__name__)


class MultiOutputKernel(Kernel, Referentiable):
    """A generic multi-output kernel.

    Args:
        \*ps (instance of :class:`.graph.GP`): Processes that make up the
            multi-valued process.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, *ps):
        self.kernels = ps[0].graph.kernels
        self.ps = ps

    @dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.concat([B.concat([self(At(p_row)(x), At(p_col)(y), B)
                                   for p_col in self.ps], axis=1)
                         for p_row in self.ps], axis=0)

    @dispatch(At, At, Cache)
    @cache
    def __call__(self, x, y, B):
        return self.kernels[type_parameter(x),
                            type_parameter(y)](x.get(), y.get(), B)

    @dispatch({tuple, list}, {tuple, list}, Cache)
    @cache
    def __call__(self, x, y, B):
        return B.concat([B.concat([self(xi, yi, B) for yi in y], axis=1)
                         for xi in x], axis=0)

    def __str__(self):
        ks = [str(self.kernels[p]) for p in self.ps]
        return 'MultiOutputKernel({})'.format(', '.join(ks))
