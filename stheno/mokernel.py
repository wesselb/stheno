# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from plum import Dispatcher, Self, Referentiable, type_parameter

from .cache import cache, Cache
from .input import At, MultiInput
from .kernel import Kernel
from .matrix import dense, Dense

__all__ = ['MultiOutputKernel']

log = logging.getLogger(__name__)


class MultiOutputKernel(Kernel, Referentiable):
    """A generic multi-output kernel.

    Args:
        *ps (instance of :class:`.graph.GP`): Processes that make up the
            multi-valued process.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, *ps):
        self.kernels = ps[0].graph.kernels
        self.ps = ps

    @_dispatch(B.Numeric, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return self(MultiInput(*(At(p)(x) for p in self.ps)),
                    MultiInput(*(At(p)(y) for p in self.ps)), B)

    @_dispatch(At, B.Numeric, Cache)
    @cache
    def __call__(self, x, y, B):
        return self(MultiInput(x), MultiInput(*(At(p)(y) for p in self.ps)), B)

    @_dispatch(B.Numeric, At, Cache)
    @cache
    def __call__(self, x, y, B):
        return self(MultiInput(*(At(p)(x) for p in self.ps)), MultiInput(y), B)

    @_dispatch(At, At, Cache)
    @cache
    def __call__(self, x, y, B):
        return self.kernels[type_parameter(x),
                            type_parameter(y)](x.get(), y.get(), B)

    @_dispatch(MultiInput, At, Cache)
    @cache
    def __call__(self, x, y, B):
        return self(x, MultiInput(y), B)

    @_dispatch(At, MultiInput, Cache)
    @cache
    def __call__(self, x, y, B):
        return self(MultiInput(x), y, B)

    @_dispatch(MultiInput, MultiInput, Cache)
    @cache
    def __call__(self, x, y, B):
        return Dense(B.concat([B.concat([dense(self(xi, yi, B))
                                         for yi in y.get()], axis=1)
                               for xi in x.get()], axis=0))

    def __str__(self):
        ks = [str(self.kernels[p]) for p in self.ps]
        return 'MultiOutputKernel({})'.format(', '.join(ks))
