# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from plum import Dispatcher, Self, Referentiable, type_parameter

from .input import At, MultiInput, Input
from .kernel import Kernel
from .matrix import dense, Dense, Zero, Diagonal, One

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

    @_dispatch({B.Numeric, Input}, {B.Numeric, Input})
    def __call__(self, x, y):
        return self(MultiInput(*(p(x) for p in self.ps)),
                    MultiInput(*(p(y) for p in self.ps)))

    @_dispatch(At, {B.Numeric, Input})
    def __call__(self, x, y):
        return self(MultiInput(x), MultiInput(*(p(y) for p in self.ps)))

    @_dispatch({B.Numeric, Input}, At)
    def __call__(self, x, y):
        return self(MultiInput(*(p(x) for p in self.ps)), MultiInput(y))

    @_dispatch(At, At)
    def __call__(self, x, y):
        return self.kernels[type_parameter(x),
                            type_parameter(y)](x.get(), y.get())

    @_dispatch(MultiInput, At)
    def __call__(self, x, y):
        return self(x, MultiInput(y))

    @_dispatch(At, MultiInput)
    def __call__(self, x, y):
        return self(MultiInput(x), y)

    @_dispatch(MultiInput, MultiInput)
    def __call__(self, x, y):
        return B.block_matrix(*[[self(xi, yi) for yi in y.get()]
                                for xi in x.get()])

    @_dispatch({B.Numeric, Input}, {B.Numeric, Input})
    def elwise(self, x, y):
        return self.elwise(MultiInput(*(p(x) for p in self.ps)),
                           MultiInput(*(p(y) for p in self.ps)))

    @_dispatch(At, {B.Numeric, Input})
    def elwise(self, x, y):
        raise ValueError('Unclear combination of arguments given to '
                         'MultiOutputKernel.elwise.')

    @_dispatch({B.Numeric, Input}, At)
    def elwise(self, x, y):
        raise ValueError('Unclear combination of arguments given to '
                         'MultiOutputKernel.elwise.')

    @_dispatch(At, At)
    def elwise(self, x, y):
        return self.kernels[type_parameter(x),
                            type_parameter(y)].elwise(x.get(), y.get())

    @_dispatch(MultiInput, At)
    def elwise(self, x, y):
        raise ValueError('Unclear combination of arguments given to '
                         'MultiOutputKernel.elwise.')

    @_dispatch(At, MultiInput)
    def elwise(self, x, y):
        raise ValueError('Unclear combination of arguments given to '
                         'MultiOutputKernel.elwise.')

    @_dispatch(MultiInput, MultiInput)
    def elwise(self, x, y):
        if len(x.get()) != len(y.get()):
            raise ValueError('MultiOutputKernel.elwise must be called with '
                             'similarly sized MultiInputs.')
        return B.concat(*[self.elwise(xi, yi)
                          for xi, yi in zip(x.get(), y.get())], axis=0)

    def __str__(self):
        ks = [str(self.kernels[p]) for p in self.ps]
        return 'MultiOutputKernel({})'.format(', '.join(ks))
