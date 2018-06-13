# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from plum import Dispatcher, Self, Referentiable, type_parameter

from .cache import cache, Cache
from .input import At
from .mean import Mean

__all__ = []

log = logging.getLogger(__name__)


class MultiOutputMean(Mean, Referentiable):
    """A generic multi-output mean.

    Args:
        \*ps (instance of :class:`.graph.GP`): Processes that make up the
            multi-valued process.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, *ps):
        self.means = ps[0].graph.means
        self.ps = ps

    @dispatch(B.Numeric, Cache)
    @cache
    def __call__(self, x, B):
        return B.concat([self.means[p](x, B) for p in self.ps], axis=0)

    @dispatch(At, Cache)
    @cache
    def __call__(self, x, B):
        return self.means[type_parameter(x)](x.get(), B)

    @dispatch({tuple, list}, Cache)
    @cache
    def __call__(self, x, B):
        return B.concat([self(xi, B) for xi in x], axis=0)

    def __str__(self):
        ks = [str(self.means[p]) for p in self.ps]
        return 'MultiOutputMean({})'.format(', '.join(ks))
