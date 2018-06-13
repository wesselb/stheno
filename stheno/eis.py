# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
from plum import Dispatcher, Self, Referentiable

from .input import Observed, Latent, Component
from .kernel import Kernel, ZeroKernel
from .cache import cache, Cache

__all__ = ['NoisyKernel', 'ComponentKernel', 'AdditiveComponentKernel']

log = logging.getLogger(__name__)


class ComponentKernel(Kernel, Referentiable):
    """Kernel consisting of multiple components.

    Uses :class:`.input.Component`.

    Args:
        ks (matrix of instances :class:`.kernel.Kernel`): Kernel between the
            various components, indexed by type parameter.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, ks):
        self.ks = ks

    @dispatch(Component, Component, Cache)
    @cache
    def __call__(self, x, y, cache):
        return self.ks[type(x), type(y)](x.get(), y.get(), cache)

    def __str__(self):
        ks = self.ks.values()
        return 'ComponentKernel({})'.format(', '.join([str(k) for k in ks]))


class AdditiveComponentKernel(ComponentKernel, Referentiable):
    """Kernel consisting of additive, independent components.

    Uses :class:`.input.Component`, :class:`.input.Observed`, and
    :class:`.input.Latent`.

    Args:
        ks (dict of instances of :class:`.kernel.Kernel`): Kernels of the
            components.
        latent (list of instances of :class:`.input.Input`, optional): List of
            input types that make up the latent process.
    """

    def __init__(self, ks, latent=None):
        input_types = set(ks.keys())

        class KernelMatrix(object):
            def _resolve(self, t):
                if t == Latent:
                    return {} if latent is None else set(latent)
                elif t == Observed:
                    return input_types
                elif t in input_types:
                    return {t}
                else:
                    raise RuntimeError('Input type "{}" is not a component of '
                                       'the kernel.'.format(t.__name__))

            def __getitem__(self, item):
                t1s, t2s = self._resolve(item[0]), self._resolve(item[1])
                return sum([ks[t] for t in t1s & t2s], ZeroKernel())

        ComponentKernel.__init__(self, KernelMatrix())


class NoisyKernel(Kernel, Referentiable):
    """Noisy observations of a latent process.

    Uses :class:`.input.Latent` and :class:`.input.Observed`.

    Args:
        k_f (instance of :class:`.kernel.Kernel`): Kernel of the latent process.
        k_n (instance of :class:`.kernel.Kernel`): Kernel of the noise.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, k_f, k_n):
        self.k_f = k_f
        self.k_n = k_n
        self.k_y = k_f + k_n

    @dispatch({Latent, Observed}, {Latent, Observed}, Cache)
    @cache
    def __call__(self, x, y, cache):
        return self.k_f(x.get(), y.get())

    @dispatch(Observed, Observed, Cache)
    @cache
    def __call__(self, x, y, cache):
        return self.k_y(x.get(), y.get())

    def __str__(self):
        return 'NoisyKernel({}, {})'.format(self.k_f, self.k_n)
