# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import Dispatcher, Self, Referentiable

from .input import Observed, Latent, Component
from .kernel import Kernel, ZeroKernel
from .cache import cache, Cache

__all__ = ['NoisyKernel', 'ComponentKernel', 'AdditiveComponentKernel']


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
        InputTypes = {Observed, Latent} | set(ks.keys())

        class KernelMatrix(Referentiable):
            def __getitem__(self, item):
                i1, i2 = item

                # Check that both inputs are actually part of the kernel.
                if i1 not in InputTypes or i2 not in InputTypes:
                    raise RuntimeError(
                        'Either "{}" or "{}" is not a component of the kernel.'
                        ''.format(i1.__name__, i2.__name__))

                # TODO: Refactor logic here. Make more readable.
                # Handle latent.
                if latent and (i1 == Latent or i2 == Latent):
                    if i1 == i2 or i1 == Observed or i2 == Observed:
                        return sum([ks[i] for i in latent], ZeroKernel())
                    else:
                        i = i1 if i1 == Latent else i2
                        return ks[i] if i in latent else ZeroKernel()

                # Handle observed, assuming there is no latent.
                if i1 == Observed or i2 == Observed:
                    if i1 == i2:
                        return sum(ks.values(), ZeroKernel())
                    else:
                        return ks[i1] if i2 == Observed else ks[i2]

                # The rest is independent. Easy.
                return ks[i1] if i1 == i2 else ZeroKernel()

        ComponentKernel.__init__(self, KernelMatrix())

    # TODO: implement properly


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
