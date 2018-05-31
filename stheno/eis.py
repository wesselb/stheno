# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import Dispatcher, Self, Referentiable, type_parameter

from stheno import Kernel, Input, Observed, Component, ConstantKernel, \
    ZeroKernel

__all__ = ['NoisyKernel', 'ComponentKernel', 'AdditiveComponentKernel']


class ComponentKernel(Kernel, Referentiable):
    """Kernel consisting of multiple components.

    Uses :class:`.input.Component`.

    Args:
        ks (matrix of class:`.kernel.Kernel`s): Kernel between the various
            components, indexed by type parameter.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, ks):
        self.ks = ks

    @dispatch(Component, Component)
    def __call__(self, x, y):
        return self.ks[type(x), type(y)](x.get(), y.get())


class AdditiveComponentKernel(ComponentKernel, Referentiable):
    """Kernel consisting of additive, independent components.

    Uses :class:`.input.Component` and :class:`Observed`.

    Args:
        ks (dict of class:`.kernel.Kernel`s): Kernels of the components.
    """

    def __init__(self, ks):
        class KernelMatrix(object):
            def __getitem__(self, indices):
                i1, i2 = indices
                if i1 == i2 and i1 != Observed:
                    return ks[i1]
                elif i1 == i2 and i1 == Observed:
                    return sum(ks.values(), ZeroKernel())
                elif i1 == Observed:
                    return ks[i2]
                elif i2 == Observed:
                    return ks[i1]
                else:
                    return ZeroKernel()

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
        self.k_y = k_n + k_f

    @dispatch(Input, Input)
    def __call__(self, x, y):
        return self.k_f(x.get(), y.get())

    @dispatch(Observed, Observed)
    def __call__(self, x, y):
        return self.k_y(x.get(), y.get())
