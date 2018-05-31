# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import Dispatcher, Self, Referentiable, type_parameter

from stheno import Kernel, Input, Observed, Component, ConstantKernel, \
    ZeroKernel, Latent

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

    @dispatch(Component, Component)
    def __call__(self, x, y):
        return self.ks[type(x), type(y)](x.get(), y.get())


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
        latent = [] if latent is None else latent

        class KernelMatrix(object):
            def __getitem__(self, indices):
                i1, i2 = indices

                # Handle latent.
                if i1 == Latent or i2 == Latent:
                    if i1 == i2 or i1 == Observed or i2 == Observed:
                        return sum([ks[i] for i in latent], ZeroKernel())
                    else:
                        if i1 == Latent:
                            return ks[i2] if i2 in latent else ZeroKernel()
                        else:
                            return ks[i1] if i1 in latent else ZeroKernel()

                # Handle observed, assuming there is no latent.
                if i1 == Observed or i2 == Observed:
                    if i1 == i2:
                        return sum(ks.values(), ZeroKernel())
                    else:
                        return ks[i1] if i2 == Observed else ks[i2]

                # Standard implementation.
                if i1 == i2:
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
