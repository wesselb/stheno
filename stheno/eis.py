# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import Dispatcher, Self, Referentiable

from stheno import Kernel, Input, Observed

__all__ = ['NoisyKernel']


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
