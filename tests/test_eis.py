# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from stheno.eis import ComponentKernel, AdditiveComponentKernel, NoisyKernel
from stheno.input import Component, Latent, Observed
from stheno.kernel import EQ, RQ, ZeroKernel
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok, allclose


def test_component_kernel():
    x = np.random.randn(10, 2)
    k1 = EQ()
    k2 = RQ(1e-1)
    kzero = ZeroKernel()
    kc = ComponentKernel({
        (Component(1), Component(1)): k1,
        (Component(1), Component(2)): kzero,
        (Component(2), Component(1)): kzero,
        (Component(2), Component(2)): k2
    })

    yield ok, allclose(kc(Component(1)(x)), k1(x))
    yield ok, allclose(kc(Component(2)(x)), k2(x))
    yield ok, allclose(kc(Component(1)(x), Component(2)(x)), kzero(x))
    yield ok, allclose(kc(Component(2)(x), Component(1)(x)), kzero(x))

    yield eq, str(ComponentKernel({Component(1): EQ(),
                                   Component(2): EQ()})), \
          'ComponentKernel(EQ(), EQ())'


def test_compare_noisy_kernel_and_additive_component_kernel():
    def NoisyKernelCopy(k1, k2):
        return AdditiveComponentKernel({
            Component('wiggly'): k1,
            Component('noise'): k2
        }, latent=[Component('wiggly')])

    x = np.random.randn(10, 2)
    k1 = EQ()
    k2 = RQ(1e-2)
    kn1 = NoisyKernel(k1, k2)
    kn2 = NoisyKernelCopy(k1, k2)

    yield ok, allclose(kn1(Latent(x)), k1(x)), 'noisy kernel 1'
    yield ok, allclose(kn1(Latent(x), Observed(x)), k1(x)), 'noisy kernel 2'
    yield ok, allclose(kn1(Observed(x), Latent(x)), k1(x)), 'noisy kernel 3'
    yield ok, allclose(kn1(Observed(x)), (k1 + k2)(x)), 'noisy kernel 4'

    yield ok, allclose(kn2(Latent(x)), k1(x)), 'noisy copy 1'
    yield ok, allclose(kn2(Latent(x), Observed(x)), k1(x)), 'noisy copy 2'
    yield ok, allclose(kn2(Observed(x), Latent(x)), k1(x)), 'noisy copy 3'
    yield ok, allclose(kn2(Observed(x)), (k1 + k2)(x)), 'noisy copy 4'

    yield ok, allclose(kn2(Latent(x), Component('noise')(x)),
                          ZeroKernel()(x)), 'additive 1'
    yield ok, allclose(kn2(Component('noise')(x), Latent(x)),
                          ZeroKernel()(x)), 'additive 2'
    yield ok, allclose(kn2(Observed(x), Component('noise')(x)),
                          k2(x)), 'additive 3'
    yield ok, allclose(kn2(Component('noise')(x), Observed(x)),
                          k2(x)), 'additive 4'

    yield eq, str(NoisyKernel(EQ(), RQ(1))), \
          'NoisyKernel(EQ(), RQ(1))'


def test_additive_component_kernel():
    k = AdditiveComponentKernel({
        Component('wiggly'): EQ(),
        Component('noise'): EQ()
    })

    # Test independence components.
    x = np.random.randn(10, 2)
    yield ok, allclose(k(Component('wiggly')(x), Component('noise')(x)),
                          np.zeros((10, 10)))
    yield ok, allclose(k(Component('noise')(x), Component('wiggly')(x)),
                          np.zeros((10, 10)))

    # Test check.
    yield raises, RuntimeError, lambda: k(Component('erroneous')(x))
