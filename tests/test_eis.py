# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from stheno import EQ, RQ, Component, ComponentKernel, AdditiveComponentKernel, \
    NoisyKernel, ZeroKernel, Latent, Observed
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint


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

    yield ok, np.allclose(kc(Component(1)(x)), k1(x))
    yield ok, np.allclose(kc(Component(2)(x)), k2(x))
    yield ok, np.allclose(kc(Component(1)(x), Component(2)(x)), kzero(x))
    yield ok, np.allclose(kc(Component(2)(x), Component(1)(x)), kzero(x))


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

    yield ok, np.allclose(kn1(Latent(x)), k1(x)), 'noisy kernel 1'
    yield ok, np.allclose(kn1(Latent(x), Observed(x)), k1(x)), 'noisy kernel 2'
    yield ok, np.allclose(kn1(Observed(x), Latent(x)), k1(x)), 'noisy kernel 3'
    yield ok, np.allclose(kn1(Observed(x)), (k1 + k2)(x)), 'noisy kernel 4'

    yield ok, np.allclose(kn2(Latent(x)), k1(x)), 'noisy copy 1'
    yield ok, np.allclose(kn2(Latent(x), Observed(x)), k1(x)), 'noisy copy 2'
    yield ok, np.allclose(kn2(Observed(x), Latent(x)), k1(x)), 'noisy copy 3'
    yield ok, np.allclose(kn2(Observed(x)), (k1 + k2)(x)), 'noisy copy 4'

    yield ok, np.allclose(kn2(Latent(x), Component('noise')(x)),
                          ZeroKernel()(x)), 'additive 1'
    yield ok, np.allclose(kn2(Component('noise')(x), Latent(x)),
                          ZeroKernel()(x)), 'additive 2'
    yield ok, np.allclose(kn2(Observed(x), Component('noise')(x)),
                          k2(x)), 'additive 3'
    yield ok, np.allclose(kn2(Component('noise')(x), Observed(x)),
                          k2(x)), 'additive 4'
