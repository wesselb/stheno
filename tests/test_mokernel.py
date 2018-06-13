# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_allclose

from stheno.mokernel import MultiOutputKernel
from stheno.graph import Graph, GP, At
from stheno.kernel import EQ
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint


def test_mokernel():
    m = Graph()
    p1 = GP(1 * EQ(), graph=m)
    p2 = GP(2 * EQ().stretch(2), graph=m)

    mok = MultiOutputKernel(p1, p2)
    ks = m.kernels

    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(1, 2, 5)

    yield eq, str(mok), 'MultiOutputKernel(EQ(), 2 * (EQ() > 2))'

    yield assert_allclose, mok(x1, x2), \
          np.concatenate([np.concatenate([ks[p1, p1](x1, x2),
                                          ks[p1, p2](x1, x2)],
                                         axis=1),
                          np.concatenate([ks[p2, p1](x1, x2),
                                          ks[p2, p2](x1, x2)],
                                         axis=1)],
                         axis=0)
    yield assert_allclose, mok(At(p1)(x1), At(p1)(x2)), ks[p1](x1, x2)
    yield assert_allclose, mok(At(p1)(x1), At(p2)(x2)), ks[p1, p2](x1, x2)
    yield assert_allclose, mok((At(p2)(x1), At(p1)(x2)), [At(p2)(x1)]), \
          np.concatenate([ks[p2, p2](x1, x1), ks[p1, p2](x2, x1)],
                         axis=0)
