# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from stheno.graph import Graph, GP
from stheno.input import MultiInput
from stheno.kernel import EQ
from stheno.matrix import dense
from stheno.mokernel import MultiOutputKernel

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint, \
    assert_allclose, lam


def test_mokernel():
    m = Graph()
    p1 = GP(1 * EQ(), graph=m)
    p2 = GP(2 * EQ().stretch(2), graph=m)

    mok = MultiOutputKernel(p1, p2)
    ks = m.kernels

    x1 = np.linspace(0, 1, 10)
    x2 = np.linspace(1, 2, 5)
    x3 = np.linspace(1, 2, 10)

    yield eq, str(mok), 'MultiOutputKernel(EQ(), 2 * (EQ() > 2))'

    # `B.Numeric` versus `B.Numeric`:
    yield assert_allclose, mok(x1, x2), \
          np.concatenate([np.concatenate([dense(ks[p1, p1](x1, x2)),
                                          dense(ks[p1, p2](x1, x2))],
                                         axis=1),
                          np.concatenate([dense(ks[p2, p1](x1, x2)),
                                          dense(ks[p2, p2](x1, x2))],
                                         axis=1)], axis=0)
    yield assert_allclose, mok.elwise(x1, x3), \
          np.concatenate([ks[p1, p1].elwise(x1, x3),
                          ks[p2, p2].elwise(x1, x3)], axis=0)

    # `B.Numeric` versus `At`:
    yield assert_allclose, mok(p1(x1), x2), \
          np.concatenate([dense(ks[p1, p1](x1, x2)),
                          dense(ks[p1, p2](x1, x2))], axis=1)
    yield assert_allclose, mok(p2(x1), x2), \
          np.concatenate([dense(ks[p2, p1](x1, x2)),
                          dense(ks[p2, p2](x1, x2))], axis=1)
    yield assert_allclose, mok(x1, p1(x2)), \
          np.concatenate([dense(ks[p1, p1](x1, x2)),
                          dense(ks[p2, p1](x1, x2))], axis=0)
    yield assert_allclose, mok(x1, p2(x2)), \
          np.concatenate([dense(ks[p1, p2](x1, x2)),
                          dense(ks[p2, p2](x1, x2))], axis=0)
    yield raises, ValueError, lambda: mok.elwise(x1, p2(x3))
    yield raises, ValueError, lambda: mok.elwise(p1(x1), x3)

    # `At` versus `At`:
    yield assert_allclose, mok(p1(x1), p1(x2)), ks[p1](x1, x2)
    yield assert_allclose, mok(p1(x1), p2(x2)), ks[p1, p2](x1, x2)
    yield assert_allclose, mok.elwise(p1(x1), p1(x3)), ks[p1].elwise(x1, x3)
    yield assert_allclose, mok.elwise(p1(x1), p2(x3)), ks[p1, p2].elwise(x1, x3)

    # `MultiInput` versus `MultiInput`:
    yield assert_allclose, mok(MultiInput(p2(x1), p1(x2)),
                               MultiInput(p2(x1))), \
          np.concatenate([dense(ks[p2, p2](x1, x1)),
                          dense(ks[p1, p2](x2, x1))], axis=0)
    yield raises, ValueError, \
          lambda: mok.elwise(MultiInput(p2(x1), p1(x3)), MultiInput(p2(x1)))
    yield assert_allclose, mok.elwise(MultiInput(p2(x1), p1(x3)),
                                      MultiInput(p2(x1), p1(x3))), \
          np.concatenate([ks[p2, p2].elwise(x1, x1),
                          ks[p1, p1].elwise(x3, x3)], axis=0)

    # `MultiInput` versus `At`:
    yield assert_allclose, mok(MultiInput(p2(x1), p1(x2)),
                               p2(x1)), \
          np.concatenate([dense(ks[p2, p2](x1, x1)),
                          dense(ks[p1, p2](x2, x1))], axis=0)
    yield assert_allclose, mok(p2(x1),
                               MultiInput(p2(x1), p1(x2))), \
          np.concatenate([dense(ks[p2, p2](x1, x1)),
                          dense(ks[p2, p1](x1, x2))], axis=1)
    yield raises, ValueError, \
          lambda: mok.elwise(MultiInput(p2(x1), p1(x3)), p2(x1))
    yield raises, ValueError, \
          lambda: mok.elwise(p2(x1), MultiInput(p2(x1), p1(x3)))
