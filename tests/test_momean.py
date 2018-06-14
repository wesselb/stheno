# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_allclose

from stheno.graph import Graph, GP
from stheno.input import At, MultiInput
from stheno.kernel import EQ
from stheno.momean import MultiOutputMean
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint


def test_momean():
    m = Graph()
    p1 = GP(1 * EQ(), lambda x: 2 * x, graph=m)
    p2 = GP(2 * EQ().stretch(2), 1, graph=m)

    mom = MultiOutputMean(p1, p2)
    ms = m.means

    x = np.linspace(0, 1, 10)

    yield eq, str(mom), 'MultiOutputMean(<lambda>, 1)'

    yield assert_allclose, mom(x), \
          np.concatenate([ms[p1](x), ms[p2](x)], axis=0)
    yield assert_allclose, mom(At(p1)(x)), ms[p1](x)
    yield assert_allclose, mom(At(p2)(x)), ms[p2](x)
    yield assert_allclose, mom(MultiInput(At(p2)(x), At(p1)(x))), \
          np.concatenate([ms[p2](x), ms[p1](x)], axis=0)
