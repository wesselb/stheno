# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from stheno.graph import Graph, GP
from stheno.input import MultiInput
from stheno.kernel import EQ
from stheno.momean import MultiOutputMean
from .util import allclose


def test_momean():
    m = Graph()
    p1 = GP(1 * EQ(), lambda x: 2 * x, graph=m)
    p2 = GP(2 * EQ().stretch(2), 1, graph=m)

    mom = MultiOutputMean(p1, p2)
    ms = m.means

    x = np.linspace(0, 1, 10)

    assert str(mom) == 'MultiOutputMean(<lambda>, 1)'

    allclose(mom(x), np.concatenate([ms[p1](x), ms[p2](x)], axis=0))
    allclose(mom(p1(x)), ms[p1](x))
    allclose(mom(p2(x)), ms[p2](x))
    allclose(mom(MultiInput(p2(x), p1(x))),
             np.concatenate([ms[p2](x), ms[p1](x)], axis=0))
