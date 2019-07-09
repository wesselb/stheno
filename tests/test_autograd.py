# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from autograd import grad

from stheno import EQ
from stheno.field import Element


def test_boxing():
    k = EQ()
    objs = []

    def objective(x):
        x = x * 2
        x = x + 2
        objs.append(x * k)
        objs.append(x + k)
        return x

    grad(objective)(1.0)

    for obj in objs:
        assert isinstance(obj, Element)
