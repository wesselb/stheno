# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from autograd import grad
from stheno.field import Element
from stheno import EQ

# noinspection PyUnresolvedReferences
from . import eq, neq, ok, raises, benchmark, le, assert_instance


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
        yield assert_instance, obj, Element
