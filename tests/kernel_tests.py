# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call

from stheno import EQ, Kernel, StationaryKernel


def test_dispatch():
    sk = EQ()
    k = Kernel(lambda x, y: None)
    operators = [lambda x, y: x + y,
                 lambda x, y: x * y]

    for op in operators:
        yield eq, type(op(sk, k)), Kernel
        yield eq, type(op(k, sk)), Kernel
        yield eq, type(op(sk, sk)), StationaryKernel
        yield eq, type(op(k, k)), Kernel
