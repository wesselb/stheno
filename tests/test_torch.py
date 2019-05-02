# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from autograd import grad
from stheno.field import Element
from stheno.torch import EQ

import torch
import lab as B

# noinspection PyUnresolvedReferences
from . import eq, neq, ok, raises, benchmark, le, assert_instance


def test_boxing():
    k = EQ()
    num = B.randn(torch.float64)

    # Test addition and multiplication.
    yield assert_instance, num + k, Element
    yield assert_instance, num + num, B.Torch
    yield assert_instance, num * k, Element
    yield assert_instance, num * num, B.Torch

    # Test in-place addition.
    num = B.randn(torch.float64)
    num += B.randn(torch.float64)
    yield assert_instance, num, B.Torch
    num += k
    yield assert_instance, num, Element

    # Test in-place multiplication.
    num = B.randn(torch.float64)
    num *= B.randn(torch.float64)
    yield assert_instance, num, B.Torch
    num *= k
    yield assert_instance, num, Element
