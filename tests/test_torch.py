# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab.torch as B
import torch

from stheno.field import Element
from stheno.torch import EQ


def test_boxing():
    k = EQ()
    num = B.randn(torch.float64)

    # Test addition and multiplication.
    assert isinstance(num + k, Element)
    assert isinstance(num + num, B.Torch)
    assert isinstance(num * k, Element)
    assert isinstance(num * num, B.Torch)

    # Test in-place addition.
    num = B.randn(torch.float64)
    num += B.randn(torch.float64)
    assert isinstance(num, B.Torch)
    num += k
    assert isinstance(num, Element)

    # Test in-place multiplication.
    num = B.randn(torch.float64)
    num *= B.randn(torch.float64)
    assert isinstance(num, B.Torch)
    num *= k
    assert isinstance(num, Element)
