# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import pytest
import lab as B
import tensorflow as tf
import torch
from autograd import grad
from lab import torch as B

from stheno.field import Element
from stheno.torch import EQ


def test_boxing_autograd():
    k = EQ()
    objs = []

    def objective(x):
        x = x + 2
        x = x * 2
        objs.append(x + k)
        objs.append(x + k(x))
        objs.append(x * k)
        objs.append(x * k(x))
        return B.sum(x)

    grad(objective)(B.randn(10))

    for obj in objs:
        assert isinstance(obj, Element)


@pytest.mark.parametrize('dtype, Numeric',
                         [(torch.float64, B.TorchNumeric),
                          (tf.float64, B.TFNumeric)])
def test_boxing(dtype, Numeric):
    k = EQ()
    num = B.randn(dtype)
    x = B.randn(dtype, 10)

    # Test addition and multiplication.
    assert isinstance(num + k, Element)
    assert isinstance(num + k(x), Element)
    assert isinstance(num + num, Numeric)
    assert isinstance(num * k, Element)
    assert isinstance(num * k(x), Element)
    assert isinstance(num * num, Numeric)

    # Test in-place addition.
    num = B.randn(dtype)
    num += B.randn(dtype)
    assert isinstance(num, Numeric)

    num += k
    assert isinstance(num, Element)

    num = B.randn(dtype)
    num += k(x)
    assert isinstance(num, Element)

    # Test in-place multiplication.
    num = B.randn(dtype)
    num *= B.randn(dtype)
    assert isinstance(num, Numeric)

    num *= k
    assert isinstance(num, Element)

    num = B.randn(dtype)
    num *= k(x)
    assert isinstance(num, Element)
