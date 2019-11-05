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


@pytest.mark.parametrize(
    'generate, Numeric',
    [(lambda *shape: B.randn(torch.float64, *shape), B.TorchNumeric),
     (lambda *shape: B.randn(tf.float64, *shape), B.TFNumeric),
     (lambda *shape: tf.Variable(B.randn(tf.float64, *shape)), B.TFNumeric)]
)
def test_boxing(generate, Numeric):
    k = EQ()
    num = generate()
    x = generate(10)

    # Test addition and multiplication.
    assert isinstance(num + k, Element)
    assert isinstance(num + k(x), Element)
    assert isinstance(num + num, Numeric)
    assert isinstance(num * k, Element)
    assert isinstance(num * k(x), Element)
    assert isinstance(num * num, Numeric)


@pytest.mark.parametrize(
    'generate, Numeric',
    [(lambda *shape: B.randn(torch.float64, *shape), B.TorchNumeric),
     (lambda *shape: B.randn(tf.float64, *shape), B.TFNumeric)]
)
def test_boxing_inplace(generate, Numeric):
    k = EQ()
    x = generate(10)

    # Test in-place addition.
    num = generate()
    num += generate()
    assert isinstance(num, Numeric)

    num += k
    assert isinstance(num, Element)

    num = generate()
    num += k(x)
    assert isinstance(num, Element)

    # Test in-place multiplication.
    num = generate()
    num *= generate()
    assert isinstance(num, Numeric)

    num *= k
    assert isinstance(num, Element)

    num = generate()
    num *= k(x)
    assert isinstance(num, Element)
