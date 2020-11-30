import jax
import jax.numpy as jnp
import lab as B
import pytest
import tensorflow as tf
import torch
from algebra import Element
import autograd
from lab import torch as B
from matrix import AbstractMatrix, Dense

from stheno.torch import EQ


@pytest.mark.parametrize("grad", [autograd.grad, lambda f: jax.jit(jax.grad(f))])
def test_boxing_objective(grad):
    k = EQ()
    objs = []

    def objective(x):
        x = x + 2
        x = x * 2
        objs.append(x[0] + k)
        objs.append(x[0] + Dense(k(x)))
        objs.append(x[0] * k)
        objs.append(x[0] * Dense(k(x)))
        return B.sum(x)

    grad(objective)(B.randn(10))

    for obj in objs:
        assert isinstance(obj, (Element, AbstractMatrix))


@pytest.mark.parametrize(
    "generate, Numeric",
    [
        (lambda *shape: B.randn(torch.float64, *shape), B.TorchNumeric),
        (lambda *shape: B.randn(tf.float64, *shape), B.TFNumeric),
        (lambda *shape: tf.Variable(B.randn(tf.float64, *shape)), B.TFNumeric),
    ],
)
def test_boxing_direct(generate, Numeric):
    k = EQ()
    num = generate()
    x = generate(10)

    # Test addition and multiplication.
    assert isinstance(num + k, Element)
    assert isinstance(num + Dense(k(x)), AbstractMatrix)
    assert isinstance(num + num, Numeric)
    assert isinstance(num * k, Element)
    assert isinstance(num * Dense(k(x)), AbstractMatrix)
    assert isinstance(num * num, Numeric)


@pytest.mark.parametrize(
    "generate, Numeric",
    [
        (lambda *shape: B.randn(torch.float64, *shape), B.TorchNumeric),
        (lambda *shape: B.randn(tf.float64, *shape), B.TFNumeric),
    ],
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
    num += Dense(k(x))
    assert isinstance(num, AbstractMatrix)

    # Test in-place multiplication.
    num = generate()
    num *= generate()
    assert isinstance(num, Numeric)

    num *= k
    assert isinstance(num, Element)

    num = generate()
    num *= Dense(k(x))
    assert isinstance(num, AbstractMatrix)
