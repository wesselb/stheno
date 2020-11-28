import numpy as np
import pytest
from lab import B

from stheno.graph import GP, Graph
from stheno.input import Component
from stheno.kernel import OneKernel, EQ
from stheno.mean import OneMean
from stheno.util import uprank, num_elements
from .util import approx


def test_num_elements():
    assert num_elements(B.randn()) == 1
    assert num_elements(B.randn(2)) == 2
    assert num_elements(B.randn(2, 2)) == 2


def test_uprank():
    # Check upranking behaviour.
    approx(uprank(0), [[0]])
    approx(uprank(np.array([0])), [[0]])
    approx(uprank(np.array([[0]])), [[0]])

    # Check that it leaves objects alone.
    assert type(uprank(Component("test")(0))) == Component("test")

    k = OneKernel()

    assert B.shape(k(0, 0)) == (1, 1)
    assert B.shape(k(0, np.ones(5))) == (1, 5)
    assert B.shape(k(0, np.ones((5, 2)))) == (1, 5)

    assert B.shape(k(np.ones(5), 0)) == (5, 1)
    assert B.shape(k(np.ones(5), np.ones(5))) == (5, 5)
    assert B.shape(k(np.ones(5), np.ones((5, 2)))) == (5, 5)

    assert B.shape(k(np.ones((5, 2)), 0)) == (5, 1)
    assert B.shape(k(np.ones((5, 2)), np.ones(5))) == (5, 5)
    assert B.shape(k(np.ones((5, 2)), np.ones((5, 2)))) == (5, 5)

    m = OneMean()

    assert B.shape(m(0)) == (1, 1)
    assert B.shape(m(np.ones(5))) == (5, 1)
    assert B.shape(m(np.ones((5, 2)))) == (5, 1)

    p = GP(EQ(), graph=Graph())

    x = B.linspace(0, 10, 10)

    approx(p.condition(1, 1)(1).mean, np.array([[1]]))
    approx(p.condition(x, x)(x).mean, x[:, None])
    approx(p.condition(x, x[:, None])(x).mean, x[:, None])
