from lab import B

from stheno.util import uprank, num_elements
from .util import approx


def test_num_elements():
    assert num_elements(B.randn()) == 1
    assert num_elements(B.randn(2)) == 2
    assert num_elements(B.randn(2, 2)) == 2


def test_uprank():
    # Check decorative behaviour.
    @uprank
    def with_uprank(x, y):
        return x, y

    # Check upranking behaviour.
    approx(with_uprank(0, B.zeros(1)), ([[0]], [[0]]))
    approx(with_uprank(B.zeros(1), B.zeros(1)), ([[0]], [[0]]))
    approx(with_uprank(B.zeros(1, 1), B.zeros(1)), ([[0]], [[0]]))

    # Check that it leaves objects alone.
    x = []
    assert uprank(x) is x
