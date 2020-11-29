import pytest

from stheno.lazy import LazyMatrix, LazyVector


def test_indexing():
    v = LazyVector()
    m = LazyMatrix()

    v[1] = 1
    assert v[1] == 1

    # Test diagonal indexing and setting.
    m[1] = 1
    m[2, 2] = 2
    assert m[1, 1] == 1
    assert m[2] == 2

    # Test resolving of indices.
    class A:
        pass

    a1, a2 = A(), A()
    m[id(a1), a2] = (1, 2)
    m[id(a1)] = 1
    m[a2] = 2

    assert m[a1, a1] == 1
    assert m[a1, a2] == (1, 2)
    assert m[a2, a2] == 2


class Number:
    def __init__(self, x):
        self.x = x


def test_lazy_vector_building():
    v = LazyVector()

    # Test universal building.
    v.add_rule(set(range(3)), lambda x: Number(x))

    for i in range(3):
        assert v[i].x == i

    # Test bounds checking.
    with pytest.raises(RuntimeError):
        v[-1]
    with pytest.raises(RuntimeError):
        v[3]

    # Stack another universal building rule.
    v.add_rule(set(range(4)), lambda x: Number(3 + x))

    for i in range(3):
        assert v[i].x == i

    assert v[3].x == 3 + 3

    # Test bounds checking.
    with pytest.raises(RuntimeError):
        v[-1]
    with pytest.raises(RuntimeError):
        v[4]

    # Now try all rules at once and mix up access order.
    v2 = LazyVector()
    v2.add_rule(set(range(3)), lambda x: Number(x))
    v2.add_rule(set(range(4)), lambda x: Number(3 + x))

    assert v2[3].x == 3 + 3

    for i in range(3):
        assert v2[i].x == i

    # Test bounds checking.
    with pytest.raises(RuntimeError):
        v2[-1]
    with pytest.raises(RuntimeError):
        v2[4]


def test_lazy_vector_mutation_protection():
    inds = {1}
    v = LazyVector()
    v.add_rule(inds, lambda i: 1)
    inds.add(2)
    assert v[1] == 1
    with pytest.raises(RuntimeError):
        v[2]


def test_lazy_matrix_building():
    m = LazyMatrix()

    def check_bounds(bounds):
        for bound in bounds:
            with pytest.raises(RuntimeError):
                m[bound]

    # Test universal building.
    m.add_rule(set(range(3)), lambda x, y: Number(x * y))

    # Check content.
    for i in range(3):
        for j in range(3):
            assert m[i, j].x == i * j

    # Check bounds.
    check_bounds([(-1, 0), (0, -1), (-1, -1), (4, 3), (3, 4), (4, 4)])

    # Test left and right building rules.
    m.add_left_rule(3, set(range(4)), lambda y: Number(3 + y))
    m.add_right_rule(3, set(range(4)), lambda y: Number(3 + y))

    # Check content.
    for i in range(3):
        for j in range(3):
            assert m[i, j].x == i * j
    for i in range(4):
        assert m[i, 3].x == i + 3
        assert m[3, i].x == 3 + i
    assert m[3, 3].x == 3 + 3

    # Check bounds.
    check_bounds([(-1, 0), (0, -1), (-1, -1), (5, 4), (4, 5), (5, 5)])

    # Stack more left and right building rules.
    m.add_right_rule(4, set(range(5)), lambda x: Number(x ** 2 + 4 ** 2))
    m.add_left_rule(4, set(range(5)), lambda x: Number(x ** 2 + 4 ** 2))

    # Check content.
    for i in range(3):
        for j in range(3):
            assert m[i, j].x == i * j
    for i in range(4):
        assert m[i, 3].x == i + 3
        assert m[3, i].x == 3 + i
    assert m[3, 3].x == 3 + 3
    for i in range(5):
        assert m[i, 4].x == i ** 2 + 4 ** 2
        assert m[4, i].x == 4 ** 2 + i ** 2
    assert m[4, 4].x == 4 ** 2 + 4 ** 2

    # Check bounds.
    check_bounds([(-1, 0), (0, -1), (-1, -1), (6, 5), (5, 6), (6, 6)])

    # Now try all rules at once and mix up access order.
    m2 = LazyMatrix()
    m2.add_rule(set(range(3)), lambda x, y: Number(x * y))
    m2.add_left_rule(3, set(range(4)), lambda y: Number(3 + y))
    m2.add_right_rule(3, set(range(4)), lambda y: Number(3 + y))
    m2.add_right_rule(4, set(range(5)), lambda x: Number(x ** 2 + 4 ** 2))
    m2.add_left_rule(4, set(range(5)), lambda x: Number(x ** 2 + 4 ** 2))

    # Check content.
    for i in range(5):
        assert m2[i, 4].x == i ** 2 + 4 ** 2
        assert m2[4, i].x == 4 ** 2 + i ** 2
    assert m2[4, 4].x == 4 ** 2 + 4 ** 2
    for i in range(4):
        assert m2[i, 3].x == i + 3
        assert m2[3, i].x == 3 + i
    assert m2[3, 3].x == 3 + 3
    for i in range(3):
        for j in range(3):
            assert m2[i, j].x == i * j

    # Check bounds.
    check_bounds([(-1, 0), (0, -1), (-1, -1), (6, 5), (5, 6), (6, 6)])


def test_lazy_matrix_mutation_protection():
    def check_access(m):
        assert m[1, 1] == 1
        with pytest.raises(RuntimeError):
            m[1, 2]
        with pytest.raises(RuntimeError):
            m[2, 1]
        with pytest.raises(RuntimeError):
            m[2, 2]

    # Check universal rule.
    inds = {1}
    m = LazyMatrix()
    m.add_rule(inds, lambda i, j: 1)
    inds.add(2)
    check_access(m)

    # Check left rule.
    inds = {1}
    m = LazyMatrix()
    m.add_left_rule(1, inds, lambda j: 1)
    inds.add(2)
    check_access(m)

    # Check right rule.
    inds = {1}
    m = LazyMatrix()
    m.add_right_rule(1, inds, lambda i: 1)
    inds.add(2)
    check_access(m)
