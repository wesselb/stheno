# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import pytest

from stheno.lazy import Rule, LazyMatrix, LazyVector


def test_corner_cases():
    assert repr(Rule(1, {1}, 1)) == \
           'Rule(pattern=1, indices={!r}, builder=1)'.format({1})


def test_indexing():
    v = LazyVector()
    m = LazyMatrix()

    v[1] = 1
    assert v[1] == 1

    # Test diagonal.
    m[1] = 1
    m[2, 2] = 2
    assert m[1, 1] == 1
    assert m[2] == 2

    # Test resolving of indices.
    class A(object):
        pass

    a1, a2 = A(), A()
    m[id(a1), a2] = (1, 2)
    m[id(a1)] = 1
    m[a2] = 2

    assert m[a1, a1] == 1
    assert m[a1, a2] == (1, 2)
    assert m[a2, a2] == 2


def test_rule():
    rule = Rule((1, None, 2, None, 3), {1, 2, 3, 4, 5}, lambda: None)

    assert rule.applies((1, 1, 2, 5, 3))
    assert not rule.applies((1, 1, 2, 6, 3))
    assert not rule.applies((2, 1, 2, 5, 3))


def test_building():
    class ReversibleNumber(object):
        def __init__(self, x):
            self.x = x

        def __reversed__(self):
            return ReversibleNumber(self.x)

    m = LazyMatrix()

    # Test universal building.
    m.add_rule((None, None), range(3), lambda x, y: ReversibleNumber(x * y))

    for i in range(3):
        for j in range(3):
            assert m[i, j].x == i * j

    with pytest.raises(RuntimeError):
        m[-1, 0]
    with pytest.raises(RuntimeError):
        m[0, -1]
    with pytest.raises(RuntimeError):
        m[-1, -1]
    with pytest.raises(RuntimeError):
        m[4, 3]
    with pytest.raises(RuntimeError):
        m[3, 4]
    with pytest.raises(RuntimeError):
        m[4, 4]

    # Test building along first dimension.
    m.add_rule((3, None), range(4), lambda y: ReversibleNumber(3 + y))
    m.add_rule((None, 3), range(4), lambda y: ReversibleNumber(3 + y))

    for i in range(3):
        for j in range(3):
            assert m[i, j].x == i * j

    for i in range(3):
        assert m[i, 3].x == i + 3
        assert m[3, i].x == 3 + i
    assert m[3, 3].x == 3 + 3

    with pytest.raises(RuntimeError):
        m[-1, 0]
    with pytest.raises(RuntimeError):
        m[0, -1]
    with pytest.raises(RuntimeError):
        m[-1, -1]
    with pytest.raises(RuntimeError):
        m[5, 4]
    with pytest.raises(RuntimeError):
        m[4, 5]
    with pytest.raises(RuntimeError):
        m[5, 5]

    # Test building along second dimension.
    m.add_rule((None, 4), range(5), lambda x: ReversibleNumber(x ** 2 + 4 ** 2))
    m.add_rule((4, None), range(5), lambda x: ReversibleNumber(x ** 2 + 4 ** 2))

    for i in range(3):
        for j in range(3):
            assert m[i, j].x == i * j

    for i in range(3):
        assert m[i, 3].x == i + 3
        assert m[3, i].x == 3 + i
    assert m[3, 3].x == 3 + 3

    for i in range(4):
        assert m[i, 4].x == i ** 2 + 4 ** 2
        assert m[4, i].x == 4 ** 2 + i ** 2
    assert m[4, 4].x == 4 ** 2 + 4 ** 2

    with pytest.raises(RuntimeError):
        m[-1, 0]
    with pytest.raises(RuntimeError):
        m[0, -1]
    with pytest.raises(RuntimeError):
        m[-1, -1]
    with pytest.raises(RuntimeError):
        m[6, 5]
    with pytest.raises(RuntimeError):
        m[5, 6]
    with pytest.raises(RuntimeError):
        m[6, 6]

    # Now try all rules at once and mix up access order.
    m2 = LazyMatrix()
    m2.add_rule((None, None), range(3), lambda x, y: ReversibleNumber(x * y))
    m2.add_rule((3, None), range(4), lambda y: ReversibleNumber(3 + y))
    m2.add_rule((None, 3), range(4), lambda y: ReversibleNumber(3 + y))
    m2.add_rule((None, 4), range(5),
                lambda x: ReversibleNumber(x ** 2 + 4 ** 2))
    m2.add_rule((4, None), range(5),
                lambda x: ReversibleNumber(x ** 2 + 4 ** 2))

    for i in range(4):
        assert m2[i, 4].x == i ** 2 + 4 ** 2
        assert m2[4, i].x == 4 ** 2 + i ** 2
    assert m2[4, 4].x == 4 ** 2 + 4 ** 2

    for i in range(3):
        assert m2[i, 3].x == i + 3
        assert m2[3, i].x == 3 + i
    assert m2[3, 3].x == 3 + 3

    for i in range(3):
        for j in range(3):
            assert m2[i, j].x == i * j

    with pytest.raises(RuntimeError):
        m2[-1, 0]
    with pytest.raises(RuntimeError):
        m2[0, -1]
    with pytest.raises(RuntimeError):
        m2[-1, -1]
    with pytest.raises(RuntimeError):
        m2[6, 5]
    with pytest.raises(RuntimeError):
        m2[5, 6]
    with pytest.raises(RuntimeError):
        m2[6, 6]
