# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from stheno import LazyVector, LazySymmetricMatrix, Rule
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint


def test_exceptions():
    yield eq, repr(Rule(1, {1}, 1)), 'Rule(pattern=1, indices={1}, builder=1)'


def test_indexing():
    v = LazyVector()
    m = LazySymmetricMatrix()

    v[1] = 1
    yield eq, v[1], 1

    # Test diagonal.
    m[1] = 1
    m[2, 2] = 2
    yield eq, m[1, 1], 1
    yield eq, m[2], 2

    # Test symmetry.
    m[1, 2] = (1, 2)
    yield eq, m[1, 2], (1, 2)
    yield eq, tuple(m[2, 1]), (2, 1)

    # Test resolving of indices.
    class A(object):
        pass

    a1, a2 = A(), A()
    m[id(a1), a2] = (1, 2)
    m[id(a1)] = 1
    m[a2] = 2

    yield eq, m[a1, a1], 1
    yield eq, tuple(m[a2, a1]), (2, 1)
    yield eq, m[a2, a2], 2


def test_building():
    class ReversibleNumber(object):
        def __init__(self, x):
            self.x = x

        def __reversed__(self):
            return ReversibleNumber(self.x)

    m = LazySymmetricMatrix()

    # Test universal building.
    m.add_rule((None, None), range(3), lambda x, y: ReversibleNumber(x * y))

    for i in range(3):
        for j in range(3):
            yield eq, m[i, j].x, i * j

    yield raises, RuntimeError, lambda: m[-1, 0]
    yield raises, RuntimeError, lambda: m[0, -1]
    yield raises, RuntimeError, lambda: m[-1, -1]
    yield raises, RuntimeError, lambda: m[4, 3]
    yield raises, RuntimeError, lambda: m[3, 4]
    yield raises, RuntimeError, lambda: m[4, 4]

    # Test building along first dimension.
    m.add_rule((3, None), range(4), lambda y: ReversibleNumber(3 + y))

    for i in range(3):
        for j in range(3):
            yield eq, m[i, j].x, i * j

    for i in range(3):
        yield eq, m[i, 3].x, i + 3
        yield eq, m[3, i].x, 3 + i
    yield eq, m[3, 3].x, 3 + 3

    yield raises, RuntimeError, lambda: m[-1, 0]
    yield raises, RuntimeError, lambda: m[0, -1]
    yield raises, RuntimeError, lambda: m[-1, -1]
    yield raises, RuntimeError, lambda: m[5, 4]
    yield raises, RuntimeError, lambda: m[4, 5]
    yield raises, RuntimeError, lambda: m[5, 5]

    # Test building along second dimension.
    m.add_rule((None, 4), range(5), lambda x: ReversibleNumber(x ** 2 + 4 ** 2))

    for i in range(3):
        for j in range(3):
            yield eq, m[i, j].x, i * j

    for i in range(3):
        yield eq, m[i, 3].x, i + 3
        yield eq, m[3, i].x, 3 + i
    yield eq, m[3, 3].x, 3 + 3

    for i in range(4):
        yield eq, m[i, 4].x, i ** 2 + 4 ** 2
        yield eq, m[4, i].x, 4 ** 2 + i ** 2
    yield eq, m[4, 4].x, 4 ** 2 + 4 ** 2

    yield raises, RuntimeError, lambda: m[-1, 0]
    yield raises, RuntimeError, lambda: m[0, -1]
    yield raises, RuntimeError, lambda: m[-1, -1]
    yield raises, RuntimeError, lambda: m[6, 5]
    yield raises, RuntimeError, lambda: m[5, 6]
    yield raises, RuntimeError, lambda: m[6, 6]

    # Now try all rules at once and mix up access order.
    m2 = LazySymmetricMatrix()
    m2.add_rule((None, None), range(3), lambda x, y: ReversibleNumber(x * y))
    m2.add_rule((3, None), range(4), lambda y: ReversibleNumber(3 + y))
    m2.add_rule((None, 4), range(5),
                lambda x: ReversibleNumber(x ** 2 + 4 ** 2))

    for i in range(4):
        yield eq, m2[i, 4].x, i ** 2 + 4 ** 2
        yield eq, m2[4, i].x, 4 ** 2 + i ** 2
    yield eq, m2[4, 4].x, 4 ** 2 + 4 ** 2

    for i in range(3):
        yield eq, m2[i, 3].x, i + 3
        yield eq, m2[3, i].x, 3 + i
    yield eq, m2[3, 3].x, 3 + 3

    for i in range(3):
        for j in range(3):
            yield eq, m2[i, j].x, i * j

    yield raises, RuntimeError, lambda: m2[-1, 0]
    yield raises, RuntimeError, lambda: m2[0, -1]
    yield raises, RuntimeError, lambda: m2[-1, -1]
    yield raises, RuntimeError, lambda: m2[6, 5]
    yield raises, RuntimeError, lambda: m2[5, 6]
    yield raises, RuntimeError, lambda: m2[6, 6]
