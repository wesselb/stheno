from typing import Union

import logging
import abc

from plum import Dispatcher

__all__ = []

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


@_dispatch
def _resolve_index(key):
    return id(key)


@_dispatch
def _resolve_index(i: int):
    return i


@_dispatch
def _resolve_index(x: Union[tuple, reversed]):
    return tuple(_resolve_index(key) for key in x)


class LazyTensor:
    """A lazy tensor that indexes by the identity of its indices.

    Args:
        rank (int): Rank of the tensor.
    """

    def __init__(self, rank):
        self._rank = rank
        self._store = {}

    @_dispatch
    def _expand_key(self, key: tuple):
        # Nothing to do. The key needs to be a tuple.
        return key

    @_dispatch
    def _expand_key(self, key):
        return (key,) * self._rank

    def __setitem__(self, key, value):
        return self._set(_resolve_index(self._expand_key(key)), value)

    def __getitem__(self, key):
        return self._get(_resolve_index(self._expand_key(key)))

    def _set(self, i, value):
        self._store[i] = value

    def _get(self, i):
        try:
            return self._store[i]
        except KeyError:
            pass

        # Could not find element. Try building it.
        value = self._build(i)
        self._store[i] = value
        return value

    @abc.abstractmethod
    def _build(self, i):  # pragma: no cover
        pass


class LazyVector(LazyTensor):
    """A lazy vector."""

    def __init__(self):
        LazyTensor.__init__(self, 1)
        self._rules = []

    def add_rule(self, indices, builder):
        """Add a building rule.

        Note:
            For performance reasons, `indices` must already be resolved!

        Args:
            indices (set): Domain of the rule.
            builder (function): Function that takes in an index and gives back the
                corresponding element.
        """
        self._rules.append((frozenset(indices), builder))

    def _build(self, i):
        i = i[0]  # This will be a one-tuple.
        for indices, builder in self._rules:
            if i in indices:
                return builder(i)
        raise RuntimeError(f'Could not build value for index "{i}".')


class LazyMatrix(LazyTensor):
    """A lazy matrix."""

    def __init__(self):
        LazyTensor.__init__(self, 2)
        self._left_rules = []
        self._right_rules = []
        self._rules = []

    def add_rule(self, indices, builder):
        """Add a building rule.

        Note:
            For performance reasons, `indices` must already be resolved!

        Args:
            indices (set): Domain of the rule.
            builder (function): Function that takes in an index and gives back the
                corresponding element.
        """
        self._rules.append((frozenset(indices), builder))

    def add_left_rule(self, i_left, indices, builder):
        """Add a building rule for a given left index.

        Note:
            For performance reasons, `indices` must already be resolved!

        Args:
            i_left (int): Fixed left index.
            indices (set): Domain of the rule.
            builder (function): Function that takes in a right index and gives back the
                corresponding element.
        """
        self._left_rules.append((i_left, frozenset(indices), builder))

    def add_right_rule(self, i_right, indices, builder):
        """Add a building rule for a given right index.

        Note:
            For performance reasons, `indices` must already be resolved!

        Args:
            i_right (int): Fixed right index.
            indices (set): Domain of the rule.
            builder (function): Function that takes in a left index and gives back the
                corresponding element.
        """
        self._right_rules.append((i_right, frozenset(indices), builder))

    def _build(self, i):
        i_left, i_right = i

        # Check universal rules.
        for indices, builder in self._rules:
            if i_left in indices and i_right in indices:
                return builder(i_left, i_right)

        # Check left rules.
        for i_left_rule, indices, builder in self._left_rules:
            if i_left == i_left_rule and i_right in indices:
                return builder(i_right)

        # Check right rules.
        for i_right_rule, indices, builder in self._right_rules:
            if i_left in indices and i_right == i_right_rule:
                return builder(i_left)

        raise RuntimeError(f"Could not build value for index {i}.")
