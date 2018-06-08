# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from plum import Dispatcher, Referentiable, Self
import logging

__all__ = ['LazyVector', 'LazySymmetricMatrix', 'Rule']

log = logging.getLogger(__name__)


def replace_at(tup, i, val):
    """Replace a value in a tuple.

    Args:
        tup (tuple): Tuple to replace value in.
        i (int): Index to replace value at.
        val (obj): New value.
    """
    listed = list(tup)
    listed[i] = val
    return tuple(listed)


class Rule(object):
    """A rule for an :class:`.lazy.LazyTensor`.

    IMPORTANT: For performance reasons, `indices` must already be resolved!

    Args:
        pattern (tuple): Rule pattern. Each element must be a `None` or an
            element from `indices`. A `None` represents a wildcard.
        indices (set): Values to match wildcards with. These must already have
            been resolved.
        builder (function): Function that builds the element. Only the wildcard
            elements are fed to the function.
    """
    def __init__(self, pattern, indices, builder):
        self.pattern = pattern
        self.indices = set(indices)
        self.builder = builder
        self._args = None

    def applies(self, key):
        """Check whether this rule applies to a certain key.

        Args:
            key (tuple): Key.
        """
        # Already keep track of arguments for building.
        self._args = ()

        for (i, j) in zip(self.pattern, key):
            if i is None:
                if j not in self.indices:
                    return False
                else:
                    self._args += (j,)
                    continue
            else:
                if i != j:
                    return False
                else:
                    continue
        return True

    def build(self):
        """After checking, build the element."""
        return self.builder(*self._args)

    def __repr__(self):
        return 'Rule(pattern={!r}, indices={!r}, builder={!r})' \
               ''.format(self.pattern, self.indices, self.builder)


class LazySymmetricTensor(Referentiable):
    """A symmetric lazy tensor that indexes by the identity of its indices.

    Args:
        d (int): Rank of the tensor.
    """
    dispatch = Dispatcher(in_class=Self)

    def __init__(self, rank):
        self.rank = rank
        self._rules = {}
        self._store = {}

    @dispatch(object)
    def _resolve_index(self, i):
        return id(i)

    @dispatch(type(None))
    def _resolve_index(self, i):
        return None

    @dispatch(int)
    def _resolve_index(self, i):
        return i

    @dispatch(object)
    def _resolve_key(self, key):
        return (self._resolve_index(key),) * self.rank

    @dispatch({tuple, reversed})
    def _resolve_key(self, key):
        return tuple(self._resolve_index(i) for i in key)

    def __setitem__(self, key, value):
        return self._set(self._resolve_key(key), value)

    def __getitem__(self, key):
        return self._get(self._resolve_key(key))

    def _set(self, key, value):
        self._store[key] = value

    def _get(self, key):
        try:
            return self._store[key]
        except KeyError:
            pass

        reversed_key = tuple(reversed(key))
        try:
            return reversed(self._store[reversed_key])
        except KeyError:
            pass

        # Finally, try building the element.
        try:
            value = self._build(key)
        except RuntimeError:
            value = reversed(self._build(reversed_key))

        self._store[key] = value
        return value

    def _build(self, key):
        # NOTE: The building patterns should go from least specific to most
        # specific.
        # Try a universal match.
        building_patterns = [(None,) * self.rank]
        # Try single dimension patterns.
        building_patterns += [replace_at(key, i, None) for i in range(self.rank)]
        # Try key.
        building_patterns += [key]

        for pattern in building_patterns:
            # Check if a rules exists for the pattern.
            if pattern in self._rules:
                # Rules exist! Build with the first one that applies.
                for rule in self._rules[pattern]:
                    if rule.applies(key):
                        return rule.build()

        # Unable to build value for key.
        raise RuntimeError('Could not build value for key "{}".'.format(key))

    @dispatch([object])
    def add_rule(self, pattern, indices, builder):
        """Add a building rule.

        See :class:`.lazy.Rule`.

        IMPORTANT: For performance reasons, `indices` must already be resolved!

        Args:
            pattern (tuple): Pattern to match.
            indices (set): Elements to match wildcards with.
            builder (function): Building function.
        """
        pattern = self._resolve_key(pattern)
        # IMPORTANT: This assumes that the indices already have been resolved!
        rule = Rule(pattern, indices, builder)
        try:
            self._rules[pattern].append(rule)
        except KeyError:
            self._rules[pattern] = [rule]


class LazyVector(LazySymmetricTensor):
    """A lazy vector."""
    def __init__(self):
        LazySymmetricTensor.__init__(self, 1)


class LazySymmetricMatrix(LazySymmetricTensor):
    """A lazy matrix."""
    def __init__(self):
        LazySymmetricTensor.__init__(self, 2)
