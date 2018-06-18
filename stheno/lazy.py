# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from plum import Dispatcher, Referentiable, Self

__all__ = []

log = logging.getLogger(__name__)


def replace_at(tup, i, val):
    """Replace a value in a tuple.

    Args:
        tup (tuple): Tuple to replace value in.
        i (int): Index to replace value at.
        val (obj): New value.

    Returns:
        tuple: Tuple with the value replaced.
    """
    listed = list(tup)
    listed[i] = val
    return tuple(listed)


class Rule(object):
    """A rule for an :class:`.lazy.LazyTensor`.

    Note:
        For performance reasons, `indices` must already be resolved!

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

        Returns:
            bool: Boolean indicating whether the rule applies.
        """
        # Already keep track of arguments for building.
        self._args = ()

        for (i, j) in zip(self.pattern, key):
            if i is None:
                if j not in self.indices:
                    return False
                else:
                    self._args += (j,)
            elif i != j:
                return False
        return True

    def build(self):
        """After checking, build the element."""
        return self.builder(*self._args)

    def __repr__(self):
        return 'Rule(pattern={!r}, indices={!r}, builder={!r})' \
               ''.format(self.pattern, self.indices, self.builder)


class LazyTensor(Referentiable):
    """A lazy tensor that indexes by the identity of its indices.

    Args:
        d (int): Rank of the tensor.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, rank):
        self.rank = rank
        self._rules = {}
        self._store = {}

    @_dispatch(object)
    def _resolve_index(self, i):
        return id(i)

    @_dispatch(type(None))
    def _resolve_index(self, i):
        return None

    @_dispatch(int)
    def _resolve_index(self, i):
        return i

    @_dispatch(object)
    def _resolve_key(self, key):
        return (self._resolve_index(key),) * self.rank

    @_dispatch({tuple, reversed})
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

        # Finally, try building the element.
        value = self._build(key)
        self._store[key] = value
        return value

    def _build(self, key):
        # NOTE: The building patterns should go from least specific to most
        # specific.
        # Try a universal match.
        building_patterns = [(None,) * self.rank]
        # Try single dimension patterns.
        building_patterns += [replace_at(key, i, None) for i in
                              range(self.rank)]
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

    @_dispatch([object])
    def add_rule(self, pattern, indices, builder):
        """Add a building rule.

        See :class:`.lazy.Rule`.

        Note:
            For performance reasons, `indices` must already be resolved!

        Args:
            pattern (tuple): Rule pattern. Each element must be a `None` or an
                element from `indices`. A `None` represents a wildcard.
            indices (set): Values to match wildcards with. These must already have
                been resolved.
            builder (function): Function that builds the element. Only the wildcard
                elements are fed to the function.
        """
        pattern = self._resolve_key(pattern)
        # IMPORTANT: This assumes that the indices already have been resolved!
        rule = Rule(pattern, indices, builder)
        try:
            self._rules[pattern].append(rule)
        except KeyError:
            self._rules[pattern] = [rule]


class LazyVector(LazyTensor):
    """A lazy vector."""

    def __init__(self):
        LazyTensor.__init__(self, 1)


class LazyMatrix(LazyTensor):
    """A lazy matrix."""

    def __init__(self):
        LazyTensor.__init__(self, 2)
