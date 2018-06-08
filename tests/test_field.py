# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from stheno import EQ, RQ, Linear, OneKernel, ZeroKernel, Delta
# noinspection PyUnresolvedReferences
from tests import ok, raises
from . import eq


def test_exceptions():
    yield raises, IndexError


def test_cancellations_zero():
    # Sums:
    yield eq, str(EQ() + EQ()), '2 * EQ()'
    yield eq, str(ZeroKernel() + EQ()), 'EQ()'
    yield eq, str(EQ() + ZeroKernel()), 'EQ()'
    yield eq, str(ZeroKernel() + ZeroKernel()), '0'

    # Products:
    yield eq, str(EQ() * EQ()), 'EQ() * EQ()'
    yield eq, str(ZeroKernel() * EQ()), '0'
    yield eq, str(EQ() * ZeroKernel()), '0'
    yield eq, str(ZeroKernel() * ZeroKernel()), '0'

    # Scales:
    yield eq, str(5 * ZeroKernel()), '0'
    yield eq, str(ZeroKernel() * 5), '0'
    yield eq, str(EQ() * 5), '5 * EQ()'
    yield eq, str(5 * EQ()), '5 * EQ()'

    # Stretches:
    yield eq, str(ZeroKernel().stretch(5)), '0'
    yield eq, str(EQ().stretch(5)), 'EQ() > 5'

    # Periodicisations:
    yield eq, str(ZeroKernel().periodic(5)), '0'
    yield eq, str(EQ().periodic(5)), 'EQ() per 5'

    # Reversals:
    yield eq, str(reversed(ZeroKernel())), '0'
    yield eq, str(reversed(EQ())), 'EQ()'
    yield eq, str(reversed(Linear())), 'Reversed(Linear())'

    # Integration:
    yield eq, str(EQ() * EQ() + ZeroKernel() * EQ()), 'EQ() * EQ()'
    yield eq, str(EQ() * ZeroKernel() + ZeroKernel() * EQ()), '0'


def test_cancellations_one():
    # Products:
    yield eq, str(EQ() * EQ()), 'EQ() * EQ()'
    yield eq, str(OneKernel() * EQ()), 'EQ()'
    yield eq, str(EQ() * OneKernel()), 'EQ()'
    yield eq, str(OneKernel() * OneKernel()), '1'


def test_grouping():
    # Scales:
    yield eq, str(5 * EQ()), '5 * EQ()'
    yield eq, str(5 * (5 * EQ())), '25 * EQ()'

    # Stretches:
    yield eq, str(EQ().stretch(5)), 'EQ() > 5'
    yield eq, str(EQ().stretch(5).stretch(5)), 'EQ() > 25'

    # Products:
    yield eq, str((5 * EQ()) * (5 * EQ())), '25 * EQ()'
    yield eq, str((5 * (EQ() * EQ())) * (5 * EQ() * EQ())), \
          '25 * EQ() * EQ()'
    yield eq, str((5 * RQ(1)) * (5 * RQ(2))), '25 * RQ(1) * RQ(2)'

    # Sums:
    yield eq, str((5 * EQ()) + (5 * EQ())), '10 * EQ()'
    yield eq, str(EQ() + (5 * EQ())), '6 * EQ()'
    yield eq, str((5 * EQ()) + EQ()), '6 * EQ()'
    yield eq, str((EQ() + EQ())), '2 * EQ()'
    yield eq, str((5 * (EQ() * EQ())) + (5 * (EQ() * EQ()))), \
          '10 * EQ() * EQ()'
    yield eq, str((5 * RQ(1)) + (5 * RQ(2))), '5 * RQ(1) + 5 * RQ(2)'

    # Reversal:
    yield eq, str(reversed(Linear() + EQ())), 'Reversed(Linear()) + EQ()'
    yield eq, str(reversed(Linear() * EQ())), 'Reversed(Linear()) * EQ()'


def test_parentheses():
    yield eq, str((reversed(Linear() * Linear() +
                            2 * EQ().stretch(1).periodic(2) +
                            RQ(3).periodic(4))) *
                  (EQ().stretch(1) + EQ())), \
          '(Reversed(Linear()) * Reversed(Linear()) + ' \
          '2 * ((EQ() > 1) per 2) + RQ(3) per 4) * (EQ() > 1 + EQ())'


def test_terms():
    yield eq, str((EQ() + EQ() * Linear() +
                   RQ(1) * RQ(2) + Delta()).term(0)), 'EQ()'
    yield eq, str((EQ() + EQ() * Linear() +
                   RQ(1) * RQ(2) + Delta()).term(1)), 'EQ() * Linear()'
    yield eq, str((EQ() + EQ() * Linear() +
                   RQ(1) * RQ(2) + Delta()).term(2)), 'RQ(1) * RQ(2)'
    yield eq, str((EQ() + EQ() * Linear() +
                   RQ(1) * RQ(2) + Delta()).term(3)), 'Delta()'
    yield raises, IndexError, lambda: (EQ() + EQ() * Linear() +
                                       RQ(1) * RQ(2) + Delta()).term(4)


def test_factors():
    yield eq, str((EQ() * Linear()).factor(0)), 'EQ()'
    yield eq, str((EQ() * Linear()).factor(1)), 'Linear()'
    yield raises, IndexError, lambda: (EQ() * Linear()).factor(2)

    yield eq, str(((EQ() + EQ()) * Delta() * (RQ(1) + Linear())).factor(0)), \
          '2'
    yield eq, str(((EQ() + EQ()) * Delta() * (RQ(1) + Linear())).factor(1)), \
          'EQ()'
    yield eq, str(((EQ() + EQ()) * Delta() * (RQ(1) + Linear())).factor(2)), \
          'Delta()'
    yield eq, str(((EQ() + EQ()) * Delta() * (RQ(1) + Linear())).factor(3)), \
          'RQ(1) + Linear()'
    yield raises, IndexError, lambda: ((EQ() + EQ()) * Delta() *
                                       (RQ(1) + Linear())).factor(4)
