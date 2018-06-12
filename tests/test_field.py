# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from stheno.field import mul, add, stretch, SumType, new, get_field, shift, \
    transform, select
from stheno.kernel import EQ, RQ, Linear, OneKernel, ZeroKernel, Delta
from stheno.mean import FunctionMean, ZeroMean, OneMean
# noinspection PyUnresolvedReferences
from tests import ok, raises
from . import eq


def test_exceptions():
    yield raises, IndexError, lambda: EQ().stretch(1)[1]
    yield raises, IndexError, lambda: (EQ() + RQ(1))[2]
    yield raises, RuntimeError, lambda: mul(1, 1)
    yield raises, RuntimeError, lambda: add(1, 1)
    yield raises, RuntimeError, lambda: stretch(1, 1)
    yield raises, RuntimeError, lambda: transform(1, 1)
    yield raises, RuntimeError, lambda: select(1, 1)
    yield raises, RuntimeError, lambda: shift(1, 1)
    yield raises, RuntimeError, lambda: get_field(1)
    yield raises, RuntimeError, lambda: new(1, SumType)
    yield eq, repr(EQ()), str(EQ())


def test_cancellations_zero():
    # With constants:
    yield eq, str(1 * EQ()), 'EQ()'
    yield eq, str(0 * EQ()), '0'
    yield eq, str(0 + EQ()), 'EQ()'
    yield eq, str(EQ() + 0), 'EQ()'
    yield eq, str(0 + OneMean()), '1'
    yield eq, str(OneMean() + 0), '1'

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

    # Shifts:
    yield eq, str(Linear().shift(5)), 'Linear() shift 5'
    yield eq, str(Linear().shift(5).shift(5)), 'Linear() shift 10'

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
    yield raises, IndexError, lambda: EQ().term(1)


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
    yield raises, IndexError, lambda: EQ().factor(1)


def test_shifting():
    # Kernels:
    yield eq, str(ZeroKernel().shift(5)), '0'
    yield eq, str(EQ().shift(5)), 'EQ()'
    yield eq, str(Linear().shift(5)), 'Linear() shift 5'
    yield eq, str((5 * EQ()).shift(5)), '5 * EQ()'
    yield eq, str((5 * Linear()).shift(5)), '(5 * Linear()) shift 5'

    # Means:
    def mean(x): return x

    m = FunctionMean(mean)
    yield eq, str(ZeroMean().shift(5)), '0'
    yield eq, str(m.shift(5)), 'mean shift 5'
    yield eq, str(m.shift(5).shift(5)), 'mean shift 10'
    yield eq, str((5 * m).shift(5)), '(5 * mean) shift 5'


def test_selection():
    yield eq, str(EQ().select(0)), 'EQ() : [0]'
    yield eq, str(EQ().select(0, 2)), 'EQ() : [0, 2]'

    yield eq, str(ZeroKernel().select(0)), '0'
    yield eq, str(OneMean().select(0)), '1'


def test_input_transform():
    yield eq, str(EQ().transform(lambda x: x)), 'EQ() transform <lambda>'
    yield eq, str(EQ().transform(lambda x: x, lambda x: x)), \
          'EQ() transform (<lambda>, <lambda>)'

    yield eq, str(ZeroKernel().transform(lambda x: x)), '0'
    yield eq, str(OneMean().transform(lambda x: x)), '1'
