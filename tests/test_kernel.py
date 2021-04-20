import numpy as np
import pytest
import tensorflow as tf
from lab.tensorflow import B
from matrix import Diagonal, Zero

from stheno.input import Input, Unique, WeightedUnique, MultiInput
from stheno.kernel import (
    EQ,
    RQ,
    Matern12,
    Matern32,
    Matern52,
    Delta,
    FixedDelta,
    Kernel,
    Linear,
    OneKernel,
    ZeroKernel,
    PosteriorKernel,
    ShiftedKernel,
    TensorProductKernel,
    CorrectiveKernel,
    DecayingKernel,
    LogKernel,
    perturb,
    num_elements,
)
from stheno.measure import FDD
from .util import approx


def standard_kernel_tests(k, shapes=None, dtype=np.float64):
    if shapes is None:
        shapes = [
            ((10, 2), (5, 2)),
            ((10, 1), (5, 1)),
            ((10,), (5, 1)),
            ((10, 1), (5,)),
            ((10,), (5,)),
            ((10,), ()),
            ((), (5,)),
            ((), ()),
        ]

    # Check various shapes of arguments.
    for shape1, shape2 in shapes:
        x1 = B.randn(dtype, *shape1)
        x2 = B.randn(dtype, *shape2)

        # Check that the kernel computes consistently.
        approx(k(x1, x2), B.transpose(reversed(k)(x2, x1)))

        # Check `elwise`.
        x2 = B.randn(dtype, *shape1)
        approx(k.elwise(x1, x2), B.diag(k(x1, x2))[:, None])
        # Check againtst fallback brute force computation.
        approx(k.elwise(x1, x2), Kernel.elwise(k, x1, x2))
        # The element-wise computation is more accurate, which is why we allow a
        # discrepancy a bit larger than the square root of the machine epsilon.
        approx(k.elwise(x1)[:, 0], B.diag(k(x1)), atol=1e-6, rtol=1e-6)
        approx(k.elwise(x1), Kernel.elwise(k, x1))


def test_corner_cases():
    with pytest.raises(RuntimeError):
        Kernel()(1.0)


@pytest.mark.parametrize(
    "x1", [B.randn(10), Input(B.randn(10)), FDD(None, B.randn(10))]
)
@pytest.mark.parametrize(
    "x2", [B.randn(10), Input(B.randn(10)), FDD(None, B.randn(10))]
)
def test_construction(x1, x2):
    k = EQ()

    k(x1)
    k(x1, x2)

    k.elwise(x1)
    k.elwise(x1, x2)

    # Test `MultiInput` construction.
    approx(
        k(MultiInput(x1, x2)),
        B.concat2d([k(x1, x1), k(x1, x2)], [k(x2, x1), k(x2, x2)]),
    )
    approx(k(x1, MultiInput(x1, x2)), B.concat(k(x1, x1), k(x1, x2), axis=1))
    approx(k(MultiInput(x1, x2), x2), B.concat(k(x1, x2), k(x2, x2), axis=0))

    approx(
        k.elwise(MultiInput(x1, x2)),
        B.concat(k.elwise(x1, x1), k.elwise(x2, x2), axis=0),
    )
    with pytest.raises(ValueError):
        k.elwise(MultiInput(x1), MultiInput(x1, x2))
    with pytest.raises(ValueError):
        k.elwise(x1, MultiInput(x1, x2))
    with pytest.raises(ValueError):
        k.elwise(MultiInput(x1, x2), x2)


def test_basic_arithmetic():
    k1 = EQ()
    k2 = RQ(1e-1)
    k3 = Matern12()
    k4 = Matern32()
    k5 = Matern52()
    k6 = Delta()
    k7 = Linear()
    xs1 = B.randn(10, 2), B.randn(20, 2)
    xs2 = B.randn(), B.randn()

    approx(k6(xs1[0]), k6(xs1[0], xs1[0]))
    approx((k1 * k2)(*xs1), k1(*xs1) * k2(*xs1))
    approx((k1 * k2)(*xs2), k1(*xs2) * k2(*xs2))
    approx((k3 + k4)(*xs1), k3(*xs1) + k4(*xs1))
    approx((k3 + k4)(*xs2), k3(*xs2) + k4(*xs2))
    approx((5.0 * k5)(*xs1), 5.0 * k5(*xs1))
    approx((5.0 * k5)(*xs2), 5.0 * k5(*xs2))
    approx((5.0 + k7)(*xs1), 5.0 + k7(*xs1))
    approx((5.0 + k7)(*xs2), 5.0 + k7(*xs2))
    approx(k1.stretch(2.0)(*xs1), k1(xs1[0] / 2.0, xs1[1] / 2.0))
    approx(k1.stretch(2.0)(*xs2), k1(xs2[0] / 2.0, xs2[1] / 2.0))
    approx(k1.periodic(1.0)(*xs1), k1.periodic(1.0)(xs1[0], xs1[1] + 5.0))
    approx(k1.periodic(1.0)(*xs2), k1.periodic(1.0)(xs2[0], xs2[1] + 5.0))


def test_reversal():
    x1 = B.randn(10, 2)
    x2 = B.randn(5, 2)
    x3 = B.randn()

    # Test with a stationary and non-stationary kernel.
    for k in [EQ(), Linear()]:
        approx(k(x1), reversed(k)(x1))
        approx(k(x3), reversed(k)(x3))
        approx(k(x1, x2), reversed(k)(x1, x2))
        approx(k(x1, x2), reversed(k)(x2, x1).T)

        # Test double reversal does the right thing.
        approx(k(x1), reversed(reversed(k))(x1))
        approx(k(x3), reversed(reversed(k))(x3))
        approx(k(x1, x2), reversed(reversed(k))(x1, x2))
        approx(k(x1, x2), reversed(reversed(k))(x2, x1).T)

    # Verify that the kernel has the right properties.
    k = reversed(EQ())
    assert k.stationary

    k = reversed(Linear())
    assert not k.stationary
    assert str(k) == "Reversed(Linear())"

    # Check equality.
    assert reversed(Linear()) == reversed(Linear())
    assert reversed(Linear()) != Linear()
    assert reversed(Linear()) != reversed(EQ())
    assert reversed(Linear()) != reversed(DecayingKernel(1, 1))

    # Standard tests:
    standard_kernel_tests(k)


def test_delta_properties():
    k = Delta()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "Delta()"

    # Check equality.
    assert Delta() == Delta()
    assert Delta() != Delta(epsilon=k.epsilon * 10)
    assert Delta() != EQ()


@pytest.mark.parametrize(
    "x1, w1, x2, w2",
    [
        (np.array(0), np.ones(1), np.array(1), np.ones(1)),
        (B.randn(10), np.ones(10), B.randn(5), np.ones(5)),
        (B.randn(10, 1), np.ones(10), B.randn(5, 1), np.ones(5)),
        (B.randn(10, 2), np.ones(10), B.randn(5, 2), np.ones(5)),
    ],
)
def test_delta_evaluations(x1, w1, x2, w2):
    k = Delta()
    n1 = num_elements(x1)
    n2 = num_elements(x2)

    # Check uniqueness checks.
    approx(k(x1), B.eye(n1))
    approx(k(x1, x2), B.zeros(n1, n2))

    # Standard tests:
    standard_kernel_tests(k)

    # Test `Unique` inputs.
    assert isinstance(k(Unique(x1), Unique(x1.copy())), Zero)
    assert isinstance(k(Unique(x1), Unique(x1)), Diagonal)
    assert isinstance(k(Unique(x1), x1), Zero)
    assert isinstance(k(x1, Unique(x1)), Zero)

    approx(k.elwise(Unique(x1), Unique(x1.copy())), B.zeros(n1, 1))
    approx(k.elwise(Unique(x1), Unique(x1)), B.ones(n1, 1))
    approx(k.elwise(Unique(x1), x1), B.zeros(n1, 1))

    # Test `WeightedUnique` inputs.
    assert isinstance(k(WeightedUnique(x1, w1), WeightedUnique(x1.copy(), w1)), Zero)
    assert isinstance(k(WeightedUnique(x1, w1), WeightedUnique(x1, w1)), Diagonal)
    assert isinstance(k(WeightedUnique(x1, w1), x1), Zero)
    assert isinstance(k(x1, WeightedUnique(x1, w1)), Zero)

    approx(
        k.elwise(WeightedUnique(x1, w1), WeightedUnique(x1.copy(), w1)), B.zeros(n1, 1)
    )
    approx(k.elwise(WeightedUnique(x1, w1), WeightedUnique(x1, w1)), B.ones(n1, 1))
    approx(k.elwise(WeightedUnique(x1, w1), x1), B.zeros(n1, 1))
    approx(k.elwise(x1, WeightedUnique(x1, w1)), B.zeros(n1, 1))
    approx(k.elwise(x1, WeightedUnique(x1, w1)), B.zeros(n1, 1))


def test_fixed_delta():
    noises = B.rand(3)
    k = FixedDelta(noises)

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "FixedDelta()"

    # Check equality.
    assert FixedDelta(noises) == FixedDelta(noises)
    assert FixedDelta(noises) != FixedDelta(2 * noises)
    assert FixedDelta(noises) != EQ()

    # Standard tests:
    standard_kernel_tests(k)

    # Check correctness.
    x1 = B.randn(5)
    x2 = B.randn(5)
    approx(k(x1), B.zeros(5, 5))
    approx(k.elwise(x1), B.zeros(5, 1))
    approx(k(x1, x2), B.zeros(5, 5))
    approx(k.elwise(x1, x2), B.zeros(5, 1))

    x1 = B.randn(3)
    x2 = B.randn(3)
    approx(k(x1), B.diag(noises))
    approx(k.elwise(x1), B.uprank(noises))
    approx(k(x1, x2), B.zeros(3, 3))
    approx(k.elwise(x1, x2), B.zeros(3, 1))


def test_eq():
    k = EQ()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "EQ()"

    # Test equality.
    assert EQ() == EQ()
    assert EQ() != Linear()

    # Standard tests:
    standard_kernel_tests(k)


def test_rq():
    k = RQ(1e-1)

    # Verify that the kernel has the right properties.
    assert k.alpha == 1e-1
    assert k.stationary
    assert str(k) == "RQ(0.1)"

    # Test equality.
    assert RQ(1e-1) == RQ(1e-1)
    assert RQ(1e-1) != RQ(2e-1)
    assert RQ(1e-1) != Linear()

    # Standard tests:
    standard_kernel_tests(k)


def test_exp():
    k = Matern12()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "Exp()"

    # Test equality.
    assert Matern12() == Matern12()
    assert Matern12() != Linear()

    # Standard tests:
    standard_kernel_tests(k)


def test_mat32():
    k = Matern32()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "Matern32()"

    # Test equality.
    assert Matern32() == Matern32()
    assert Matern32() != Linear()

    # Standard tests:
    standard_kernel_tests(k)


def test_mat52():
    k = Matern52()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "Matern52()"

    # Test equality.
    assert Matern52() == Matern52()
    assert Matern52() != Linear()

    # Standard tests:
    standard_kernel_tests(k)


def test_one():
    k = OneKernel()

    x1 = B.randn(10, 2)
    x2 = B.randn(5, 2)

    # Test that the kernel computes correctly.
    approx(k(x1, x2), np.ones((10, 5)))

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "1"

    # Test equality.
    assert OneKernel() == OneKernel()
    assert OneKernel() != Linear()

    # Standard tests:
    standard_kernel_tests(k)


def test_zero():
    k = ZeroKernel()
    x1 = B.randn(10, 2)
    x2 = B.randn(5, 2)

    # Test that the kernel computes correctly.
    approx(k(x1, x2), np.zeros((10, 5)))

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "0"

    # Test equality.
    assert ZeroKernel() == ZeroKernel()
    assert ZeroKernel() != Linear()

    # Standard tests:
    standard_kernel_tests(k)


def test_linear():
    k = Linear()

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == "Linear()"

    # Test equality.
    assert Linear() == Linear()
    assert Linear() != EQ()

    # Standard tests:
    standard_kernel_tests(k)


def test_decaying_kernel():
    k = DecayingKernel(3.0, 4.0)

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == "DecayingKernel(3.0, 4.0)"

    # Test equality.
    assert DecayingKernel(3.0, 4.0) == DecayingKernel(3.0, 4.0)
    assert DecayingKernel(3.0, 4.0) != DecayingKernel(3.0, 5.0)
    assert DecayingKernel(3.0, 4.0) != DecayingKernel(4.0, 4.0)
    assert DecayingKernel(3.0, 4.0) != EQ()

    # Standard tests:
    standard_kernel_tests(k)


def test_log_kernel():
    k = LogKernel()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "LogKernel()"

    # Test equality.
    assert LogKernel() == LogKernel()
    assert LogKernel() != EQ()

    # Standard tests:
    standard_kernel_tests(k)


def test_posterior_kernel():
    k = PosteriorKernel(EQ(), EQ(), EQ(), B.randn(5, 2), EQ()(B.randn(5, 1)))

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == "PosteriorKernel()"

    # Standard tests:
    standard_kernel_tests(k, shapes=[((10, 2), (5, 2))])


def test_corrective_kernel():
    a, b = B.randn(3, 3), B.randn(3, 3)
    a, b = a.dot(a.T), b.dot(b.T)
    z = B.randn(3, 2)
    k = CorrectiveKernel(EQ(), EQ(), z, a, b)

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == "CorrectiveKernel()"

    # Standard tests:
    standard_kernel_tests(k, shapes=[((10, 2), (5, 2))])


def test_sum():
    k1 = EQ().stretch(2)
    k2 = 3 * RQ(1e-2).stretch(5)
    k = k1 + k2

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert EQ() + Linear() == EQ() + Linear()
    assert EQ() + Linear() == Linear() + EQ()
    assert EQ() + Linear() != EQ() + RQ(1e-1)
    assert EQ() + Linear() != RQ(1e-1) + Linear()

    # Standard tests:
    standard_kernel_tests(k)


def test_product():
    k = (2 * EQ().stretch(10)) * (3 * RQ(1e-2).stretch(20))

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert EQ() * Linear() == EQ() * Linear()
    assert EQ() * Linear() == Linear() * EQ()
    assert EQ() * Linear() != EQ() * RQ(1e-1)
    assert EQ() * Linear() != RQ(1e-1) * Linear()

    # Standard tests:
    standard_kernel_tests(k)


def test_stretched():
    k = EQ().stretch(2)

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert EQ().stretch(2) == EQ().stretch(2)
    assert EQ().stretch(2) != EQ().stretch(3)
    assert EQ().stretch(2) != Matern12().stretch(2)

    # Standard tests:
    standard_kernel_tests(k)

    k = EQ().stretch(1, 2)

    # Verify that the kernel has the right properties.
    assert not k.stationary

    # Check passing in a list.
    k = EQ().stretch(np.array([1, 2]))
    k(B.randn(10, 2))


def test_periodic():
    k = EQ().stretch(2).periodic(3)

    # Verify that the kernel has the right properties.
    assert str(k) == "(EQ() > 2) per 3"
    assert k.stationary

    # Test equality.
    assert EQ().periodic(2) == EQ().periodic(2)
    assert EQ().periodic(2) != EQ().periodic(3)
    assert Matern12().periodic(2) != EQ().periodic(2)

    # Standard tests:
    standard_kernel_tests(k)

    k = 5 * k.stretch(5)

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Check passing in a list.
    k = EQ().periodic(np.array([1, 2]))
    k(B.randn(10, 2))

    # Check periodication of a zero.
    k = ZeroKernel()
    assert k.periodic(3) is k


def test_scaled():
    k = 2 * EQ()

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert 2 * EQ() == 2 * EQ()
    assert 2 * EQ() != 3 * EQ()
    assert 2 * EQ() != 2 * Matern12()

    # Standard tests:
    standard_kernel_tests(k)


def test_shifted():
    k = ShiftedKernel(2 * EQ(), 5)

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert Linear().shift(2) == Linear().shift(2)
    assert Linear().shift(2) != Linear().shift(3)
    assert Linear().shift(2) != DecayingKernel(1, 1).shift(2)

    # Standard tests:
    standard_kernel_tests(k)

    k = (2 * EQ()).shift(5, 6)

    # Verify that the kernel has the right properties.
    assert not k.stationary

    # Check computation.
    x1 = B.randn(10, 2)
    x2 = B.randn(5, 2)
    k = Linear()
    approx(k.shift(5)(x1, x2), k(x1 - 5, x2 - 5))

    # Check passing in a list.
    k = Linear().shift(np.array([1, 2]))
    k(B.randn(10, 2))


def test_selection():
    k = (2 * EQ().stretch(5)).select(0)

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert EQ().select(0) == EQ().select(0)
    assert EQ().select(0) != EQ().select(1)
    assert EQ().select(0) != Matern12().select(0)

    # Standard tests:
    standard_kernel_tests(k)

    # Verify that the kernel has the right properties.
    k = (2 * EQ().stretch(5)).select([2, 3])
    assert k.stationary

    k = (2 * EQ().stretch(np.array([1, 2, 3]))).select([0, 2])
    assert k.stationary

    k = (2 * EQ().periodic(np.array([1, 2, 3]))).select([1, 2])
    assert k.stationary

    k = (2 * EQ().stretch(np.array([1, 2, 3]))).select([0, 2], [1, 2])
    assert not k.stationary

    k = (2 * EQ().periodic(np.array([1, 2, 3]))).select([0, 2], [1, 2])
    assert not k.stationary

    # Test computation of the kernel.
    k1 = EQ().select([1, 2])
    k2 = EQ()
    x = B.randn(10, 3)
    approx(k1(x), k2(x[:, [1, 2]]))


def test_input_transform():
    k = Linear().transform(lambda x: x - 5)

    # Verify that the kernel has the right properties.
    assert not k.stationary

    def f1(x):
        return x

    def f2(x):
        return x ** 2

    # Test equality.
    assert EQ().transform(f1) == EQ().transform(f1)
    assert EQ().transform(f1) != EQ().transform(f2)
    assert EQ().transform(f1) != Matern12().transform(f1)

    # Standard tests:
    standard_kernel_tests(k)

    # Test computation of the kernel.
    k = Linear()
    x1, x2 = B.randn(10, 2), B.randn(10, 2)

    k2 = k.transform(lambda x: x ** 2)
    k3 = k.transform(lambda x: x ** 2, lambda x: x - 5)

    approx(k(x1 ** 2, x2 ** 2), k2(x1, x2))
    approx(k(x1 ** 2, x2 - 5), k3(x1, x2))


def test_tensor_product():
    k = TensorProductKernel(lambda x: B.sum(x ** 2, axis=1))

    # Verify that the kernel has the right properties.
    assert not k.stationary

    # Test equality.
    assert k == k
    assert k != TensorProductKernel(lambda x: x)
    assert k != EQ()

    # Standard tests:
    standard_kernel_tests(k)

    # Test computation of the kernel.
    k = TensorProductKernel(lambda x: x)
    x1 = np.linspace(0, 1, 100)[:, None]
    x2 = np.linspace(0, 1, 50)[:, None]
    approx(k(x1), x1 * x1.T)
    approx(k(x1, x2), x1 * x2.T)

    k = TensorProductKernel(lambda x: x ** 2)
    approx(k(x1), x1 ** 2 * (x1 ** 2).T)
    approx(k(x1, x2), (x1 ** 2) * (x2 ** 2).T)


def test_derivative():
    k = EQ().diff(0)

    # Check that the kernel has the right properties.
    assert not k.stationary

    # Test equality.
    assert EQ().diff(0) == EQ().diff(0)
    assert EQ().diff(0) != EQ().diff(1)
    assert Matern12().diff(0) != EQ().diff(0)

    # Standard tests:
    for k in [EQ().diff(0), EQ().diff(None, 0), EQ().diff(0, None)]:
        standard_kernel_tests(k, dtype=tf.float64)

    # Check that a derivative must be specified.
    with pytest.raises(RuntimeError):
        EQ().diff(None, None)(np.array([1.0]))
    with pytest.raises(RuntimeError):
        EQ().diff(None, None).elwise(np.array([1.0]))


def test_derivative_eq():
    # Test derivative of kernel `EQ()`.
    k = EQ()
    x1 = B.randn(tf.float64, 10, 1)
    x2 = B.randn(tf.float64, 5, 1)

    # Test derivative with respect to first input.
    approx(k.diff(0, None)(x1, x2), -k(x1, x2) * (x1 - B.transpose(x2)))
    approx(k.diff(0, None)(x1), -k(x1) * (x1 - B.transpose(x1)))

    # Test derivative with respect to second input.
    approx(k.diff(None, 0)(x1, x2), -k(x1, x2) * (B.transpose(x2) - x1))
    approx(k.diff(None, 0)(x1), -k(x1) * (B.transpose(x1) - x1))

    # Test derivative with respect to both inputs.
    ref = k(x1, x2) * (1 - (x1 - B.transpose(x2)) ** 2)
    approx(k.diff(0, 0)(x1, x2), ref)
    approx(k.diff(0)(x1, x2), ref)
    ref = k(x1) * (1 - (x1 - B.transpose(x1)) ** 2)
    approx(k.diff(0, 0)(x1), ref)
    approx(k.diff(0)(x1), ref)


def test_derivative_linear():
    # Test derivative of kernel `Linear()`.
    k = Linear()
    x1 = B.randn(tf.float64, 10, 1)
    x2 = B.randn(tf.float64, 5, 1)

    # Test derivative with respect to first input.
    approx(k.diff(0, None)(x1, x2), B.ones(tf.float64, 10, 5) * B.transpose(x2))
    approx(k.diff(0, None)(x1), B.ones(tf.float64, 10, 10) * B.transpose(x1))

    # Test derivative with respect to second input.
    approx(k.diff(None, 0)(x1, x2), B.ones(tf.float64, 10, 5) * x1)
    approx(k.diff(None, 0)(x1), B.ones(tf.float64, 10, 10) * x1)

    # Test derivative with respect to both inputs.
    ref = B.ones(tf.float64, 10, 5)
    approx(k.diff(0, 0)(x1, x2), ref)
    approx(k.diff(0)(x1, x2), ref)
    ref = B.ones(tf.float64, 10, 10)
    approx(k.diff(0, 0)(x1), ref)
    approx(k.diff(0)(x1), ref)


@pytest.mark.parametrize(
    "x,result",
    [
        (np.float64([1]), np.float64([1e-20 + 1 + 1e-14])),
        (np.float32([1]), np.float32([1e-20 + 1 + 1e-7])),
    ],
)
def test_perturb(x, result):
    assert perturb(x) == result  # Test NumPy.
    assert perturb(tf.constant(x)).numpy() == result  # Test TF.


def test_perturb_type_check():
    with pytest.raises(ValueError):
        perturb(0)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_nested_derivatives(dtype):
    x = B.randn(dtype, 10, 2)

    res = EQ().diff(0, 0).diff(0, 0)(x)
    assert ~B.isnan(res[0, 0])

    res = EQ().diff(1, 1).diff(1, 1)(x)
    assert ~B.isnan(res[0, 0])
