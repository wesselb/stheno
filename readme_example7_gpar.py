import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from varz.spec import parametrised, Positive
from varz.tensorflow import Vars, minimise_l_bfgs_b
from wbml.plot import tweak

from stheno.tensorflow import B, GP, EQ

# Define points to predict at.
x = B.linspace(tf.float64, 0, 10, 200)
x_obs1 = B.linspace(tf.float64, 0, 10, 30)
inds2 = np.random.permutation(len(x_obs1))[:10]
x_obs2 = B.take(x_obs1, inds2)

# Construction functions to predict and observations.
f1_true = B.sin(x)
f2_true = B.sin(x) ** 2

y1_obs = B.sin(x_obs1) + 0.1 * B.randn(*x_obs1.shape)
y2_obs = B.sin(x_obs2) ** 2 + 0.1 * B.randn(*x_obs2.shape)


@parametrised
def model(
    vs,
    var1: Positive = 1,
    scale1: Positive = 1,
    noise1: Positive = 0.1,
    var2: Positive = 1,
    scale2: Positive = 1,
    noise2: Positive = 0.1,
):
    # Build layers:
    f1 = GP(var1 * EQ().stretch(scale1))
    f2 = GP(var2 * EQ().stretch(scale2))
    return (f1, noise1), (f2, noise2)


def objective(vs):
    (f1, noise1), (f2, noise2) = model(vs)
    x1 = x_obs1
    x2 = B.stack(x_obs2, B.take(y1_obs, inds2), axis=1)
    evidence = f1(x1, noise1).logpdf(y1_obs) + f2(x2, noise2).logpdf(y2_obs)
    return -evidence


# Learn hyperparameters.
vs = Vars(tf.float64)
minimise_l_bfgs_b(objective, vs)

# Compute posteriors.
(f1, noise1), (f2, noise2) = model(vs)
x1 = x_obs1
x2 = B.stack(x_obs2, B.take(y1_obs, inds2), axis=1)
f1_post = f1 | (f1(x1, noise1), y1_obs)
f2_post = f2 | (f2(x2, noise2), y2_obs)

# Predict first output.
mean1, lower1, upper1 = f1_post(x).marginal_credible_bounds()

# Predict second output with Monte Carlo.
samples = [
    f2_post(B.stack(x, f1_post(x).sample()[:, 0], axis=1)).sample()[:, 0]
    for _ in range(100)
]
mean2 = np.mean(samples, axis=0)
lower2 = np.percentile(samples, 2.5, axis=0)
upper2 = np.percentile(samples, 100 - 2.5, axis=0)

# Plot result.
plt.figure()

plt.subplot(2, 1, 1)
plt.title("Output 1")
plt.plot(x, f1_true, label="True", style="test")
plt.scatter(x_obs1, y1_obs, label="Observations", style="train", s=20)
plt.plot(x, mean1, label="Prediction", style="pred")
plt.fill_between(x, lower1, upper1, style="pred")
tweak()

plt.subplot(2, 1, 2)
plt.title("Output 2")
plt.plot(x, f2_true, label="True", style="test")
plt.scatter(x_obs2, y2_obs, label="Observations", style="train", s=20)
plt.plot(x, mean2, label="Prediction", style="pred")
plt.fill_between(x, lower2, upper2, style="pred")
tweak()

plt.savefig("readme_example7_gpar.png")
plt.show()
