import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.plot

from stheno.tensorflow import B, Measure, GP, EQ, Delta

# Define points to predict at.
x = B.linspace(tf.float64, 0, 10, 200)
x_obs = B.linspace(tf.float64, 0, 10, 10)

with Measure() as prior:
    # Construct a model.
    f = 0.7 * GP(EQ()).stretch(1.5)
    e = 0.2 * GP(Delta())

    # Construct derivatives.
    df = f.diff()
    ddf = df.diff()
    dddf = ddf.diff() + e

# Fix the integration constants.
zero = B.cast(tf.float64, 0)
one = B.cast(tf.float64, 1)
prior = prior | ((f(zero), one), (df(zero), zero), (ddf(zero), -one))

# Sample observations.
y_obs = B.sin(x_obs) + 0.2 * B.randn(*x_obs.shape)

# Condition on the observations to make predictions.
post = prior | (dddf(x_obs), y_obs)

# And make predictions.
pred_iiif = post(f)(x)
pred_iif = post(df)(x)
pred_if = post(ddf)(x)
pred_f = post(dddf)(x)


# Plot result.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label="True", style="test")
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
    mean, lower, upper = pred.marginal_credible_bounds()
    plt.plot(x, mean, label="Prediction", style="pred")
    plt.fill_between(x, lower, upper, style="pred")
    wbml.plot.tweak()


plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.title("Function")
plot_prediction(x, np.sin(x), pred_f, x_obs=x_obs, y_obs=y_obs)

plt.subplot(2, 2, 2)
plt.title("Integral of Function")
plot_prediction(x, -np.cos(x), pred_if)

plt.subplot(2, 2, 3)
plt.title("Second Integral of Function")
plot_prediction(x, -np.sin(x), pred_iif)

plt.subplot(2, 2, 4)
plt.title("Third Integral of Function")
plot_prediction(x, np.cos(x), pred_iiif)

plt.savefig("readme_example5_integration.png")
plt.show()
