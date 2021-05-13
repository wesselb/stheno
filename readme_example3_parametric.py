import matplotlib.pyplot as plt
import tensorflow as tf
import wbml.out as out
from varz.spec import parametrised, Positive
from varz.tensorflow import Vars, minimise_l_bfgs_b
from wbml.plot import tweak

from stheno.tensorflow import B, Measure, GP, EQ, Delta

# Define points to predict at.
x = B.linspace(tf.float64, 0, 5, 100)
x_obs = B.linspace(tf.float64, 0, 3, 20)


@parametrised
def model(
    vs,
    u_var: Positive = 0.5,
    u_scale: Positive = 0.5,
    e_var: Positive = 0.5,
    alpha: Positive = 1.2,
):
    prior = Measure()

    # Random fluctuation:
    u = GP(u_var * EQ().stretch(u_scale), measure=prior)

    # Noise:
    e = GP(e_var * Delta(), measure=prior)

    # Construct model:
    f = u + (lambda x: x ** alpha)
    y = f + e

    return f, y


# Sample a true, underlying function and observations.
vs = Vars(tf.float64)
f_true = x ** 1.8 + B.sin(2 * B.pi * x)
f, y = model(vs)
post = f.measure | (f(x), f_true)
y_obs = post(f(x_obs)).sample()


def objective(vs):
    f, y = model(vs)
    evidence = y(x_obs).logpdf(y_obs)
    return -evidence


# Learn hyperparameters.
minimise_l_bfgs_b(objective, vs, jit=True)
f, y = model(vs)

# Print the learned parameters.
out.kv("Prior", y.display(out.format))
vs.print()

# Condition on the observations to make predictions.
post = f.measure | (y(x_obs), y_obs)
mean, lower, upper = post(f(x)).marginals()

# Plot result.
plt.plot(x, B.squeeze(f_true), label="True", style="test")
plt.scatter(x_obs, B.squeeze(y_obs), label="Observations", style="train", s=20)
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.savefig("readme_example3_parametric.png")
plt.show()
