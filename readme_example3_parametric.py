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
    noise: Positive = 0.5,
    alpha: Positive = 1.2,
):
    with Measure():
        # Random fluctuation:
        u = GP(u_var * EQ().stretch(u_scale))
        # Construct model.
        f = u + (lambda x: x ** alpha)
    return f, noise


# Sample a true, underlying function and observations.
vs = Vars(tf.float64)
f_true = x ** 1.8 + B.sin(2 * B.pi * x)
f, y = model(vs)
post = f.measure | (f(x), f_true)
y_obs = post(f(x_obs)).sample()


def objective(vs):
    f, noise = model(vs)
    evidence = f(x_obs, noise).logpdf(y_obs)
    return -evidence


# Learn hyperparameters.
minimise_l_bfgs_b(objective, vs, jit=True)
f, noise = model(vs)

# Print the learned parameters.
out.kv("Prior", f.display(out.format))
vs.print()

# Condition on the observations to make predictions.
f_post = f | (f(x_obs, noise), y_obs)
mean, lower, upper = f_post(x).marginal_credible_bounds()

# Plot result.
plt.plot(x, B.squeeze(f_true), label="True", style="test")
plt.scatter(x_obs, B.squeeze(y_obs), label="Observations", style="train", s=20)
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.savefig("readme_example3_parametric.png")
plt.show()
