import matplotlib.pyplot as plt
import tensorflow as tf
import wbml.out
import wbml.plot
from varz.tensorflow import Vars, minimise_l_bfgs_b

from stheno.tensorflow import B, Graph, GP, EQ, Delta

# Define points to predict at.
x = B.linspace(tf.float64, 0, 5, 100)
x_obs = B.linspace(tf.float64, 0, 3, 20)


def model(vs):
    g = Graph()

    # Random fluctuation:
    u = GP(vs.pos(.5, name='u/var') *
           EQ().stretch(vs.pos(0.5, name='u/scale')), graph=g)

    # Noise:
    e = GP(vs.pos(0.5, name='e/var') * Delta(), graph=g)

    # Construct model:
    alpha = vs.pos(1.2, name='alpha')
    f = u + (lambda x: x ** alpha)
    y = f + e

    return f, y


# Sample a true, underlying function and observations.
vs = Vars(tf.float64)
f_true = x ** 1.8 + B.sin(2 * B.pi * x)
f, y = model(vs)
y_obs = (y | (f(x), f_true))(x_obs).sample()


def objective(vs):
    f, y = model(vs)
    evidence = y(x_obs).logpdf(y_obs)
    return -evidence


# Learn hyperparameters.
minimise_l_bfgs_b(tf.function(objective, autograph=False), vs)
f, y = model(vs)

# Print the learned parameters.
wbml.out.kv('Alpha', vs['alpha'])
wbml.out.kv('Prior', y.display(wbml.out.format))

# Condition on the observations to make predictions.
mean, lower, upper = (f | (y(x_obs), y_obs))(x).marginals()

# Plot result.
plt.plot(x, B.squeeze(f_true), label='True', c='tab:blue')
plt.scatter(x_obs, B.squeeze(y_obs), label='Observations', c='tab:red')
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig('readme_example3_parametric.png')
plt.show()
