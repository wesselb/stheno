import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.plot
from varz.tensorflow import Vars, minimise_l_bfgs_b

from stheno.tensorflow import B, Graph, GP, Delta, EQ

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


def model(vs):
    g = Graph()

    # Construct model for first layer:
    f1 = GP(vs.pos(1., name='f1/var') *
            EQ().stretch(vs.pos(1., name='f1/scale')), graph=g)
    e1 = GP(vs.pos(0.1, name='e1/var') * Delta(), graph=g)
    y1 = f1 + e1

    # Construct model for second layer:
    f2 = GP(vs.pos(1., name='f2/var') *
            EQ().stretch(vs.pos(np.array([1., .5]), name='f2/scale')), graph=g)
    e2 = GP(vs.pos(0.1, name='e2/var') * Delta(), graph=g)
    y2 = f2 + e2

    return f1, y1, f2, y2


def objective(vs):
    f1, y1, f2, y2 = model(vs)

    x1 = x_obs1
    x2 = B.stack(x_obs2, B.take(y1_obs, inds2), axis=1)
    evidence = y1(x1).logpdf(y1_obs) + y2(x2).logpdf(y2_obs)

    return -evidence


# Learn hyperparameters.
vs = Vars(tf.float64)
minimise_l_bfgs_b(objective, vs)
f1, y1, f2, y2 = model(vs)

# Condition to make predictions.
x1 = x_obs1
x2 = B.stack(x_obs2, B.take(y1_obs, inds2), axis=1)
f1 = f1 | (y1(x1), y1_obs)
f2 = f2 | (y2(x2), y2_obs)

# Predict first output.
mean1, lower1, upper1 = f1(x).marginals()

# Predict second output with Monte Carlo.
samples = [f2(B.stack(x, f1(x).sample()[:, 0], axis=1)).sample()[:, 0]
           for _ in range(100)]
mean2 = np.mean(samples, axis=0)
lower2 = np.percentile(samples, 2.5, axis=0)
upper2 = np.percentile(samples, 100 - 2.5, axis=0)

# Plot result.
plt.figure()

plt.subplot(2, 1, 1)
plt.title('Output 1')
plt.plot(x, f1_true, label='True', c='tab:blue')
plt.scatter(x_obs1, y1_obs, label='Observations', c='tab:red')
plt.plot(x, mean1, label='Prediction', c='tab:green')
plt.plot(x, lower1, ls='--', c='tab:green')
plt.plot(x, upper1, ls='--', c='tab:green')
wbml.plot.tweak()

plt.subplot(2, 1, 2)
plt.title('Output 2')
plt.plot(x, f2_true, label='True', c='tab:blue')
plt.scatter(x_obs2, y2_obs, label='Observations', c='tab:red')
plt.plot(x, mean2, label='Prediction', c='tab:green')
plt.plot(x, lower2, ls='--', c='tab:green')
plt.plot(x, upper2, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig('readme_example7_gpar.png')
plt.show()
