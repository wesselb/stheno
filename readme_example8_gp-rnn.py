import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.plot
from varz.tensorflow import Vars, minimise_adam
from wbml.net import rnn as rnn_constructor

from stheno.tensorflow import B, Graph, GP, Delta, EQ, Obs

# Increase regularisation because we are dealing with float32.
B.epsilon = 1e-6

# Construct points which to predict at.
x = B.linspace(tf.float32, 0, 1, 100)[:, None]
inds_obs = np.arange(0, int(0.75 * len(x)))  # Train on the first 75% only.
x_obs = B.take(x, inds_obs)

# Construct function and observations.
#   Draw random modulation functions.
a_true = GP(1e-2 * EQ().stretch(0.1))(x).sample()
b_true = GP(1e-2 * EQ().stretch(0.1))(x).sample()
#   Construct the true, underlying function.
f_true = (1 + a_true) * B.sin(2 * np.pi * 7 * x) + b_true
#   Add noise.
y_true = f_true + 0.1 * B.randn(*f_true.shape)

# Normalise and split.
f_true = (f_true - B.mean(y_true)) / B.std(y_true)
y_true = (y_true - B.mean(y_true)) / B.std(y_true)
y_obs = B.take(y_true, inds_obs)


def model(vs):
    g = Graph()

    # Construct an RNN.
    f_rnn = rnn_constructor(output_size=1,
                            widths=(10,),
                            nonlinearity=B.tanh,
                            final_dense=True)

    # Set the weights for the RNN.
    num_weights = f_rnn.num_weights(input_size=1)
    weights = Vars(tf.float32, source=vs.get(shape=(num_weights,), name='rnn'))
    f_rnn.initialise(input_size=1, vs=weights)

    # Construct GPs that modulate the RNN.
    a = GP(1e-2 * EQ().stretch(vs.pos(0.1, name='a/scale')), graph=g)
    b = GP(1e-2 * EQ().stretch(vs.pos(0.1, name='b/scale')), graph=g)
    e = GP(vs.pos(1e-2, name='e/var') * Delta(), graph=g)

    # GP-RNN model:
    f_gp_rnn = (1 + a) * (lambda x: f_rnn(x)) + b
    y_gp_rnn = f_gp_rnn + e

    return f_rnn, f_gp_rnn, y_gp_rnn, a, b


def objective_rnn(vs):
    f_rnn, _, _, _, _ = model(vs)
    return B.mean((f_rnn(x_obs) - y_obs) ** 2)


def objective_gp_rnn(vs):
    _, _, y_gp_rnn, _, _ = model(vs)
    evidence = y_gp_rnn(x_obs).logpdf(y_obs)
    return -evidence


# Pretrain the RNN.
vs = Vars(tf.float32)
minimise_adam(tf.function(objective_rnn, autograph=False),
              vs, rate=1e-2, iters=1000, trace=True)

# Jointly train the RNN and GPs.
minimise_adam(tf.function(objective_gp_rnn, autograph=False),
              vs, rate=1e-3, iters=1000, trace=True)

_, f_gp_rnn, y_gp_rnn, a, b = model(vs)

# Condition.
f_gp_rnn, a, b = (f_gp_rnn, a, b) | Obs(y_gp_rnn(x_obs), y_obs)

# Predict and plot results.
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.title('$(1 + a)\\cdot {}$RNN${} + b$')
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
mean, lower, upper = f_gp_rnn(x).marginals()
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.subplot(2, 2, 3)
plt.title('$a$')
mean, lower, upper = a(x).marginals()
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.subplot(2, 2, 4)
plt.title('$b$')
mean, lower, upper = b(x).marginals()
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig(f'readme_example8_gp-rnn.png')
plt.show()
