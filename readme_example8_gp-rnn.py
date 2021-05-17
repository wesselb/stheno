import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from varz.spec import parametrised, Positive
from varz.tensorflow import Vars, minimise_adam
from wbml.net import rnn as rnn_constructor
from wbml.plot import tweak

from stheno.tensorflow import B, Measure, GP, EQ

# Increase regularisation because we are dealing with `tf.float32`s.
B.epsilon = 1e-6

# Construct points which to predict at.
x = B.linspace(tf.float32, 0, 1, 100)[:, None]
inds_obs = B.range(0, int(0.75 * len(x)))  # Train on the first 75% only.
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


@parametrised
def model(vs, a_scale: Positive = 0.1, b_scale: Positive = 0.1, noise: Positive = 0.01):
    # Construct an RNN.
    f_rnn = rnn_constructor(
        output_size=1, widths=(10,), nonlinearity=B.tanh, final_dense=True
    )

    # Set the weights for the RNN.
    num_weights = f_rnn.num_weights(input_size=1)
    weights = Vars(tf.float32, source=vs.get(shape=(num_weights,), name="rnn"))
    f_rnn.initialise(input_size=1, vs=weights)

    with Measure():
        # Construct GPs that modulate the RNN.
        a = GP(1e-2 * EQ().stretch(a_scale))
        b = GP(1e-2 * EQ().stretch(b_scale))

        # GP-RNN model:
        f_gp_rnn = (1 + a) * (lambda x: f_rnn(x)) + b

    return f_rnn, f_gp_rnn, noise, a, b


def objective_rnn(vs):
    f_rnn, _, _, _, _ = model(vs)
    return B.mean((f_rnn(x_obs) - y_obs) ** 2)


def objective_gp_rnn(vs):
    _, f_gp_rnn, noise, _, _ = model(vs)
    evidence = f_gp_rnn(x_obs, noise).logpdf(y_obs)
    return -evidence


# Pretrain the RNN.
vs = Vars(tf.float32)
minimise_adam(objective_rnn, vs, rate=5e-3, iters=1000, trace=True, jit=True)

# Jointly train the RNN and GPs.
minimise_adam(objective_gp_rnn, vs, rate=1e-3, iters=1000, trace=True, jit=True)

_, f_gp_rnn, noise, a, b = model(vs)

# Condition.
post = f_gp_rnn.measure | (f_gp_rnn(x_obs, noise), y_obs)

# Predict and plot results.
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.title("$(1 + a)\\cdot {}$RNN${} + b$")
plt.plot(x, f_true, label="True", style="test")
plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
mean, lower, upper = post(f_gp_rnn(x)).marginal_credible_bounds()
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.subplot(2, 2, 3)
plt.title("$a$")
mean, lower, upper = post(a(x)).marginal_credible_bounds()
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.subplot(2, 2, 4)
plt.title("$b$")
mean, lower, upper = post(b(x)).marginal_credible_bounds()
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.savefig(f"readme_example8_gp-rnn.png")
plt.show()
