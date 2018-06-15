import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
from wbml import Vars, rnn as rnn_constructor

from stheno.tf import GP, Delta, model, EQ, RQ

# Construct variable storages.
vs_gp = Vars(np.float32)
vs_rnn = Vars(np.float32)

# Construct a 1-layer RNN with GRUs.
f_rnn = rnn_constructor(1, 1, (10,))
f_rnn.initialise(vs_rnn)


# Wrap the RNN to be compatible with Stheno.
def rnn(x):
    return f_rnn(x[:, :, None])[:, :, 0]


# Construct session.
s = tf.Session()

# Construct points which to predict at.
x = np.linspace(0, 1, 100, dtype=np.float32)
inds_obs = np.arange(0, int(0.75 * len(x)))  # Train on the first 75% only.
x_obs = x[inds_obs]

# Construct function and observations.
#   Draw a random fluctuation.
k_u = .2 * RQ(1e-1).stretch(0.05)
u = s.run(GP(k_u)(np.array(x, dtype=np.float64)).sample()).squeeze()
#   Construct the true, underlying function.
f_true = np.sin(2 * np.pi * 7 * x) + np.array(u, dtype=np.float32)
#   Add noise.
y_true = f_true + 0.2 * np.array(np.random.randn(*x.shape), dtype=np.float32)

# Normalise and split.
f_true = (f_true - np.mean(y_true)) / np.std(y_true)
y_true = (y_true - np.mean(y_true)) / np.std(y_true)
y_obs = y_true[inds_obs]

# Construct the model.
a = vs_gp.pos(1.0) * GP(EQ()).stretch(vs_gp.pos(0.1))
b = vs_gp.pos(1.0) * GP(EQ()).stretch(vs_gp.pos(0.1))
e = vs_gp.pos(0.2) * GP(Delta())

# RNN-only model:
y_rnn = rnn + e

# GP-RNN model:
f_gp_rnn = (1 + a) * rnn + b
y_gp_rnn = f_gp_rnn + e

# Construct evidences.
lml_rnn = y_rnn(x_obs).log_pdf(y_obs)
lml_gp_rnn = y_gp_rnn(x_obs).log_pdf(y_obs)

# Construct optimisers and initialise.
opt_rnn = tf.train.AdamOptimizer(1e-2).minimize(
    -lml_rnn, var_list=vs_rnn.vars
)
opt_gp = SOI(-lml_gp_rnn,
             options={'disp': True, 'maxiter': 10},
             var_list=vs_gp.vars)
opt_jointly = tf.train.AdamOptimizer(1e-3).minimize(
    -lml_gp_rnn, var_list=vs_rnn.vars + vs_gp.vars
)
s.run(tf.global_variables_initializer())

# Pre-train the RNN.
for i in range(500):
    _, val = s.run([opt_rnn, lml_rnn])
    if i % 100 == 0:
        print(i, val)

# Pre-train the GPs.
opt_gp.minimize(s)

# Jointly train the RNN and GPs.
for i in range(5000):
    _, val = s.run([opt_jointly, lml_gp_rnn])
    if i % 100 == 0:
        print(i, val)

# Condition.
model.condition(y_gp_rnn @ x_obs, y_obs)

# Predict and plot results.
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.title('$(1 + a) \\cdot $ RNN ${}+b$')
plt.plot(x, f_true.squeeze(), label='True', c='tab:blue')
plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
mean, lower, upper = s.run(f_gp_rnn.predict(x))
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()

plt.subplot(2, 2, 3)
plt.title('$a$')
mean, lower, upper = s.run(a.predict(x))
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()

plt.subplot(2, 2, 4)
plt.title('$b$')
mean, lower, upper = s.run(b.predict(x))
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
plt.legend()

plt.show()
