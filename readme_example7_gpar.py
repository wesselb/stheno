import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
from wbml import Vars

from stheno.tf import GP, Delta, EQ, Graph, B

s = tf.Session()

# Define points to predict at.
x = np.linspace(0, 10, 200)
x_obs1 = np.linspace(0, 10, 30)
inds2 = np.random.permutation(len(x_obs1))[:10]
x_obs2 = x_obs1[inds2]

# Construct variable storages.
vs1 = Vars(np.float64)
vs2 = Vars(np.float64)

# Construct a model for each output.
m1 = Graph()
m2 = Graph()
f1 = vs1.pos(1.) * GP(EQ(), graph=m1).stretch(vs1.pos(1.))
f2 = vs2.pos(1.) * GP(EQ(), graph=m2).stretch(vs2.pos([1., .5]))
sig1 = vs1.pos(0.1)
sig2 = vs2.pos(0.1)

# Initialise variables.
vs1.init(s)
vs2.init(s)

# Noise models:
e1 = sig1 * GP(Delta(), graph=m1)
e2 = sig2 * GP(Delta(), graph=m2)

# Observation models:
y1 = f1 + e1
y2 = f2 + e2

# Construction functions to predict and observations.
f1_true = np.sin(x)
f2_true = np.sin(x) ** 2

y1_obs = np.sin(x_obs1) + 0.1 * np.random.randn(*x_obs1.shape)
y2_obs = np.sin(x_obs2) ** 2 + 0.1 * np.random.randn(*x_obs2.shape)

# Learn.
lml1 = y1(x_obs1).logpdf(y1_obs)
SOI(-lml1, var_list=vs1.vars).minimize(s)

lml2 = y2(np.stack((x_obs2, y1_obs[inds2]), axis=1)).logpdf(y2_obs)
SOI(-lml2, var_list=vs2.vars).minimize(s)

# Predict first output.
f1 = f1 | (y1(x_obs1), y1_obs)
mean1, lower1, upper1 = s.run(f1.predict(x))

# Predict second output with Monte Carlo.
f2 = f2 | (y2(np.stack((x_obs2, y1_obs[inds2]), axis=1)), y2_obs)
sample = f2(B.concat([x[:, None], f1(x).sample()], axis=1)).sample()
samples = [s.run(sample).squeeze() for _ in range(100)]
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
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Output 2')
plt.plot(x, f2_true, label='True', c='tab:blue')
plt.scatter(x_obs2, y2_obs, label='Observations', c='tab:red')
plt.plot(x, mean2, label='Prediction', c='tab:green')
plt.plot(x, lower2, ls='--', c='tab:green')
plt.plot(x, upper2, ls='--', c='tab:green')
plt.legend()

plt.show()
