import matplotlib.pyplot as plt
import numpy as np
import wbml.plot

from stheno import GP, EQ, Delta, model, Obs

# Define points to predict at.
x = np.linspace(0, 10, 100)
x_obs = np.linspace(0, 10, 20)

# Constuct a prior:
w = lambda x: np.exp(-x ** 2 / 0.5)  # Window
b = [(GP(EQ()) * w).shift(xi) for xi in x_obs]  # Weighted basis functions
f = sum(b)  # Latent function
e = GP(Delta())  # Noise
y = f + 0.2 * e  # Observation model

# Sample a true, underlying function and observations.
f_true, y_obs = model.sample(f(x), y(x_obs))

# Condition on the observations to make predictions.
obs = Obs(y(x_obs), y_obs)
f, b = (f | obs, b | obs)

# Plot result.
for i, bi in enumerate(b):
    mean, lower, upper = bi(x).marginals()
    kw_args = {'label': 'Basis functions'} if i == 0 else {}
    plt.plot(x, mean, c='tab:orange', **kw_args)
plt.plot(x, f_true, label='True', c='tab:blue')
plt.scatter(x_obs, y_obs, label='Observations', c='tab:red')
mean, lower, upper = f(x).marginals()
plt.plot(x, mean, label='Prediction', c='tab:green')
plt.plot(x, lower, ls='--', c='tab:green')
plt.plot(x, upper, ls='--', c='tab:green')
wbml.plot.tweak()

plt.savefig('readme_example11_nonparametric_basis.png')
plt.show()
