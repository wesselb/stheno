import matplotlib.pyplot as plt
import numpy as np

from stheno import GP, Delta, model, Unique, dense, B, Linear, Cache

import gc
gc.disable()

B.epsilon = 1e-8

# Define points to predict at.
x = Unique(np.linspace(0, 10, 10_000))
x_obs = Unique(np.linspace(-10, 5, 50_000))

# Construct the model.
slope = GP(1.)
intercept = GP(5.)
f = slope * (lambda x: x) + intercept

e = 0.2 * GP(Delta())  # Noise model

y = f + e  # Observation model

# Sample a slope, intercept, underlying function, and observations.
true_slope, true_intercept = model.sample(slope @ 0., intercept @ 0.)

slope_reg = slope + 1e-1 * GP(Delta())
intercept_reg = intercept + 1e-1 * GP(Delta())
f_reg = f + 1e-1 * GP(Delta())

print('conditioning')
model.condition(slope_reg @ Unique(0.), true_slope)
model.condition(intercept_reg @ Unique(0.), true_intercept)
print('sampling (1)')
f_true = f(x_obs).sample()
model.condition(f_reg @ x_obs, f_true)
print('sampling (2)')
y_obs = y(x_obs).sample()
model.revert_prior()

# Condition on the observations to make predictions.
print('conditioning and predicting')
mean, lower, upper = f.condition(y @ x_obs, y_obs).predict(x)
mean_slope, mean_intercept = slope(0).mean, intercept(0).mean

print('true slope', true_slope)
print('predicted slope', mean_slope)
print('true intercept', true_intercept)
print('predicted intercept', mean_intercept)

# Plot result.
exit()
plt.plot(x.get(), f_true, label='True', c='tab:blue')
plt.scatter(x_obs.get(), y_obs, label='Observations', c='tab:red')
plt.plot(x.get(), mean, label='Prediction', c='tab:green')
plt.plot(x.get(), lower, ls='--', c='tab:green')
plt.plot(x.get(), upper, ls='--', c='tab:green')
plt.legend()
plt.show()
