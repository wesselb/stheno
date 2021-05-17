import matplotlib.pyplot as plt
from wbml.plot import tweak

from stheno import B, GP, EQ

# Define points to predict at.
x = B.linspace(0, 10, 100)
x_obs = B.linspace(0, 7, 20)

# Construct a prior.
f = GP(EQ().periodic(5.0))

# Sample a true, underlying function and noisy observations.
f_true, y_obs = f.measure.sample(f(x), f(x_obs, 0.5))

# Now condition on the observations to make predictions.
f_post = f | (f(x_obs, 0.5), y_obs)
mean, lower, upper = f_post(x).marginal_credible_bounds()

# Plot result.
plt.plot(x, f_true, label="True", style="test")
plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()
plt.savefig("readme_example1_simple_regression.png")
plt.show()
