import matplotlib.pyplot as plt
import wbml.out as out
from wbml.plot import tweak

from stheno import B, Measure, GP, Delta

# Define points to predict at.
x = B.linspace(0, 10, 200)
x_obs = B.linspace(0, 10, 10)

# Construct the model.
prior = Measure()
slope = GP(1, measure=prior)
intercept = GP(5, measure=prior)
f = slope * (lambda x: x) + intercept

e = 0.2 * GP(Delta(), measure=prior)  # Noise model

y = f + e  # Observation model

# Sample a slope, intercept, underlying function, and observations.
true_slope, true_intercept, f_true, y_obs = prior.sample(
    slope(0), intercept(0), f(x), y(x_obs)
)

# Condition on the observations to make predictions.
post = prior | (y(x_obs), y_obs)
mean, lower, upper = post(f(x)).marginals()

out.kv("True slope", true_slope[0, 0])
out.kv("Predicted slope", post(slope(0)).mean[0, 0])
out.kv("True intercept", true_intercept[0, 0])
out.kv("Predicted intercept", post(intercept(0)).mean[0, 0])

# Plot result.
plt.plot(x, f_true, label="True", style="test")
plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
plt.plot(x, mean, label="Prediction", style="pred")
plt.fill_between(x, lower, upper, style="pred")
tweak()

plt.savefig("readme_example6_blr.png")
plt.show()
