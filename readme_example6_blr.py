import matplotlib.pyplot as plt
import wbml.out as out
from wbml.plot import tweak

from stheno import B, Measure, GP

B.epsilon = 1e-10  # Very slightly regularise.

# Define points to predict at.
x = B.linspace(0, 10, 200)
x_obs = B.linspace(0, 10, 10)

with Measure() as prior:
    # Construct a linear model.
    slope = GP(1)
    intercept = GP(5)
    f = slope * (lambda x: x) + intercept

# Sample a slope, intercept, underlying function, and observations.
true_slope, true_intercept, f_true, y_obs = prior.sample(
    slope(0), intercept(0), f(x), f(x_obs, 0.2)
)

# Condition on the observations to make predictions.
post = prior | (f(x_obs, 0.2), y_obs)
mean, lower, upper = post(f(x)).marginal_credible_bounds()

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
