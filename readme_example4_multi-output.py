import matplotlib.pyplot as plt
from wbml.plot import tweak

from stheno import B, Measure, GP, EQ, Delta


class VGP:
    """A vector-valued GP."""

    def __init__(self, ps):
        self.ps = ps

    def __add__(self, other):
        return VGP([f + g for f, g in zip(self.ps, other.ps)])

    def lmatmul(self, A):
        m, n = A.shape
        ps = [0 for _ in range(m)]
        for i in range(m):
            for j in range(n):
                ps[i] += A[i, j] * self.ps[j]
        return VGP(ps)


# Define points to predict at.
x = B.linspace(0, 10, 100)
x_obs = B.linspace(0, 10, 10)

# Model parameters:
m = 2
p = 4
H = B.randn(p, m)


with Measure() as prior:
    # Construct latent functions.
    us = VGP([GP(EQ()) for _ in range(m)])

    # Construct multi-output prior.
    fs = us.lmatmul(H)

    # Construct noise.
    e = VGP([GP(0.5 * Delta()) for _ in range(p)])

    # Construct observation model.
    ys = e + fs

# Sample a true, underlying function and observations.
samples = prior.sample(*(p(x) for p in fs.ps), *(p(x_obs) for p in ys.ps))
fs_true, ys_obs = samples[:p], samples[p:]

# Compute the posterior and make predictions.
post = prior.condition(*((p(x_obs), y_obs) for p, y_obs in zip(ys.ps, ys_obs)))
preds = [post(p(x)) for p in fs.ps]


# Plot results.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f, label="True", style="test")
    if x_obs is not None:
        plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
    mean, lower, upper = pred.marginal_credible_bounds()
    plt.plot(x, mean, label="Prediction", style="pred")
    plt.fill_between(x, lower, upper, style="pred")
    tweak()


plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title(f"Output {i + 1}")
    plot_prediction(x, fs_true[i], preds[i], x_obs, ys_obs[i])
plt.savefig("readme_example4_multi-output.png")
plt.show()
