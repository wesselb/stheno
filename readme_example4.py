import matplotlib.pyplot as plt
import numpy as np
from plum import Dispatcher, Referentiable, Self

from stheno import GP, EQ, Delta, model, Kernel, B


class VGP(Referentiable):
    """A vector-valued GP.

    Args:
        dim (int): Dimensionality.
        kernel (instance of :class:`stheno.kernel.Kernel`): Kernel.
    """
    dispatch = Dispatcher(in_class=Self)

    @dispatch(int, Kernel)
    def __init__(self, dim, kernel):
        self.ps = [GP(kernel) for _ in range(dim)]

    @dispatch([GP])
    def __init__(self, *ps):
        self.ps = ps

    @dispatch(Self)
    def __add__(self, other):
        return VGP(*[f + g for f, g in zip(self.ps, other.ps)])

    @dispatch(np.ndarray)
    def lmatmul(self, A):
        m, n = A.shape
        ps = [0 for i in range(m)]
        for i in range(m):
            for j in range(n):
                ps[i] += A[i, j] * self.ps[j]
        return VGP(*ps)

    def sample(self, x):
        return model.sample(*(p @ x for p in self.ps))

    def condition(self, x, ys):
        model.condition(*((p @ x, y) for p, y in zip(self.ps, ys)))
        return self

    def predict(self, x):
        return [p.predict(x) for p in self.ps]


# Define points to predict at.
x = np.linspace(0, 10, 100)
x_obs = np.linspace(0, 10, 10)

# Model parameters:
m = 2
p = 4
H = np.random.randn(p, m)

# Construct latent functions
us = VGP(m, EQ())
fs = us.lmatmul(H)

# Construct noise.
e = VGP(p, 0.5 * Delta())

# Construct observation model.
ys = e + fs

# Sample observations and a true, underlying function.
ys_obs = ys.sample(x_obs)
ys.condition(x_obs, ys_obs)
fs_true = fs.sample(x)
model.revert_prior()

# Condition the model on the observations to make predictions.
ys.condition(x_obs, ys_obs)
preds = fs.predict(x)


# Plot results.
def plot_prediction(x, f, pred, x_obs=None, y_obs=None):
    plt.plot(x, f.squeeze(), label='True', c='tab:blue')
    if x_obs is not None:
        plt.scatter(x_obs, y_obs.squeeze(), label='Observations', c='tab:red')
    mean, lower, upper = pred
    plt.plot(x, mean, label='Prediction', c='tab:green')
    plt.plot(x, lower, ls='--', c='tab:green')
    plt.plot(x, upper, ls='--', c='tab:green')
    plt.legend()


plt.figure(figsize=(10, 6))

for i in range(p):
    plt.subplot(int(p ** .5), int(p ** .5), i + 1)
    plt.title('Output {}'.format(i + 1))
    plot_prediction(x, fs_true[i], preds[i], x_obs, ys_obs[i])

plt.show()
