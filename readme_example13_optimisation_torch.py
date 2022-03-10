import lab as B
import matplotlib.pyplot as plt
import torch
from wbml.plot import tweak

from stheno.torch import EQ, GP

# Increase regularisation because PyTorch defaults to 32-bit floats.
B.epsilon = 1e-6

# Define points to predict at.
x = torch.linspace(0, 2, 100)
x_obs = torch.linspace(0, 2, 50)

# Sample a true, underlying function and observations with observation noise `0.05`.
f_true = torch.sin(5 * x)
y_obs = torch.sin(5 * x_obs) + 0.05**0.5 * torch.randn(50)


class Model(torch.nn.Module):
    """A GP model with learnable parameters."""

    def __init__(self, init_var=0.3, init_scale=1, init_noise=0.2):
        super().__init__()
        # Ensure that the parameters are positive and make them learnable.
        self.log_var = torch.nn.Parameter(torch.log(torch.tensor(init_var)))
        self.log_scale = torch.nn.Parameter(torch.log(torch.tensor(init_scale)))
        self.log_noise = torch.nn.Parameter(torch.log(torch.tensor(init_noise)))

    def construct(self):
        self.var = torch.exp(self.log_var)
        self.scale = torch.exp(self.log_scale)
        self.noise = torch.exp(self.log_noise)
        kernel = self.var * EQ().stretch(self.scale)
        return GP(kernel), self.noise


model = Model()
f, noise = model.construct()

# Condition on observations and make predictions before optimisation.
f_post = f | (f(x_obs, noise), y_obs)
prior_before = f, noise
pred_before = f_post(x, noise).marginal_credible_bounds()

# Perform optimisation.
opt = torch.optim.Adam(model.parameters(), lr=5e-2)
for _ in range(1000):
    opt.zero_grad()
    f, noise = model.construct()
    loss = -f(x_obs, noise).logpdf(y_obs)
    loss.backward()
    opt.step()

f, noise = model.construct()

# Condition on observations and make predictions after optimisation.
f_post = f | (f(x_obs, noise), y_obs)
prior_after = f, noise
pred_after = f_post(x, noise).marginal_credible_bounds()


def plot_prediction(prior, pred):
    f, noise = prior
    mean, lower, upper = pred
    plt.scatter(x_obs, y_obs, label="Observations", style="train", s=20)
    plt.plot(x, f_true, label="True", style="test")
    plt.plot(x, mean, label="Prediction", style="pred")
    plt.fill_between(x, lower, upper, style="pred")
    plt.ylim(-2, 2)
    plt.text(
        0.02,
        0.02,
        f"var = {f.kernel.factor(0):.2f}, "
        f"scale = {f.kernel.factor(1).stretches[0]:.2f}, "
        f"noise = {noise:.2f}",
        transform=plt.gca().transAxes,
    )
    tweak()


# Plot result.
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Before optimisation")
plot_prediction(prior_before, pred_before)
plt.subplot(1, 2, 2)
plt.title("After optimisation")
plot_prediction(prior_after, pred_after)
plt.savefig("readme_example13_optimisation_torch.png")
plt.show()
