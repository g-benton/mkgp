import torch
import gpytorch
import math
import matplotlib.pyplot as plt


def data_gen(full_x):
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    lh = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(None, None, lh)

    model.eval();
    lh.eval();

    full_y = model(full_x).sample(sample_shape=torch.Size([1]))

    return full_y[0]

if __name__ == '__main__':
    full_x = torch.linspace(0, 10, 1000)
    full_y = data_gen(full_x)

    plt.plot(full_x.numpy(), full_y.detach().numpy())
    plt.show()
