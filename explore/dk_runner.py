import math
import numpy as np
from scipy.io import loadmat
import torch
import gpytorch
from matplotlib import pyplot as plt
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/explore/")

# import local_gpt as gpytorch
import dk_kernel

class MultitaskModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        # self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = dk_kernel.MultitaskRBFKernel(data_covar_module=gpytorch.kernels.RBFKernel(),
                                                         num_tasks=2, log_task_lengthscales=torch.Tensor([1, 0]))
        # self.covar_module = gpytorch.kernels.ScaleKernel(mk_kernel.multi_kernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def main():
    train_x = torch.linspace(0, 1, 100)

    train_y = torch.stack([torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,], -1)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskModel(train_x, train_y, likelihood)

    # Set into eval mode
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimize

    optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_iter = 50
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        optimizer.step()

    model.eval();
    likelihood.eval();

    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

    # # Make predictions
    with torch.no_grad(), gpytorch.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        # lower, upper = predictions.confidence_region()

    # This contains predictions for both tasks, flattened out
    # The first half of the predictions is for the first task
    # The second half is for the second task

    # Plot training data as black stars
    y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
    # Shade in confidence
    # y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y1_ax.set_title('Observed Values (Likelihood)')

    # Plot training data as black stars
    y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
    # Shade in confidence
    # y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y2_ax.set_title('Observed Values (Likelihood)')
    plt.show()

    plt.figure()
    plt.plot(mean[:, 0].numpy(), 'b')
    plt.plot(mean[:, 1].numpy(), 'k')
    plt.show()

if __name__ == '__main__':
    main()
