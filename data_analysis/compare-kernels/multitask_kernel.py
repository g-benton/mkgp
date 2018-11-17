import math
import torch
import gpytorch
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/data_analysis/compare-kernels/")

import mk_kernel
from data_gen import data_gen



def multitask(test_data, test_y):
    class MultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
            )
            self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    # test_data = torch.linspace(0, 10, 100)
    # mod1, mod2 = data_gen(test_data, num_samples=1)
    # dat = torch.stack([mod1, mod2,], -1)[0]
    dat = test_y
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPModel(test_data, dat, likelihood)

    model.train();
    likelihood.train();

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_iter = 50
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(test_data)
        loss = -mll(output, dat)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   logscale: %.3f  log_noise: %.3f' % (
        #     i + 1, n_iter, loss.item(),
        #     model.covar_module.data_covar_module.log_lengthscale.data.item(),
        #     model.likelihood.log_noise.item()
        # ))
        # print(model.covar_module.task_covar_module.covar_matrix.evaluate())
        optimizer.step()

    model.eval();
    likelihood.eval();

    # f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

    # # Make predictions
    with torch.no_grad(), gpytorch.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        predictions = likelihood(model(test_data))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # y1_ax.plot(test_data.detach().numpy(), dat[:, 0].detach().numpy(), 'k*')
    # # Predictive mean as blue line
    # y1_ax.plot(test_data.numpy(), mean[:, 0].numpy(), 'b')
    # # Shade in confidence
    # y1_ax.fill_between(test_data.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    # # y1_ax.set_ylim([-3, 3])
    # y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # y1_ax.set_title('Observed Values (Likelihood)')
    #
    # # Plot training data as black stars
    # y2_ax.plot(test_data.detach().numpy(), dat[:, 1].detach().numpy(), 'k*')
    # # Predictive mean as blue line
    # y2_ax.plot(test_data.numpy(), mean[:, 1].numpy(), 'b')
    # # Shade in confidence
    # y2_ax.fill_between(test_data.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    # # y2_ax.set_ylim([-3, 3])
    # y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # y2_ax.set_title('Observed Values (Likelihood)')
    # plt.show()
    return mean

if __name__ == '__main__':
    test_data = torch.linspace(0,10, 100)
    dat1, mean1,  dat2, mean2 = data_gen(test_data)
    dat = torch.stack([dat1, dat2], -1)[0]
    mean = multitask(test_data, dat)

    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

    y1_ax.plot(test_data.detach().numpy(), dat[:, 0].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y1_ax.plot(test_data.numpy(), mean[:, 0].numpy(), 'b')
    # Shade in confidence
    # y1_ax.fill_between(test_data.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    # y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y1_ax.set_title('Observed Values (Likelihood)')

    # Plot training data as black stars
    y2_ax.plot(test_data.detach().numpy(), dat[:, 1].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y2_ax.plot(test_data.numpy(), mean[:, 1].numpy(), 'b')
    # Shade in confidence
    # y2_ax.fill_between(test_data.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    # y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y2_ax.set_title('Observed Values (Likelihood)')
    plt.show()
