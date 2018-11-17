import math
import gpytorch
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/data_analysis/compare-kernels/")

import mk_kernel
from data_gen import data_gen


def indep_rbf(test_data, test_y):

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    # TODO: edit this so it reads in stored data (necessary for good comparison)

    # test_data = torch.linspace(0, 10, 100)
    # mod1, mod2 = data_gen(test_data, num_samples=1)
    # mod1_mean = mod1.mean()
    # mod2_mean = mod2.mean()
    # mod1 = mod1[0] - mod1_mean
    # mod2 = mod2[0] - mod2_mean
    mod1 = test_y[:, 0]
    mod2 = test_y[:, 1]
    l1 = gpytorch.likelihoods.GaussianLikelihood()
    model1 =  ExactGPModel(test_data, mod1, l1)

    l2 = gpytorch.likelihoods.GaussianLikelihood()
    model2 = ExactGPModel(test_data, mod2, l2)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(l1, model1)

    model1.train();
    l1.train();
    optimizer = torch.optim.Adam([{'params': model1.parameters()},], lr=0.1)

    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model1(test_data)

        # Calc loss and backprop gradients
        loss = -mll(output, mod1)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model1.covar_module.base_kernel.log_lengthscale.item(),
        #     model1.likelihood.log_noise.item()
        # ))
        optimizer.step();

    mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(l2, model2)
    model2.train();
    l2.train();
    optimizer = torch.optim.Adam([{'params': model2.parameters()},], lr=0.1)

    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model2(test_data)

        # Calc loss and backprop gradients
        loss = -mll2(output, mod2)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model2.covar_module.base_kernel.log_lengthscale.item(),
        #     model2.likelihood.log_noise.item()
        # ))
        optimizer.step();


    model1.eval();
    model2.eval();
    l1.eval();
    l2.eval();
    with torch.no_grad(), gpytorch.fast_pred_var():
        # test_x = torch.linspace(0, 1, 51)
        predictions = l1(model1(test_data))
        predictions = model1(test_data)
        mean1 = predictions.mean

        predictions = l2(model2(test_data))
        predictions = model2(test_data)
        mean2 = predictions.mean

    # f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
    # y1_ax.plot(test_data.detach().numpy(), mod1.numpy(), 'k*')
    # # Predictive mean as blue line
    # y1_ax.plot(test_data.numpy(), mean1.detach().numpy(), 'b')
    # # Shade in confidence
    # # y1_ax.fill_between(test_data.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    # # y1_ax.set_ylim([-3, 3])
    # y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # y1_ax.set_title('Observed Values (Likelihood)')
    #
    # # Plot training data as black stars
    # y2_ax.plot(test_data.detach().numpy(), mod2.detach().numpy(), 'k*')
    # # Predictive mean adetach().s blue line
    # y2_ax.plot(test_data.numpy(), mean2.detach().numpy(), 'b')
    # # Shade in confidence
    # # y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    # # y2_ax.set_ylim([-3, 3])
    # y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # y2_ax.set_title('Observed Values (Likelihood)')
    # plt.show()

    return torch.stack([mean1, mean2], -1)
    
if __name__ == '__main__':
    test_data = torch.linspace(0,10, 100)
    dat1, mean1,  dat2, mean2 = data_gen(test_data)
    dat = torch.stack([dat1, dat2], -1)[0]
    mean = indep_rbf(test_data, dat)

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
