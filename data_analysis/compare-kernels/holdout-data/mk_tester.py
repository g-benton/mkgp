import math
import gpytorch
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/data_analysis/compare-kernels/holdout_data/")


import mk_kernel
from data_gen import data_gen



def mk_tester(train_x, train_y, test_x):

    class MultitaskModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
            )
            # self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = mk_kernel.MultitaskRBFKernel(num_tasks=2,log_task_lengthscales=torch.Tensor([math.log(2.5), math.log(0.3)]))
            self.covar_module = mk_kernel.MultitaskRBFKernel(num_tasks=2)
            # self.covar_module = gpytorch.kernels.ScaleKernel(mk_kernel.multi_kernel())
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    # test_data = torch.linspace(0, 10, 100)
    # mod1, mod2 = data_gen(test_data, num_samples=1)

    # dat = torch.stack([mod1, mod2,], -1)[0]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskModel(train_x, train_y, likelihood)

    model.train();
    likelihood.train();

    optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_iter = 50
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        # print('Iter %d/%d - Loss: %.3f   log_length1: %.3f log_length2: %.3f log_noise1: %.3f  log_noise2: %.3f' % (
        #     i + 1, n_iter, loss.item(),
        #     model.covar_module.in_task1.lengthscale.item(),
        #     model.covar_module.in_task2.lengthscale.item(),
        #     model.likelihood.log_task_noises.data[0][0],
        #     model.likelihood.log_task_noises.data[0][1]
        # ))

        optimizer.step()

    model.eval();
    likelihood.eval();


    with torch.no_grad(), gpytorch.fast_pred_var():
        # test_x = torch.linspace(0, 1, 51)
        predictions = likelihood(model(test_x))
        mean = predictions.mean

    return likelihood(model(test_x))
    # return mean


if __name__ == '__main__':
    test_data = torch.linspace(0,10, 100)
    dat1, mean1,  dat2, mean2 = data_gen(test_data)
    dat = torch.stack([dat1, dat2], -1)[0]
    mean = mk_tester(test_data, dat)

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
