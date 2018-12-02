import math
import random
import matplotlib.pyplot as plt
import gpytorch
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/modular-kernel/")

import mk_kernel
from data_gen_3_tasks import data_gen

def mk_tester(train_x, train_y):

    class MultitaskModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=3
            )
            # self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = mk_kernel.MultitaskRBFKernel(num_tasks=2,log_task_lengthscales=torch.Tensor([math.log(2.5), math.log(0.3)]))
            self.covar_module = mk_kernel.MultiKernel([gpytorch.kernels.RBFKernel(),
                                                       gpytorch.kernels.RBFKernel(),
                                                       gpytorch.kernels.RBFKernel()])

            # self.covar_module = gpytorch.kernels.ScaleKernel(mk_kernel.multi_kernel())
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskModel(train_x, train_y, likelihood)

    model.train();
    likelihood.train();

    optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_iter = 100
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
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
        pred_model = likelihood(model(test_x))
        # mean = predictions.mean

    return pred_model


if __name__ == '__main__':
    num_pts = 100
    test_x = torch.linspace(0,10, num_pts)
    dat1, mean1,  dat2, mean2, dat3, mean3  = data_gen(test_x)
    test_y = torch.stack([dat1, dat2, dat3], -1)[0]
    num_train = 20
    indices = random.sample(range(num_pts), num_train)
    train_x = test_x[indices]
    train_y = test_y[indices, :]
    pred_model = mk_tester(train_x, train_y)

    pred_mean = pred_model.mean

    f, (y1_ax, y2_ax, y3_ax) = plt.subplots(1, 3, figsize=(8, 3))

    y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y1_ax.plot(test_x.numpy(), pred_mean[:, 0].numpy(), 'b')
    # Shade in confidence
    # y1_ax.fill_between(test_data.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    # y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y1_ax.set_title('Observed Values (Likelihood)')

    # Plot training data as black stars
    y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y2_ax.plot(test_x.numpy(), pred_mean[:, 1].numpy(), 'b')
    # Shade in confidence
    # y2_ax.fill_between(test_data.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    # y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y2_ax.set_title('Observed Values (Likelihood)')

    y3_ax.plot(train_x.detach().numpy(), train_y[:, 2].detach().numpy(), 'k*')
    y3_ax.plot(test_x.detach().numpy(), pred_mean[:, 2].detach().numpy(), 'b')
    plt.show()
