import math
import numpy as np
from scipy.io import loadmat
import torch
import gpytorch
from matplotlib import pyplot as plt
import sys

# import local_gpt as gpytorch
import mk_kernel

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

## set up data ##
dat = np.genfromtxt("data/PHKHistoricalPricing.csv",delimiter=",")
nav = dat[0:100,1]
price = dat[0:100,4]
train_x = torch.tensor(range(len(nav))).float()
train_y = torch.stack([torch.tensor(nav),torch.tensor(price)],-1).float()
train_y[:,0] = train_y[:,0] - torch.mean(train_y[:,0])
train_y[:,1] = train_y[:,1] - torch.mean(train_y[:,1])
print(train_y.shape)
train_xold = torch.linspace(0, 1, 100)
test_xold = torch.linspace(0.1, 1.1, 52)
train_yold = torch.stack([torch.sin(train_xold * (4 * math.pi)) + torch.randn(train_xold.size()) * 0.2 + 1,torch.cos(train_xold * (2 * math.pi)) + torch.randn(train_xold.size()) * 0.2,], -1)
# train_y1 = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
# train_y2 = torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
# train_y = torch.cat((train_y1, train_y2), 0)
plt.scatter(train_x,train_y[:,0])
plt.scatter(train_x,train_y[:,1])
plt.show()

## set up model ##
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskModel(train_x, train_y, likelihood)
model.covar_module.in_task1.log_lengthscale.data[0][0][0] = -5
# model.covar_module.log_task_lengthscales = torch.Tensor([math.log(2.5), math.log(0.3)])

model.train();
likelihood.train();


for i in model.named_parameters():
    print(i)
optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iter = 50
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    # print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    print('Iter %d/%d - Loss: %.3f   log_length1: %.3f log_length2: %.3f log_noise1: %.3f  log_noise2: %.3f' % (
        i + 1, n_iter, loss.item(),
        model.covar_module.in_task1.lengthscale.item(),
        model.covar_module.in_task2.lengthscale.item(),
        model.likelihood.log_task_noises.data[0][0],
        model.likelihood.log_task_noises.data[0][1]
    ))

    # for ind, ii in enumerate(model.named_parameters()):
    #     print(ii[1].grad)
    optimizer.step()




    model.eval();
    likelihood.eval();

    # print(model.covar_module.log_task_lengthscales)

    with torch.no_grad(), gpytorch.fast_pred_var():
        test_x = train_x
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()


    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
    y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
    # Shade in confidence
    y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y1_ax.set_title('NAV')

    # Plot training data as black stars
    y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
    # Shade in confidence
    y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y2_ax.set_title('Price')
    plt.show()

    #
    # plt.figure()
    # plt.plot(mean[:, 0].numpy(), 'b')
    # plt.plot(mean[:, 1].numpy(), 'k')
    # plt.show()
if __name__ == '__main__':
    main()
