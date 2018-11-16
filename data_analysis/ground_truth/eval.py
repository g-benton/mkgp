import math
import numpy as np
from scipy.io import loadmat
import torch
import gpytorch
from matplotlib import pyplot as plt
import sys

sys.path.append("/Users/davidk/school/mkgp/data_analysis/")
import mk_kernel

class MultitaskModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = mk_kernel.MultitaskRBFKernel(num_tasks=2)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

## EXAMPLE 1
multi_mse_task1 = multi_mse_task2 = kron_mse_task1 = kron_mse_task2 = np.array(())
n = 50
ex1 = np.genfromtxt("example1.csv",delimiter=",")
train_indices = np.sort(np.random.randint(0,1000,n))
test_indices = [i for i in range(1000) if i not in train_indices]
train_x = torch.tensor(ex1[train_indices,0]).type(torch.FloatTensor)
train_y = torch.tensor(ex1[train_indices,1:3]).type(torch.FloatTensor)
test_x = ex1[test_indices,0]
test_y = ex1[test_indices,1:3]
train_y[:,0] = train_y[:,0]+torch.randn((n))
train_y[:,1] = train_y[:,1]+torch.randn((n)).mul(0.5)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskModel(train_x,train_y,likelihood)
model.train()
likelihood.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
n_iter = 50
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d MLL: %.3f length1: %.3f length2: %.3f output1: %.3f output2: %.3f cross: %.3f noise1: %.3f noise2: %.3f' % (
        i + 1,
        n_iter,
        loss.item(),
        model.covar_module.in_task1.lengthscale,
        model.covar_module.in_task2.lengthscale,
        model.covar_module.output_scale_kernel.covar_matrix[0,0,0],
        model.covar_module.output_scale_kernel.covar_matrix[0,1,1],
        model.covar_module.output_scale_kernel.covar_matrix[0,1,0],
        likelihood.log_task_noises[0,0].exp(),
        likelihood.log_task_noises[0,1].exp()
    )
    )
    optimizer.step()

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.fast_pred_var():
    eval_x = torch.linspace(0,10,1000)
    predictions = likelihood(model(eval_x))
    mean = predictions.mean
    lower,upper = predictions.confidence_region()

multi_mse_task1 = np.append(multi_mse_task1, np.sum(np.power(mean[test_indices,0].numpy() - test_y[:,0],2))/np.shape(test_y)[0])
multi_mse_task2 = np.append(multi_mse_task2, np.sum(np.power(mean[test_indices,1].numpy() - test_y[:,1],2))/np.shape(test_y)[0])

class KronMultitaskModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(KronMultitaskModel, self).__init__(train_x, train_y, likelihood)
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


kronlikelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
kronmodel = KronMultitaskModel(train_x,train_y,kronlikelihood)
kronmodel.train()
kronlikelihood.train()
optimizer = torch.optim.Adam([
    {'params': kronmodel.parameters()},
], lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(kronlikelihood, kronmodel)
n_iter = 50
for i in range(n_iter):
    optimizer.zero_grad()
    output = kronmodel(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d MLL: %.3f length: %.3f output1: %.3f output2: %.3f cross: %.3f noise1: %.3f noise2: %.3f' % (
        i + 1,
        n_iter,
        loss.item(),
        kronmodel.covar_module.data_covar_module.lengthscale,
        kronmodel.covar_module.task_covar_module.covar_matrix[0,0,0],
        kronmodel.covar_module.task_covar_module.covar_matrix[0,1,1],
        kronmodel.covar_module.task_covar_module.covar_matrix[0,1,0],
        kronlikelihood.log_task_noises[0,0].exp(),
        kronlikelihood.log_task_noises[0,1].exp()
    )
    )
    optimizer.step()

kronmodel.eval()
kronlikelihood.eval()
with torch.no_grad(), gpytorch.fast_pred_var():
    eval_x = torch.linspace(0,10,1000)
    kronpredictions = kronlikelihood(kronmodel(eval_x))
    kronmean = kronpredictions.mean
    kronlower,kronupper = kronpredictions.confidence_region()

#plt.subplot(2,1,1)
#plt.fill_between(eval_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.2,color="C0")
#plt.fill_between(eval_x.numpy(), kronlower[:, 0].numpy(), kronupper[:, 0].numpy(), alpha=0.2,color="C3")
#plt.scatter(train_x.detach().numpy(),train_y[:,0].detach().numpy(),color="C1")
#plt.plot(test_x,test_y[:,0],color="C2")
#plt.plot(eval_x.numpy(), kronmean[:, 0].numpy(), "C3")
#plt.plot(eval_x.numpy(), mean[:, 0].numpy(), "C0")
#plt.subplot(2,1,2)
#plt.fill_between(eval_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.2,color="C0")
#plt.fill_between(eval_x.numpy(), kronlower[:, 1].numpy(), kronupper[:, 1].numpy(), alpha=0.2,color="C3")
#plt.scatter(train_x.detach().numpy(),train_y[:,1].detach().numpy(),color="C1")
#plt.plot(test_x,test_y[:,1],color="C2")
#plt.plot(eval_x.numpy(), kronmean[:, 1].numpy(), "C3")
#plt.plot(eval_x.numpy(), mean[:, 1].numpy(), "C0")
#plt.show()

kron_mse_task1 = np.append(kron_mse_task1, np.sum(np.power(kronmean[test_indices,0].numpy() - test_y[:,0],2))/np.shape(test_y)[0])
kron_mse_task2 = np.append(kron_mse_task2, np.sum(np.power(kronmean[test_indices,1].numpy() - test_y[:,1],2))/np.shape(test_y)[0])
