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

class SimpleModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SimpleModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

nsim = 100
## EXAMPLE 3
task1_mse = task2_mse = multi_mse_task1 = multi_mse_task2 = kron_mse_task1 = kron_mse_task2 = np.array(())
n = 50
ex3 = np.genfromtxt("example3.csv",delimiter=",")
for rep in range(nsim):
    print(rep)
    train_indices = np.sort(np.random.randint(0,1000,n))
    test_indices = [i for i in range(1000) if i not in train_indices]
    train_x = torch.tensor(ex3[train_indices,0]).type(torch.FloatTensor)
    train_y = torch.tensor(ex3[train_indices,1:3]).type(torch.FloatTensor)
    test_x = ex3[test_indices,0]
    test_y = ex3[test_indices,1:3]
    train_y[:,0] = train_y[:,0]+torch.randn((n)).mul(0.5)
    train_y[:,1] = train_y[:,1]+torch.randn((n)).mul(0.5)

    # FIT MULTITASK/MULTIKERNEL METHOD
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
        #print('Iter %d/%d MLL: %.3f length1: %.3f length2: %.3f output1: %.3f output2: %.3f cross: %.3f noise1: %.3f noise2: %.3f' % (
        #    i + 1,
        #    n_iter,
        #    loss.item(),
        #    model.covar_module.in_task1.lengthscale,
        #    model.covar_module.in_task2.lengthscale,
        #    model.covar_module.output_scale_kernel.covar_matrix[0,0,0],
        #    model.covar_module.output_scale_kernel.covar_matrix[0,1,1],
        #    model.covar_module.output_scale_kernel.covar_matrix[0,1,0],
        #    likelihood.log_task_noises[0,0].exp(),
        #    likelihood.log_task_noises[0,1].exp()
        #))
        optimizer.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.fast_pred_var():
        eval_x = torch.linspace(0,10,1000)
        predictions = likelihood(model(eval_x))
        mean = predictions.mean
        #lower,upper = predictions.confidence_region()

    multi_mse_task1 = np.append(multi_mse_task1, np.sum(np.power(mean[test_indices,0].numpy() - test_y[:,0],2))/np.shape(test_y)[0])
    multi_mse_task2 = np.append(multi_mse_task2, np.sum(np.power(mean[test_indices,1].numpy() - test_y[:,1],2))/np.shape(test_y)[0])

    ## FIT KRONECKER MULTITASK METHOD
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
        #print('Iter %d/%d MLL: %.3f length: %.3f output1: %.3f output2: %.3f cross: %.3f noise1: %.3f noise2: %.3f' % (
        #    i + 1,
        #    n_iter,
        #    loss.item(),
        #    kronmodel.covar_module.data_covar_module.lengthscale,
        #    kronmodel.covar_module.task_covar_module.covar_matrix[0,0,0],
        #    kronmodel.covar_module.task_covar_module.covar_matrix[0,1,1],
        #    kronmodel.covar_module.task_covar_module.covar_matrix[0,1,0],
        #    kronlikelihood.log_task_noises[0,0].exp(),
        #    kronlikelihood.log_task_noises[0,1].exp()
        #))
        optimizer.step()

    kronmodel.eval()
    kronlikelihood.eval()
    with torch.no_grad(), gpytorch.fast_pred_var():
        eval_x = torch.linspace(0,10,1000)
        kronpredictions = kronlikelihood(kronmodel(eval_x))
        kronmean = kronpredictions.mean
        #kronlower,kronupper = kronpredictions.confidence_region()

    kron_mse_task1 = np.append(kron_mse_task1, np.sum(np.power(kronmean[test_indices,0].numpy() - test_y[:,0],2))/np.shape(test_y)[0])
    kron_mse_task2 = np.append(kron_mse_task2, np.sum(np.power(kronmean[test_indices,1].numpy() - test_y[:,1],2))/np.shape(test_y)[0])

    ## FIT EACH TASK SEPARATELY
    task1likelihood = gpytorch.likelihoods.GaussianLikelihood()
    task1model = SimpleModel(train_x, train_y[:,0], task1likelihood)
    task1model.train()
    task1likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': task1model.parameters()},  # Includes GaussianTask1likelihood parameters
       ], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(task1likelihood, task1model)
    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from task1model
        output = task1model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y[:,0])
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f   length: %.3f   log_noise: %.3f' % (
        #    i + 1, training_iter, loss.item(),
        #    task1model.covar_module.base_kernel.lengthscale,
        #    task1model.likelihood.log_noise.item()
        #))
        optimizer.step()
    task2likelihood = gpytorch.likelihoods.GaussianLikelihood()
    task2model = SimpleModel(train_x, train_y[:,1], task2likelihood)
    task2model.train()
    task2likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': task2model.parameters()},  # Includes GaussianTask2likelihood parameters
       ], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(task2likelihood, task2model)
    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from task2model
        output = task2model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y[:,1])
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f   length: %.3f   log_noise: %.3f' % (
        #    i + 1, training_iter, loss.item(),
        #    task2model.covar_module.base_kernel.lengthscale,
        #    task2model.likelihood.log_noise.item()
        #))
        optimizer.step()
    task1model.eval()
    task1likelihood.eval()
    with torch.no_grad(), gpytorch.fast_pred_var():
        eval_x = torch.linspace(0,10,1000)
        task1pred = task1likelihood(task1model(eval_x))
        task1mean = task1pred.mean
        #task1lower,task1upper = task1pred.confidence_region()
    task2model.eval()
    task2likelihood.eval()
    with torch.no_grad(), gpytorch.fast_pred_var():
        eval_x = torch.linspace(0,10,1000)
        task2pred = task2likelihood(task2model(eval_x))
        task2mean = task2pred.mean
        #task2lower,task2upper = task2pred.confidence_region()
    task1_mse = np.append(task1_mse, np.sum(np.power(task1mean[test_indices].numpy() - test_y[:,0],2))/np.shape(test_y)[0])
    task2_mse = np.append(task2_mse, np.sum(np.power(task2mean[test_indices].numpy() - test_y[:,1],2))/np.shape(test_y)[0])

mse = np.stack([np.concatenate([np.repeat("Multi",2*nsim),
                          np.repeat("Kron",2*nsim),
                          np.repeat("Simple",2*nsim)]),
          np.tile(np.concatenate([np.repeat("1",nsim),
                                  np.repeat("2",nsim)]),3),
          np.concatenate([multi_mse_task1,
                          multi_mse_task2,
                          kron_mse_task1,
                          kron_mse_task2,
                          task1_mse,
                          task2_mse])],axis=1)

np.savetxt("ex3_mse.csv",mse,delimiter=",",fmt="%s")
