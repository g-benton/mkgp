import torch
import math
import random
import gpytorch
import slus

def data_gen(domain, num_tasks):
    kern_list = [gpytorch.kernels.RBFKernel for _ in range(num_tasks)]
    class MKModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MKModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_tasks
            )
            self.covar_module = mk_kernel.MultiKernel(kern_list)
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


    like = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MKModel()

    return train_x, train_y
