import math
import random
import torch
import gpytorch
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/modular-kernel/")

from data_gen import data_gen


def main():
    num_pts = 100
    test_x = torch.linspace(0, 10, num_pts)
    dat1, mean1, dat2, mean2 = data_gen(test_x)
    test_y = torch.stack([dat1, dat2], -1)[0]
    num_train = 20
    indices = random.sample(range(num_pts), num_train)
    train_x = test_x[indices]
    train_y = test_y[indices, :]

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
    optimizer = torch.optim.Adam([ {'params': kronmodel.parameters()}, ], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(kronlikelihood, kronmodel)
    n_iter = 50
    for i in range(n_iter):
        optimizer.zero_grad()
        output = kronmodel(train_x)
        loss = -mll(output, train_y)
        loss.backward()
    kronmodel.eval()
    kronlikelihood.eval()
    with torch.no_grad(), gpytorch.fast_pred_var():
        eval_x = torch.linspace(0,10,1000)
        kronpredictions = kronlikelihood(kronmodel(eval_x))
        kronmean = kronpredictions.mean
        kronlower,kronupper = kronpredictions.confidence_region()

    print("lower = ", kronlower)
    return 1

if __name__ == '__main__':
    main()
