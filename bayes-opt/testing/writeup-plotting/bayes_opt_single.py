import seaborn as sns
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import random
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/single-task/")
from data_gen import data_gen
from helper_functions import expected_improvement

def bayes_opt_single(full_x, full_y, obs_inds, end_sample_count=30):
    current_max = full_y[obs_inds].max()
    found_maxes = [current_max]

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    lh = gpytorch.likelihoods.GaussianLikelihood()
    lh.log_noise.data[0,0] = -8
    model = ExactGPModel(full_x[obs_inds], full_y[obs_inds], lh)


    ## standard training stuff ##
    model.train();
    lh.train();

    optimizer = torch.optim.Adam([ {'params': model.covar_module.parameters()}, ], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, model)

    n_iter = 40
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(full_x[obs_inds])
        loss = -mll(output, full_y[obs_inds])
        loss.backward()
        optimizer.step()

    ## do predictions ##
    model.eval();
    lh.eval();

    pred = model(full_x)
    means = pred.mean
    sd = pred.stddev
    lower, upper = pred.confidence_region()

    found = 0
    expec_improve = list(expected_improvement(means, sd, current_max).detach().numpy())
    while not found:
        max_ind = expec_improve.index(max(expec_improve))
        if max_ind not in obs_inds:
            found = 1
        else:
            expec_improve[max_ind] = min(expec_improve)

    current_max = full_y[obs_inds].max()
    found_maxes.append(current_max)

    return pred, max_ind
