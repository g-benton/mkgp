import seaborn as sns
import matplotlib.pyplot as plt
import torch
import gpytorch
import math
import random
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/multi-task/")
import mk_kernel
from gen_correlated_rbfs import gen_correlated_rbfs
from helper_functions import expected_improvement
from data_gen import data_gen


def bayes_opt_kron(full_x, full_y, obs_inds, ei_tol=0.01, max_iters=25):
    current_max = full_y[:, 0].max()
    found_maxes = [current_max]
    n_tasks = full_y.shape[1]

    ## set up the model ##
    class MultitaskModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=n_tasks
            )
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                    data_covar_module=gpytorch.kernels.RBFKernel(), num_tasks=n_tasks, rank=1
            )
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


    entered = 0
    expec_improve = (1,)
    iter_count = 0
    while(max(expec_improve) > ei_tol and iter_count < max_iters):
        iter_count += 1
        lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
        model = MultitaskModel(full_x[obs_inds], full_y[obs_inds, :], lh)
        model.likelihood.log_noise.data[0,0] = -6

        if entered:
        #     # overwrite parameters #
            model.covar_module.task_covar_module.covar_factor = stored_covar_factor
            model.covar_module.data_covar_module.lengthscale = stored_length
        #     model.covar_module.task_covar_module.var = stored_var

        model.train();
        lh.train();

        ## need to train a little more each time ##
        optimizer = torch.optim.Adam([ {'params': model.covar_module.parameters()}, ], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, model)

        n_iter = 20
        for i in range(n_iter):
            # print("iter hit")
            optimizer.zero_grad()
            output = model(full_x[obs_inds])
            loss = -mll(output, full_y[obs_inds, :])
            loss.backward()
            optimizer.step()

        ## store covar parameters ##
        entered = 1
        stored_length = model.covar_module.data_covar_module.lengthscale
        stored_covar_factor = model.covar_module.task_covar_module.covar_factor

        ## predict full domain ##
        lh.eval();
        model.eval();
        pred = model(full_x)

        means = pred.mean[:, 0]
        sd = pred.stddev[:, 0]
        lower, upper = pred.confidence_region()

        ## observe function at max of expected improvment ##
        found = 0
        expec_improve = list(expected_improvement(means, sd, current_max).detach().numpy())
        while not found:
            max_ind = expec_improve.index(max(expec_improve))
            if max_ind not in obs_inds:
                obs_inds.append(int(max_ind))
                found = 1
            else:
                expec_improve[max_ind] = min(expec_improve)

        current_max = full_y[obs_inds, 0].max()
        found_maxes.append(current_max)

    return found_maxes
