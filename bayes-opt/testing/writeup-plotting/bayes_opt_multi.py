import seaborn as sns
import matplotlib.pyplot as plt
import torch
import gpytorch
import math
import random
import sys
sys.path.append("/Users/davidk/school/mkgp_pure/bayes-opt/multi-task/")
import mk_kernel
from gen_correlated_rbfs import gen_correlated_rbfs
from helper_functions import expected_improvement
from data_gen import data_gen

def plot_truth(y):
    plt.plot(y[:, 0].numpy())
    plt.plot(y[:, 1].numpy())
    plt.show()
    pass

def trash_genner(full_x):
    cntr = full_x.mean()
    ## change this to be a random draw from a GP? ##
    y1 = (full_x - cntr).pow(2).mul(-1) + 10
    y2 = y1 + torch.sin(full_x * 2 * math.pi)

    return torch.stack([y2, y1], -1)


def bayes_opt_multi(full_x, full_y, obs_inds):
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
            self.covar_module = mk_kernel.MultiKernel(
                [gpytorch.kernels.RBFKernel() for _ in range(n_tasks)]
            )
            # self.covar_module = mk_kernel.MultitaskRBFKernel(num_tasks=2, rank=2)
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


    ## set up parameter storage ##
    stored_lengths = [None for _ in range(n_tasks)]
    # stored_covar_factor = None
    # stored_var = None
    lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    model = MultitaskModel(full_x[obs_inds], full_y[obs_inds, :], lh)
    model.likelihood.log_noise.data[0,0] = -6
    model.likelihood.log_task_noises.data[0,0] = -6
    model.likelihood.log_task_noises.data[0,1] = -6

    model.train();
    lh.train();

    ## need to train a little more each time ##
    # Use the adam optimizer
    optimizer = torch.optim.Adam([ {'params': model.covar_module.parameters()}, ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, model)

    n_iter = 40
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(full_x[obs_inds])
        loss = -mll(output, full_y[obs_inds, :])
        loss.backward()
        optimizer.step()


    ## predict full domain ##
    lh.eval();
    model.eval();
    pred = model(full_x)
    dump = pred.covariance_matrix ## just to build the cache

    means = pred.mean[:, 0]
    sd = pred.stddev[:, 0]
    lower, upper = pred.confidence_region()

    ## observe function at max of expected improvment ##
    found = 0
    expec_improve = list(expected_improvement(means, sd, current_max).detach().numpy())
    while not found:
        max_ind = expec_improve.index(max(expec_improve))
        if max_ind not in obs_inds:
            found = 1
        else:
            expec_improve[max_ind] = min(expec_improve)

    current_max = full_y[obs_inds, 0].max()
    found_maxes.append(current_max)

    return pred, max_ind
