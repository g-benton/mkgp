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

def main():
    ## set ups and inits ##
    low_x = 0
    high_x = 10
    num_pts = 1000

    full_x = torch.linspace(low_x, high_x, num_pts)
    full_y = data_gen(full_x)

    end_sample_count = 30
    n_start = 2

    obs_inds = random.sample(range(num_pts), n_start)
    current_max = full_y[obs_inds].max()

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    entered = 0
    while(len(obs_inds) < end_sample_count):
        lh = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(full_x[obs_inds], full_y[obs_inds], lh)

        if entered:
            model.covar_module.base_kernel.log_lengthscale.data[0,0,0] = stored_length

        ## standard training stuff ##
        model.train();
        lh.train();

        optimizer = torch.optim.Adam([ {'params': model.parameters()}, ], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, model)

        n_iter = 50
        for i in range(n_iter):
            optimizer.zero_grad()
            output = model(full_x[obs_inds])
            loss = -mll(output, full_y[obs_inds])
            loss.backward()
            optimizer.step()

        entered = 1
        stored_length = model.covar_module.base_kernel.log_lengthscale.data[0,0,0]

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
                obs_inds.append(int(max_ind))
                found = 1
            else:
                expec_improve[max_ind] = min(expec_improve)

        current_max = full_y[obs_inds].max()

        full_col = sns.xkcd_palette(["windows blue"])[0]
        gp_col = sns.xkcd_palette(["amber"])[0]

        if len(obs_inds) % 5 == 0:
            plt.figure()
            plt.plot(full_x.numpy(), full_y.numpy(), c=full_col, ls='-')
            plt.plot(full_x[obs_inds].numpy(), full_y[obs_inds].numpy(), c=full_col, marker='.', ls="None")
            plt.plot(full_x[int(max_ind)].numpy(), full_y[int(max_ind)].numpy(), marker="*", c='r')
            plt.plot(full_x.numpy(), means.detach().numpy(), ls='-', c=gp_col)
            plt.fill_between(full_x.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5,
                color=gp_col)
            plt.show()

if __name__ == '__main__':
    main()
