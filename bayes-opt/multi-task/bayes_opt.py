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


def main():
    ## set up and inits ##
    low_x = 0
    high_x = 10
    num_pts = 1000

    end_sample_count = 30 # this seems like a bad criteria...
    full_x = torch.linspace(low_x, high_x, num_pts)
    n_tasks = 2
    # full_y = gen_correlated_rbfs(full_x, _num_tasks=n_tasks)
    full_y = trash_genner(full_x) # this is just a holdover until I can do something better
    _, y1, _, y2 = data_gen(full_x)
    full_y = torch.stack([y1[0], y2[0]], -1)

    # plt.plot(full_x.numpy(), full_y[:, 0].numpy())
    # plt.plot(full_x.numpy(), full_y[:, 1].numpy())
    # plt.show()
    # get out starting points #
    n_start = 2
    obs_inds = random.sample(range(num_pts), n_start)
    obs_x = full_x[obs_inds]
    obs_y = full_y[obs_inds, :]
    current_max = obs_y[:, 0].max()
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

    # lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)

    ## set up parameter storage ##
    stored_lengths = [None for _ in range(n_tasks)]
    # stored_covar_factor = None
    # stored_var = None
    entered = 0

    while(len(obs_inds) < end_sample_count):
        lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
        model = MultitaskModel(full_x[obs_inds], full_y[obs_inds, :], lh)

        if entered:
        #     # overwrite parameters #
            for tt in range(n_tasks):
                model.covar_module.in_task_covar[tt].log_lengthscale.data[0,0,0] = stored_lengths[tt]
        #     model.covar_module.output_scale_kernel.covar_factor = stored_covar_factor
        #     model.covar_module.output_scale_kernel.var = stored_var
        model.train();
        lh.train();

        ## need to train a little more each time ##
        # Use the adam optimizer
        optimizer = torch.optim.Adam([ {'params': model.parameters()}, ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, model)

        n_iter = 50
        for i in range(n_iter):
            optimizer.zero_grad()
            output = model(full_x[obs_inds])
            loss = -mll(output, full_y[obs_inds, :])
            loss.backward()
            optimizer.step()

        ## store covar parameters ##
        entered = 1
        stored_lengths = [model.covar_module.in_task_covar[tt].log_lengthscale.data[0,0,0]
                            for tt in range(n_tasks)]
        # stored_covar_factor = model.covar_module.output_scale_kernel.covar_factor
        # stored_var = model.covar_module.output_scale_kernel.var

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
                obs_inds.append(int(max_ind))
                found = 1
            else:
                expec_improve[max_ind] = min(expec_improve)
                
        current_max = full_y[obs_inds, 0].max()
        # max, max_ind = torch.max(expec_improve, 0)
        # obs_x = full_x[obs_inds]
        # obs_y = full_y[obs_inds, :]

        ## Plotting To Track Progress ##
        full_col = sns.xkcd_palette(["windows blue"])[0]
        gp_col = sns.xkcd_palette(["amber"])[0]
        if len(obs_inds) % 5 == 0:
            plt.figure()
            plt.plot(full_x.numpy(), full_y[:, 0].numpy(), c=full_col, ls='-')
            plt.plot(full_x[obs_inds].numpy(), full_y[obs_inds, 0].numpy(), c=full_col, marker='.', ls="None")
            plt.plot(full_x[int(max_ind)].numpy(), full_y[int(max_ind), 0].numpy(), marker="*", c='r')
            plt.plot(full_x.numpy(), means.detach().numpy(), ls='-', c=gp_col)
            plt.fill_between(full_x.numpy(), lower[:, 0].detach().numpy(), upper[:, 0].detach().numpy(), alpha=0.5,
                color=gp_col)
            plt.show()

        print("seen ", len(obs_inds), " observations")
        print("observered ", obs_inds)
        print("(", len(obs_inds), " points)")

    return 1

if __name__ == '__main__':
    main()
