import timeit
import math
import random
import torch
import gpytorch
import numpy as np
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/runtime-testing/")

import mk_kernel

def main():
    ## set up and inits ##
    n_pt_sizes = 8
    n_task_sizes = 20
    n_pts_list = [10**ii for ii in range(1, n_pt_sizes)]
    n_tasks_list = [ii for ii in range(2, n_task_sizes)]
    n_pts_list
    runtimes = np.zeros((len(n_pts_list), len(n_tasks_list)))
    ## big ol' loops ##
    for pt_ind, n_pts in enumerate(n_pts_list):
        pred_pts = torch.linspace(0, 100, n_pts)

        for task_ind, n_tasks in enumerate(n_tasks_list):
            # print("hit")
            kern_list = [gpytorch.kernels.RBFKernel() for _ in range(n_tasks)]

            # make model class #
            class MKModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super(MKModel, self).__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.MultitaskMean(
                        gpytorch.means.ConstantMean(), num_tasks=n_tasks
                    )
                    self.covar_module = mk_kernel.MultiKernel(kern_list)
                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

            like = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
            mk_model = MKModel(torch.tensor(0), torch.tensor(0), like)

            # mk_model.eval();
            # like.eval();
            #
            start_time = timeit.default_timer()
            prior_pred = mk_model.forward(pred_pts)
            stop_time = timeit.default_timer()
            runtimes[pt_ind, task_ind] = stop_time - start_time

            print("done with ", n_pts, "points and ", n_tasks, " tasks")
    ## save runtime matrix ##
    np.savez("runtimes.npz", n_pts=np.array(n_pts_list), n_tasks=np.array(n_tasks_list),
             runtimes=runtimes)
    print("saved")
    # exit #
    return 1


if __name__ == '__main__':
    main()
