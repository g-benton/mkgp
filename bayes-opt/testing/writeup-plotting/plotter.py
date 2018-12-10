import math
import torch
import gpytorch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from copy import deepcopy
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/writeup-plotting/")

from data_gen import data_gen
from bayes_opt_multi import bayes_opt_multi
from bayes_opt_single import bayes_opt_single

def main():
    n_pts = 1000
    full_x = torch.linspace(0, 10, n_pts)
    n_start = 4

    _, y1, _, y2 = data_gen(full_x)
    full_y1 = y1[0]
    full_y = torch.stack([y1[0], y2[0]], -1)
    obs_inds = random.sample(range(n_pts), n_start)
    obs_inds2 = deepcopy(obs_inds)

    ## init plot ##
    plt_ind = 1
    cols = sns.color_palette("muted", 4)

    iters_per_plot = 5 ## ONLY CHANGE THIS ##
    n_plots = 3

    for iter_count in range(iters_per_plot * n_plots):
        pred_model, next_pt = bayes_opt_multi(full_x, full_y, obs_inds)
        means = pred_model.mean
        lower, upper = pred_model.confidence_region()

        if iter_count % iters_per_plot == 0:
            # plot #
            plt.subplot(n_plots, 2, plt_ind)
            plt.plot(full_x.numpy(), means[:, 0].detach().numpy(), c=cols[0])
            plt.scatter(full_x[obs_inds].numpy(), full_y[obs_inds, 0].numpy(), c=cols[0], marker='o')
            plt.plot(full_x.numpy(), full_y[:, 0].numpy(), c=cols[0], ls=':')
            plt.plot(full_x.numpy(), full_y[:, 1].numpy(), c=cols[1], ls=":")
            plt.plot(full_x.numpy(), means[:, 1].detach().numpy(), c=cols[1])
            plt.scatter(full_x[obs_inds].numpy(), full_y[obs_inds, 1].numpy(), c=cols[1], marker='o')
            plt.fill_between(full_x.numpy(), lower[:, 0].detach().numpy(), upper[:, 0].detach().numpy(),
                                color=cols[0], alpha=0.2)
            plt.scatter(full_x[next_pt].numpy(), full_y[next_pt, 0].numpy(), c='r', marker="*")
            plt_ind += 1

        obs_inds.append(next_pt)

        pred_model, next_pt = bayes_opt_single(full_x, full_y[:, 0], obs_inds2)
        mean = pred_model.mean
        lower, upper = pred_model.confidence_region()

        if iter_count % iters_per_plot == 0:
            # plot #
            plt.subplot(n_plots, 2, plt_ind)
            plt.plot(full_x.numpy(), mean.detach().numpy(), c=cols[0])
            plt.plot(full_x.numpy(), full_y[:, 0].numpy(), c=cols[0], ls=':')
            plt.scatter(full_x[obs_inds2].numpy(), full_y[obs_inds2, 0].numpy(), c=cols[0], marker='o')
            plt.scatter(full_x[next_pt].numpy(), full_y[next_pt, 0].numpy(), c='r', marker="*")
            plt.fill_between(full_x.numpy(), lower.detach().numpy(), upper.detach().numpy(),
                                color=cols[0], alpha=0.2)
            plt_ind += 1

        obs_inds2.append(next_pt)

    plt.show()
    return 1

if __name__ == '__main__':
    main()
