import math
import torch
import gpytorch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from copy import deepcopy
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/test-conv-iters/")

from data_gen import data_gen
from bayes_opt_multi import bayes_opt_multi
from bayes_opt_single import bayes_opt_single
from bayes_opt_kron import bayes_opt_kron


def main():
    n_pts = 1000
    full_x = torch.linspace(0, 10, n_pts)
    n_start = 2

    n_trials = 50
    multi_iters = np.zeros(n_trials + 1)
    kron_iters = np.zeros(n_trials + 1)
    single_iters = np.zeros(n_trials + 1)
    iter = -1
    while iter < n_trials:
        iter += 1
        _, y1, _, y2 = data_gen(full_x)
        full_y1 = y1[0]
        full_y = torch.stack([y1[0], y1[0]], -1)

        obs_inds = random.sample(range(n_pts), n_start)
        obs_inds2 = deepcopy(obs_inds)
        try:
            multi_iters[iter] = len(bayes_opt_multi(full_x, full_y, obs_inds, ei_tol=0.001, max_iters=30))
            kron_iters[iter] = len(bayes_opt_kron(full_x, full_y, obs_inds, ei_tol=0.001, max_iters=30))
            single_iters[iter] = len(bayes_opt_single(full_x, full_y1, obs_inds2, ei_tol=0.001, max_iters=30))
            multi_iters
            single_iters
            print("trial ", iter, " done")
            if iter % 5 == 0:
                np.savez("conv_iter_data.npz", multi_iters=multi_iters, single_iters=single_iters, kron_iters=kron_iters)
        except:
            print("error hit")
            iter -= 1

    np.savez("conv_iter_data.npz", multi_iters=multi_iters, single_iters=single_iters, kron_iters=kron_iters)

    return 1

if __name__ == '__main__':
    main()
