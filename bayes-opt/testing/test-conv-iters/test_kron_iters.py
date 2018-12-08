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
from bayes_opt_kron import bayes_opt_kron

def main():
    n_pts = 1000
    full_x = torch.linspace(0, 10, n_pts)
    n_start = 2

    n_trials = 50
    kron_iters = np.zeros(n_trials + 1)
    fname = "conv_iter_kron.npz"
    iter = -1
    while iter < n_trials:
        iter += 1
        _, y1, _, y2 = data_gen(full_x)
        full_y1 = y1[0]
        full_y = torch.stack([y1[0], y1[0]], -1)

        obs_inds = random.sample(range(n_pts), n_start)
        obs_inds2 = deepcopy(obs_inds)
        try:
            kron_iters[iter] = len(bayes_opt_kron(full_x, full_y, obs_inds, ei_tol=0.001, max_iters=30))
            print("trial ", iter, " done")
            if iter % 5 == 0:
                np.savez(fname, kron_iters=kron_iters)
        except:
            print("error hit")
            iter -= 1

    np.savez(fname, kron_iters=kron_iters)

    return 1

if __name__ == '__main__':
    main()
