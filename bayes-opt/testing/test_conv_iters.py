import math
import torch
import gpytorch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from copy import deepcopy
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/")

from data_gen import data_gen
from bayes_opt_multi import bayes_opt_multi
from bayes_opt_single import bayes_opt_single


def main():
    n_pts = 1000
    full_x = torch.linspace(0, 10, n_pts)
    n_start = 2

    n_trials = 3
    for iter in range(n_trials):

        _, y1, _, y2 = data_gen(full_x)
        full_y1 = y1[0]
        full_y = torch.stack([y1[0], y1[0]], -1)

        obs_inds = random.sample(range(n_pts), n_start)
        obs_inds2 = deepcopy(obs_inds)
        multi_maxes = bayes_opt_multi(full_x, full_y, obs_inds, end_sample_count=15)
        single_maxes = bayes_opt_single(full_x, full_y1, obs_inds2, end_sample_count=15)


    # plt.plot(single_maxes[n_start:], marker='*')
    # plt.plot(multi_maxes[n_start:], marker='o')
    # plt.show()

    return 1

if __name__ == '__main__':
    main()
