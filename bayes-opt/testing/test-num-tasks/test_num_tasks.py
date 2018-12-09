import math
import torch
import gpytorch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from copy import deepcopy
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/test-num-tasks/")

from data_gen import data_gen
from bayes_opt_multi import bayes_opt_multi
from bayes_opt_single import bayes_opt_single

def main():
    n_pts = 1000
    full_x = torch.linspace(0, 10, n_pts)
    n_start = 2

    n_trials = 4
    n_samples = 5
    one_maxes = np.zeros((n_trials+1, n_samples - n_start - 1))
    two_maxes = np.zeros((n_trials+1, n_samples - n_start - 1))
    three_maxes = np.zeros((n_trials+1, n_samples - n_start - 1))
    four_maxes = np.zeros((n_trials+1, n_samples - n_start - 1))
    five_maxes = np.zeros((n_trials+1, n_samples - n_start - 1))

    iter = -1
    while(iter < n_trials):
        iter += 1

        full_y = data_gen(full_x)
        obs_inds = random.sample(range(n_pts), n_start)
        obs_inds2 = deepcopy(obs_inds)
        obs_inds3 = deepcopy(obs_inds)
        obs_inds4 = deepcopy(obs_inds)
        obs_inds5 = deepcopy(obs_inds)
        y1 = full_y[:, 0]
        one_maxes[iter, :] = bayes_opt_single(full_x, y1, obs_inds, end_sample_count=n_samples)
        two_maxes[iter, :] = bayes_opt_multi(full_x, full_y[:, 0:1], obs_inds2, end_sample_count=n_samples)
        three_maxes[iter, :] = bayes_opt_multi(full_x, full_y[:, 0:2], obs_inds3, end_sample_count=n_samples)
        four_maxes[iter, :] = bayes_opt_multi(full_x, full_y[:, 0:3], obs_inds4, end_sample_count=n_samples)
        five_maxes[iter, :] = bayes_opt_multi(full_x, full_y, obs_inds5, end_sample_count=n_samples)


    return 1



if __name__ == '__main__':
    main()
