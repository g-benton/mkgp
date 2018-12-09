import numpy as np


if __name__ == '__main__':
    kron_dat = np.load("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/test-conv-iters/conv_iter_kron.npz")
    kron_dat = kron_dat["kron_iters"]
    other_dat = np.load("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/test-conv-iters/conv_iter_data.npz")
    single_dat = other_dat["single_iters"]
    multi_dat = other_dat["multi_iters"]

    kron_dat[kron_dat == 31] = None
    single_dat[single_dat == 31] = None
    multi_dat[multi_dat == 31] = None

    np.nanmean(kron_dat)
    np.nanmean(single_dat)
    np.nanmean(multi_dat)
