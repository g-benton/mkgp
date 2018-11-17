import seaborn as sns
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import sys

sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/data_analysis/compare-kernels/")
from data_gen import data_gen
from indep_rbf import indep_rbf
from mk_tester import mk_tester
from multitask_kernel import multitask


def main():

    test_data = torch.linspace(0, 10, 100)
    test_y1, y1_mean, test_y2, y2_mean = data_gen(test_data)
    stack_y = torch.stack([test_y1, test_y2], -1)[0]

    ## get out mean predictions ##
    mk_mean = mk_tester(test_data, stack_y);
    print("multi-kernel done")
    rbf_mean = indep_rbf(test_data, stack_y);
    rbf_mean.shape
    print("rbf done")
    mt_mean = multitask(test_data, stack_y);
    print("multitask done")

    ## calculate errors ##
    mk_mean1 = mk_mean[:, 0]
    mk_mean2 = mk_mean[:, 1]
    mk_error = (mk_mean1 - test_y1).pow(2).mean()
    mk_error += (mk_mean2 - test_y2).pow(2).mean()

    rbf_mean1 = rbf_mean[:, 0]
    rbf_mean2 = rbf_mean[:, 1]
    rbf_error = (rbf_mean1 - test_y1).pow(2).mean()
    rbf_error += (rbf_mean2 - test_y2).pow(2).mean()

    mt_mean1 = mt_mean[:, 0]
    mt_mean2 = mt_mean[:, 1]
    mt_error = (mt_mean1 - test_y1).pow(2).mean()
    mt_error += (mt_mean2 - test_y2).pow(2).mean()

    # print("MK ERROR: ", mk_error)
    # print("RBF ERROR: ", rbf_error)
    # print("MT ERROR: ", mt_error)

    ## PLOTTING ##
    true_col = sns.xkcd_palette(["windows blue"])[0]
    mod_col = sns.xkcd_palette(["amber"])[0]
    plt.subplot(3, 2, 1)
    plt.plot(test_y1[0].numpy(), marker='o', c=true_col)
    plt.plot(mk_mean1.detach().numpy(), ls='-', c=mod_col)

    plt.subplot(3, 2, 2)
    plt.plot(test_y2[0].numpy(), marker='o', c=true_col)
    plt.plot(mk_mean2.detach().numpy(), ls='-', c=mod_col)

    plt.subplot(3, 2, 3)
    plt.plot(test_y1[0].numpy(), marker='o', c=true_col)
    plt.plot(rbf_mean1.detach().numpy(), ls='-', c=mod_col)

    plt.subplot(3, 2, 4)
    plt.plot(test_y2[0].numpy(), marker='o', c=true_col)
    plt.plot(rbf_mean2.detach().numpy(), ls='-', c=mod_col)

    plt.subplot(3, 2, 5)
    plt.plot(test_y1[0].numpy(), marker='o', c=true_col)
    plt.plot(mt_mean1.detach().numpy(), ls='-', c=mod_col)

    plt.subplot(3, 2, 6)
    plt.plot(test_y2[0].numpy(), marker='o', c=true_col)
    plt.plot(mt_mean2.detach().numpy(), ls='-', c=mod_col)

    plt.show()
    # plt.figure()
    # plt.plot(mk_mean1.numpy(), "k+")
    # plt.plot(y1_mean.detach().numpy(), "b*")
    # plt.show

if __name__ == '__main__':
    main()
