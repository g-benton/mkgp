import random
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import gpytorch
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/data_analysis/compare-kernels/holdout-data/")
from indep_rbf import indep_rbf
from mk_tester import mk_tester
from multitask_kernel import multitask

sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/data_analysis/compare-kernels/")
from data_gen import data_gen


def main():
    num_pts = 40
    num_train = 15
    all_x = torch.linspace(0, 50, num_pts)
    num_trial = 100

    all_mk_error1 = [None for _ in range(num_trial)]
    all_mk_error2 = [None for _ in range(num_trial)]
    all_rbf_error1 = [None for _ in range(num_trial)]
    all_rbf_error2 = [None for _ in range(num_trial)]
    all_mt_error1 = [None for _ in range(num_trial)]
    all_mt_error2 = [None for _ in range(num_trial)]

    for trial in range(num_trial):

        y1, y1_mean, y2, y2_mean = data_gen(all_x)
        stack_y = torch.stack([y1, y2], -1)[0]

        ## subset data into training and heldout points ##
        indices = random.sample(range(num_pts), num_train)
        inds = [i for i in sorted(indices)]
        holdout_inds = [i for i in range(num_pts) if i not in inds]

        train_x = all_x[inds]
        train_y = stack_y[inds, :]
        holdout_x = all_x[holdout_inds]
        holdout_y = stack_y[holdout_inds, :]

        ## set the testing points ##
        test_x = all_x
        test_y1 = y1
        test_y2 = y2

        ## get out mean predictions ##
        mk_mean = mk_tester(train_x, train_y, test_x);
        # print("multi-kernel done")
        rbf_mean = indep_rbf(train_x, train_y, test_x);
        # print("rbf done")
        mt_mean = multitask(train_x, train_y, test_x);
        # print("multitask done")

        ## COMPUTE OUTPUTS ##
        mk_mean1 = mk_mean[:, 0]
        mk_mean2 = mk_mean[:, 1]
        rbf_mean1 = rbf_mean[:, 0]
        rbf_mean2 = rbf_mean[:, 1]
        mt_mean1 = mt_mean[:, 0]
        mt_mean2 = mt_mean[:, 1]

        all_mk_error1[trial] = (mk_mean1 - test_y1).pow(2).mean()
        all_mk_error2[trial] = (mk_mean2 - test_y2).pow(2).mean()

        all_rbf_error1[trial] = (rbf_mean1 - test_y1).pow(2).mean()
        all_rbf_error2[trial] = (rbf_mean2 - test_y2).pow(2).mean()

        all_mt_error1[trial] = (mt_mean1 - test_y1).pow(2).mean()
        all_mt_error2[trial] = (mt_mean2 - test_y2).pow(2).mean()

        print("trial ", trial, " done")

    # print("MK ERROR: ", mk_error)
    # print("RBF ERROR: ", rbf_error)
    # print("MT ERROR: ", mt_error)

    # plotting #
    boxplot_list = [np.array(all_mk_error2), np.array(all_rbf_error2), np.array(all_mt_error2)]
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bpl = ax.boxplot(boxplot_list)
    ax.set_xticklabels(["MK Method", "Indep. RBF", "MT Method"])
    ax.set_ylabel("MSE")
    ## just graphical stuff ##
    box_col = sns.xkcd_palette(["windows blue"])[0]
    med_col = sns.xkcd_palette(["amber"])[0]

    for box in bpl["boxes"]:
        box.set(color=box_col, linewidth=2)
    for flier in bpl["fliers"]:
        flier.set(marker='o', c=box_col, alpha=0.5)
    for median in bpl["medians"]:
        median.set(color=med_col, linewidth=1.5)
    for whisker in bpl["whiskers"]:
        whisker.set(color=box_col, linewidth=2)
    for cap in bpl["caps"]:
        cap.set(color=box_col, linewidth=2)
    plt.show()

    print("mk1 mse:", sum(all_mk_error1)/len(all_mk_error1))
    print("mk2 mse:", sum(all_mk_error2)/len(all_mk_error2))
    print("rbf1 mse:", sum(all_rbf_error1)/len(all_rbf_error1))
    print("rbf2 mse:", sum(all_rbf_error2)/len(all_rbf_error2))
    print("mt1 mse:", sum(all_mt_error1)/len(all_mt_error1))
    print("mt2 mse:", sum(all_mt_error2)/len(all_mt_error2))
    ## saving ##
    mse = np.stack([np.concatenate([np.repeat("Multi",2*num_trial),
                          np.repeat("Kron",2*num_trial),
                          np.repeat("Simple",2*num_trial)]),
          np.tile(np.concatenate([np.repeat("1",num_trial),
                                  np.repeat("2",num_trial)]),3),
          np.concatenate([all_mk_error1,
                          all_mk_error2,
                          all_mt_error1,
                          all_mt_error2,
                          all_rbf_error1,
                          all_rbf_error2])],axis=1)

    np.savetxt("all_mse.csv", mse, delimiter=",", fmt="%s")

    # all_mse = np.column_stack((np.array(all_mk_error), np.array(all_rbf_error), np.array(all_mt_error)))
    # np.savetxt("all_mse.csv", all_mse)


if __name__ == '__main__':
    main()

# list = [i for i in range(5)]
# math.mean
# out = np.array(list)
