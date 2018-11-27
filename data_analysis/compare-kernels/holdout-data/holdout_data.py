import random
import seaborn as sns
import matplotlib.pyplot as plt
import math
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
    num_pts = 100
    num_train = 25
    all_x = torch.linspace(0, 10, num_pts)
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
    model_out = mk_tester(train_x, train_y, test_x);
    mk_mean = model_out.mean
    lower, upper = model_out.confidence_region()
    # mk_lower, mk_upper = model_out.confidence_region()
    print("multi-kernel done")
    rbf_mean = indep_rbf(train_x, train_y, test_x);
    print("rbf done")
    mt_mean = multitask(train_x, train_y, test_x);
    print("multitask done")

    ## calculate errors ##
    mk_mean1 = mk_mean[:, 0]
    mk_mean2 = mk_mean[:, 1]
    rbf_mean1 = rbf_mean[:, 0]
    rbf_mean2 = rbf_mean[:, 1]
    mt_mean1 = mt_mean[:, 0]
    mt_mean2 = mt_mean[:, 1]

    mk_error = (mk_mean1 - test_y1).pow(2).mean()
    mk_error += (mk_mean2 - test_y2).pow(2).mean()

    rbf_error = (rbf_mean1 - test_y1).pow(2).mean()
    rbf_error += (rbf_mean2 - test_y2).pow(2).mean()

    mt_error = (mt_mean1 - test_y1).pow(2).mean()
    mt_error += (mt_mean2 - test_y2).pow(2).mean()

    print("MK ERROR: ", mk_error)
    print("RBF ERROR: ", rbf_error)
    print("MT ERROR: ", mt_error)

    ## PLOTTING ##
    true_col = sns.xkcd_palette(["windows blue"])[0]
    mod_col = sns.xkcd_palette(["amber"])[0]
    train_y1 = train_y[:, 0]
    train_y2 = train_y[:, 1]
    test_y1 = holdout_y[:, 0]
    test_y2 = holdout_y[:, 1]
    col_titles = ["Task 1", "Task 2"]
    row_titles = ["M.K.", "Indep. RBF", "Kron"]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

    for ind, ax in enumerate(axes[0]):
        ax.set_title(col_titles[ind])
    for ind, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(row_titles[ind])

    train_dat, = axes[0, 0].plot(train_x.numpy(), train_y1.numpy(), marker='*', c=true_col, ls='None')
    # axes[0, 0].plot(holdout_x.numpy(), test_y1.numpy(), marker='o', c=true_col, ls='None')
    pred_mean, = axes[0, 0].plot(test_x.numpy(), mk_mean1.detach().numpy(), ls='-', c=mod_col)
    true_mean, = axes[0, 0].plot(test_x.numpy(), y1_mean[0].numpy(), ls='-', c=true_col)

    axes[0, 1].plot(train_x.numpy(), train_y2.numpy(), marker='*', c=true_col, ls='None')
    # axes[0, 1].plot(holdout_x.numpy(), test_y2.numpy(), marker='o', c=true_col, ls='None')
    axes[0, 1].plot(test_x.numpy(), mk_mean2.detach().numpy(), ls='-', c=mod_col)
    axes[0, 1].plot(test_x.numpy(), y2_mean[0].numpy(), ls='-', c=true_col)

    axes[1, 0].plot(train_x.numpy(), train_y1.numpy(), marker='*', c=true_col, ls='None')
    # axes[1, 0].plot(holdout_x.numpy(), test_y1.numpy(), marker='o', c=true_col, ls='None')
    axes[1, 0].plot(test_x.numpy(), rbf_mean1.detach().numpy(), ls='-', c=mod_col)
    axes[1, 0].plot(test_x.numpy(), y1_mean[0].numpy(), ls='-', c=true_col)

    axes[1, 1].plot(train_x.numpy(), train_y2.numpy(), marker='*', c=true_col, ls='None')
    # axes[1, 1].plot(holdout_x.numpy(), test_y2.numpy(), marker='o', c=true_col, ls='None')
    axes[1, 1].plot(test_x.numpy(), rbf_mean2.detach().numpy(), ls='-', c=mod_col)
    axes[1, 1].plot(test_x.numpy(), y2_mean[0].numpy(), ls='-', c=true_col)

    axes[2, 0].plot(train_x.numpy(), train_y1.numpy(), marker='*', c=true_col, ls='None')
    # axes[2, 0].plot(holdout_x.numpy(), test_y1.numpy(), marker='o', c=true_col, ls='None')
    axes[2, 0].plot(test_x.numpy(), mt_mean1.detach().numpy(), ls='-', c=mod_col)
    axes[2, 0].plot(test_x.numpy(), y1_mean[0].numpy(), ls='-', c=true_col)

    axes[2, 1].plot(train_x.numpy(), train_y2.numpy(), marker='*', c=true_col, ls='None')
    # axes[2, 1].plot(holdout_x.numpy(), test_y2.numpy(), marker='o', c=true_col, ls='None')
    axes[2, 1].plot(test_x.numpy(), mt_mean2.detach().numpy(), ls='-', c=mod_col)
    axes[2, 1].plot(test_x.numpy(), y2_mean[0].numpy(), ls='-', c=true_col)
    plt.legend([train_dat, pred_mean, true_mean], ["Training Points", "Predicted Mean", "Underlying Process"])
    plt.show()
    # print("mean mk method: ", mean)
    # plt.figure()
    # plt.plot(mk_mean1.numpy(), "k+")
    # plt.plot(y1_mean.detach().numpy(), "b*")
    # plt.show

if __name__ == '__main__':
    main()
