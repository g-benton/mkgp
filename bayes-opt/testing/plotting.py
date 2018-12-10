import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/")

def main():
    # kron_dat = np.load("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/kron_conv_rates.npz")

    other_dat = np.load("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/conv_rates_data.npz")
    multi_maxes = other_dat["multi_maxes"]
    single_maxes = other_dat["single_maxes"]
    kron_maxes = other_dat["kron_maxes"]

    kron_avg = np.mean(kron_maxes, axis=0)
    multi_avg = np.mean(multi_maxes, axis=0)
    single_avg = np.mean(single_maxes, axis=0)

    ## more setup ##
    ticks = [ii+1 for ii in range(len(kron_avg))]
    plot_cols = sns.color_palette("muted", 3)
    ## plotting ##
    plt.plot(ticks, multi_avg, c=plot_cols[0], ls="-", linewidth=2)
    plt.plot(ticks, kron_avg, c=plot_cols[1], ls="-", linewidth=2)
    plt.plot(ticks, single_avg, c=plot_cols[2], ls="-", linewidth=2)
    plt.axhline(y=10, ls="-.", c="k")
    plt.legend(["Multi-Kernel", "Kronecker", "Single RBF", "True Max"])
    plt.title("Averaged Maximum Found per Iteration")
    plt.ylabel("Maximum Found")
    plt.xlabel("Iteration")
    plt.xticks(ticks)
    plt.xlim(1, 20)
    plt.show()
    return 1

if __name__ == '__main__':
    main()
