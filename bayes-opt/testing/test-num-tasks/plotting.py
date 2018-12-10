import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dat = np.load("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/test-num-tasks/num_task_data.npz")
    dat.files
    one_maxes = dat["one_maxes"]
    two_maxes = dat["two_maxes"]
    three_maxes = dat["three_maxes"]
    four_maxes = dat["four_maxes"]

    one_avg = np.mean(one_maxes, axis=0)
    two_avg = np.mean(two_maxes, axis=0)
    three_avg = np.mean(three_maxes, axis=0)
    four_avg = np.mean(four_maxes, axis=0)

    ## more setup ##
    ticks = [ii+1 for ii in range(len(one_avg))]
    plot_cols = sns.color_palette("muted", 4)

    ## plotting ##
    plt.plot(ticks, one_avg, c=plot_cols[0], ls='-', linewidth=2)
    plt.plot(ticks, two_avg, c=plot_cols[1], ls='-', linewidth=2)
    plt.plot(ticks, three_avg, c=plot_cols[2], ls='-', linewidth=2)
    plt.plot(ticks, four_avg, c=plot_cols[3], ls='-', linewidth=2)
    plt.axhline(y=10, ls="-.", c="k")
    plt.legend(["1 Task", "2 Tasks", "3 Tasks", "4 Tasks"])
    plt.title("Averaged Maximum Found per Iteration")
    plt.ylabel("Maximum Found")
    plt.xlabel("Iteration")
    plt.xticks(ticks)
    plt.xlim(1, 20)
    plt.show()
