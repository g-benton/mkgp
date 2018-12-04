import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():

    runtime_file = np.load("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/runtime-testing/runtimes.npz")
    runtime_file.files

    runtimes = runtime_file['runtimes']
    n_tasks = runtime_file['n_tasks']
    n_pts = runtime_file['n_pts']


    plt.contourf(n_tasks, n_pts, runtimes, 20, cmap="jet")
    plt.title("Contour Plot of Runtimes")
    plt.xlabel("Number of Tasks")
    plt.ylabel("Number of Points to Predict")
    plt.show()

    n_pts, n_tasks = np.meshgrid(n_tasks, n_pts)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(n_pts, n_tasks, runtimes)
    plt.title("Plot of Runtime")
    plt.ylabel("Num Pts")
    plt.xlabel("Number of Tasks")
    plt.show()


if __name__ == '__main__':
    main()



    
