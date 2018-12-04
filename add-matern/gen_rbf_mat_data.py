import math
import numpy as np
import torch
import gpytorch
import sys
import matplotlib.pyplot as plt
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/add-matern/")

def main():
    xx = torch.linspace(0, 3, 100)
    yy = torch.sin(xx * 2 * 3.1415)
    class MatModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MatModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    like1 = gpytorch.likelihoods.GaussianLikelihood()
    mat_mod = MatModel(xx, yy, like1)

    like1.eval();
    mat_mod.eval();

    # test_x = torch.linspace(0, 5, 100)
    pred = like1(mat_mod(xx))
    pred_mean = pred.mean

    sample = pred.sample(sample_shape=torch.Size([1]))
    # plt.plot(sample[0].detach().numpy())
    # plt.show()
    # plt.plot(pred_mean.detach().numpy())
    # plt.show()

    np.savez("training_data.npz", test_x=xx.numpy(), mat_data=sample[0].detach().numpy(), rbf_data=pred_mean.detach().numpy())
if __name__ == '__main__':
    main()
