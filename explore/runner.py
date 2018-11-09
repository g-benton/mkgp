import math
import numpy as np
from scipy.io import loadmat
import torch
import gpytorch
from matplotlib import pyplot as plt
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/explore/")

# import local_gpt as gpytorch
import mk_kernel

class MultitaskModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        # self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = mk_kernel.multi_kernel(lengthscales=torch.Tensor([1, 5]))
        # self.covar_module = gpytorch.kernels.ScaleKernel(mk_kernel.multi_kernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def main():

    ## set up data ##
    train_x = torch.linspace(0, 1, 100)
    test_x = torch.linspace(0.1, 1.1, 52)
    train_y = torch.stack([torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,], -1)
    # train_y1 = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
    # train_y2 = torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
    # train_y = torch.cat((train_y1, train_y2), 0)

    ## set up model ##
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskModel(train_x, train_y, likelihood)

    # print("custom method???", model.covar_module.has_custom_exact_predictions)

    model.eval();
    likelihood.eval();
    f_pred = model(test_x)

    f_pred.covariance_matrix[0]


    sample = f_pred.sample(sample_shape = torch.Size([0]))
    # Initialize plots
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3));

    # Make predictions
    # with torch.no_grad(), gpytorch.fast_pred_var():
    #     test_x = torch.linspace(0, 1, 51)
    #     predictions = likelihood(model(train_x))
    #     mean = predictions.mean
    #     lower, upper = predictions.confidence_region()


    # model.eval();
    # f_preds = model(test_data);
    # y_preds = likelihood(model(test_data))
    # means = y_preds.mean
    #
    # plt.plot(test_data, means[:, 0])
    # plt.plot(test_data, means[:, 1])
        # samples = f_preds.sample(sample_shape=torch.Size([1]));

if __name__ == '__main__':
    main()


def testing():
    x1 = test_data
    x1 = torch.tensor([[[[0.0500]],  [[0.1000]], [[0.1500]]]])
    x2 = torch.tensor([[[[0.0500],
              [0.1000],
              [0.1500]]]])


    diff = (x1 - x2).norm(2, dim=-1)
    diff.shape
    diff
