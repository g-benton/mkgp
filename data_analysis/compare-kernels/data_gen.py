import math
import torch
import gpytorch
import matplotlib.pyplot as plt


def data_gen(test_points, num_samples=1):
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    ## set up models ##
    likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
    model1 = ExactGPModel(None, None, likelihood1)
    model1.likelihood.log_noise.data[0][0] = math.log(1)
    model1.covar_module.log_outputscale.data[0] = math.log(20)
    model1.covar_module.base_kernel.log_lengthscale.data[0,0,0] = math.log(10)

    likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
    model2 = ExactGPModel(None, None, likelihood2)
    model2.likelihood.log_noise.data[0][0] = math.log(3)
    model2.covar_module.log_outputscale.data[0] = math.log(8)
    model2.covar_module.base_kernel.log_lengthscale.data[0,0,0] = math.log(1)

    ## DRAW SAMPLES ##
    model1.eval();
    likelihood1.eval();
    preds = likelihood1(model1(test_points))
    mod1_mean = preds.mean
    mod1_samples = preds.sample(sample_shape=torch.Size([num_samples]))

    model2.eval();
    likelihood2.eval();
    preds = likelihood2(model2(test_points))
    mod2_mean = preds.mean
    mod2_samples = preds.sample(sample_shape=torch.Size([num_samples]))
    for sample in range(num_samples):
        mod2_samples[sample, :] = mod1_samples[sample, :] + mod2_samples[sample, :]
    ## there's more ##
    return mod1_samples, mod1_mean, mod2_samples, mod2_mean


def main():
    test_data = torch.linspace(0,10, 100)

    mod1, mean1, mod2, mean2 = data_gen(test_data, num_samples=1)
    # mod1 = mod1[0]
    # mod2 = mod2[0]

    plt.subplot(1, 2, 1)
    plt.plot(test_data.numpy(), mod1[0].numpy(), 'b*')
    # plt.plot(test_data.numpy(), mean1.detach().numpy())
    plt.subplot(1, 2, 2)
    plt.plot(test_data.numpy(), mod2[0].numpy(), 'b*')
    # plt.plot(test_data.numpy(), mean2.detach().numpy())
    plt.show()




if __name__ == '__main__':
    main()
