import math
import torch
import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns

def data_gen(test_points, num_samples=1):
    # test_points = torch.linspace(0, 10, 100)
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
    sd1 = 0.1
    sd2 = 0.3
    likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
    model1 = ExactGPModel(None, None, likelihood1)
    model1.likelihood.log_noise.data[0][0] = math.log(sd1)
    model1.covar_module.log_outputscale.data[0] = math.log(40)
    model1.covar_module.base_kernel.log_lengthscale.data[0,0,0] = math.log(10)

    likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
    model2 = ExactGPModel(None, None, likelihood2)
    model2.likelihood.log_noise.data[0][0] = math.log(sd2)
    model2.covar_module.log_outputscale.data[0] = math.log(4)
    model2.covar_module.base_kernel.log_lengthscale.data[0,0,0] = math.log(1)

    ## DRAW SAMPLES ##
    model1.eval();
    likelihood1.eval();
    preds = likelihood1(model1(test_points))
    mod1_means = model1(test_points).sample(sample_shape=torch.Size([num_samples]))
    model2.eval();

    likelihood2.eval();
    preds = likelihood2(model2(test_points))
    mod2_means = model2(test_points).sample(sample_shape=torch.Size([num_samples]))
    # mod1_samples = preds.sample(sample_shape=torch.Size([num_samples]))
    mod1_samples = torch.zeros(mod1_means.shape)
    mod2_samples = torch.zeros(mod2_means.shape)

    mod1_means.shape
    for samp in range(num_samples):
        mod1_samples[samp, :] = torch.normal(mean=mod1_means[samp, :], std=sd1)
        mod2_means[samp, :] = mod1_means[samp, :] + mod2_means[samp, :]
        # mod2_samples[samp, :] = mod1_means[samp, :] + mod2_samples[samp, :]
        mod2_samples[samp, :] = torch.normal(mean=mod2_means[samp, :], std=sd2)
    # mod2_samples = preds.sample(sample_shape=torch.Size([num_samples]))
    # for sample in range(num_samples):
    ## there's more ##
    return mod1_samples, mod1_means, mod2_samples, mod2_means


def main():
    test_data = torch.linspace(0,10, 1000)

    mod1, mean1, mod2, mean2 = data_gen(test_data, num_samples=1)
    # mod1 = mod1[0]
    # mod2 = mod2[0]
    true_col = sns.xkcd_palette(["windows blue"])[0]
    mod_col = sns.xkcd_palette(["amber"])[0]

    plt.figure()
    plt.title("Multi-Task Data")
    # plt.plot(test_data.numpy(), mod1[0].numpy(), 'b*')
    plt.plot(test_data.numpy(), mean1[0].numpy(), c=true_col)
    # plt.plot(test_data.numpy(), mean1.detach().numpy())
    # plt.subplot(1, 2, 2)
    # plt.plot(test_data.numpy(), mod2[0].numpy(), 'b*')
    plt.plot(test_data.numpy(), mean2[0].numpy(), c=mod_col)
    # plt.plot(test_data.numpy(), mean2.detach().numpy())
    plt.show()


if __name__ == '__main__':
    main()
