import math
import torch
import gpytorch
import random
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/testing/")
import mk_kernel
n_tasks = 4
full_x = torch.linspace(0, 10, 100)
def data_gen(full_x, n_tasks=2):
    lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)

    class MultitaskModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=n_tasks
            )
            self.covar_module = mk_kernel.MultiKernel(
                [gpytorch.kernels.RBFKernel() for _ in range(n_tasks)]
            )
            # self.covar_module = mk_kernel.MultitaskRBFKernel(num_tasks=2, rank=2)
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


    model = MultitaskModel(torch.tensor(0), torch.tensor(0), lh)

    ## overwrite covariance parameters and lenghts ##
    model.covar_module.output_scale_kernel.covar_factor = torch.nn.Parameter(2*model.covar_module.output_scale_kernel.covar_factor)
    lengths = [math.log(random.randint(1, 15)) for _ in range(n_tasks)]
    lengths = [1, 10, 15, 100]
    lenghts = [math.log(l) for l in lengths]
    for tt in range(n_tasks):
        model.covar_module.in_task_covar[tt].log_lengthscale.data[0,0,0] = lengths[tt]

    prior_pred = model.forward(full_x)
    sample = prior_pred.rsample(torch.Size((1,)))[0]

    return sample


    # plt.plot(sample[:, 0].detach().numpy())
    # plt.plot(sample[:, 1].detach().numpy())
    # plt.plot(sample[:, 2].detach().numpy())
    # plt.plot(sample[:, 3].detach().numpy())
    # plt.show()
