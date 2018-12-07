import torch
import gpytorch
import math
import random
import sys
import matplotlib.pyplot as plt

sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/multi-task/")
import mk_kernel

def gen_correlated_rbfs(full_x, _num_tasks=2):
    _num_tasks=2
    full_x = torch.linspace(0, 1, 100)
    class MultitaskModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=_num_tasks
            )
            self.covar_module = mk_kernel.MultiKernel(
                [gpytorch.kernels.RBFKernel() for _ in range(_num_tasks)]
            )

            # self.covar_module = gpytorch.kernels.ScaleKernel(mk_kernel.multi_kernel())
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=_num_tasks)
    model = MultitaskModel(torch.tensor(0), torch.tensor(0), lh)

    ## setup random hyperparameters ##
    # lengths = [math.log(random.uniform(1, 20)) for _ in range(_num_tasks)]
    # test = torch.Tensor([1, 2])
    # model.covar_module.output_scale_kernel.covar_matrix.evaluate()
    # model.covar_module.output_scale_kernel.covar_factor = torch.nn.Parameter(torch.Tensor([[1, math.log(2)],[math.log(2), 1]]))
    # model.covar_module.output_scale_kernel.covar_matrix.evaluate()

    prior_pred = model.forward(full_x)

    sample = prior_pred.sample(sample_shape=torch.Size([1]))
# plt.plot(sample[0, :, 0].detach().numpy())
# plt.plot(sample[0, :, 1].detach().numpy())
# plt.show()

    return sample[0] + 5
