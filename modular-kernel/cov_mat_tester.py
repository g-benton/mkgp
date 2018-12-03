import torch
import gpytorch
import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/modular-kernel/")

import mk_kernel
import old_kernel
from data_gen import data_gen


class MultitaskModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
        gpytorch.means.ConstantMean(), num_tasks=2
        )
        # self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = mk_kernel.MultitaskRBFKernel(num_tasks=2,log_task_lengthscales=torch.Tensor([math.log(2.5), math.log(0.3)]))
        self.covar_module = mk_kernel.MultiKernel([gpytorch.kernels.RBFKernel(),
                                                   gpytorch.kernels.RBFKernel()])
        # self.covar_module = gpytorch.kernels.ScaleKernel(mk_kernel.multi_kernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



class OldModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(OldModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
        gpytorch.means.ConstantMean(), num_tasks=2
        )
        # self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = mk_kernel.MultitaskRBFKernel(num_tasks=2,log_task_lengthscales=torch.Tensor([math.log(2.5), math.log(0.3)]))
        self.covar_module = old_kernel.MultitaskRBFKernel(num_tasks=2)
        # self.covar_module = gpytorch.kernels.ScaleKernel(mk_kernel.multi_kernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

if __name__ == '__main__':

    train_x = torch.linspace(0, 10, 3)
    dat1, mean1,  dat2, mean2 = data_gen(train_x)
    train_y = torch.stack([dat1, dat2], -1)[0]

    like_1 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    new_mod = MultitaskModel(train_x, train_y, like_1)

    like_2 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    old_mod = OldModel(train_x, train_y, like_2)

    new_mod.eval();
    old_mod.eval();

    new_mod.covar_module.output_scale_kernel = old_mod.covar_module.output_scale_kernel
    new_mat = new_mod.covar_module(train_x).evaluate()
    print("new mat = ", new_mat)
    old_mat = old_mod.covar_module(train_x).evaluate()
    print("old mat = ", old_mat)


    old_mod.covar_module.output_scale_kernel.covar_matrix.evaluate()
