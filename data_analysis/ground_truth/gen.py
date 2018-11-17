import math
import numpy as np
from scipy.io import loadmat
import torch
import gpytorch
from matplotlib import pyplot as plt
import sys

sys.path.append("/Users/davidk/school/mkgp/data_analysis/")
import mk_kernel

# Domain will be x \in (0,10)
domain_x = torch.linspace(0,10,1000)

class MultitaskModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = mk_kernel.MultitaskRBFKernel(num_tasks=2)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskModel(torch.tensor(0),torch.tensor(0), likelihood)

# Different lengthscales
model.covar_module.in_task1.log_lengthscale.data[0][0][0] = math.log(3)
model.covar_module.in_task2.log_lengthscale.data[0][0][0] = math.log(0.3)

model.covar_module.output_scale_kernel.covar_factor.data[0] = torch.tensor([[3,10],[1,3]])
#model.covar_module.output_scale_kernel.covar_factor.data[0] = torch.tensor([[10,10],[10,10]])
model.covar_module.output_scale_kernel.log_var.data[0][0] = math.log(300)
model.covar_module.output_scale_kernel.log_var.data[0][1] = math.log(10)
print(model.covar_module.output_scale_kernel.covar_matrix.evaluate())

prior_pred = model.forward(domain_x)
sample = prior_pred.rsample(torch.Size((1,)))

plt.plot(domain_x.detach().numpy(),sample[0,:,0].detach().numpy())
plt.plot(domain_x.detach().numpy(),sample[0,:,1].detach().numpy())
plt.legend(['Task 1','Task 2'])
plt.show()

x = domain_x.detach().numpy()
y1 = sample[0,:,0].detach().numpy()
y2 = sample[0,:,1].detach().numpy()
data = np.column_stack([x,y1,y2])
np.savetxt("filename.csv",data,delimiter=',')
print(model.covar_module.output_scale_kernel.covar_matrix.evaluate())
print(model.covar_module.in_task1.lengthscale)
print(model.covar_module.in_task2.lengthscale)
