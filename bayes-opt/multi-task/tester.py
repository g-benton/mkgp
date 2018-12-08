import torch
import gpytorch
import math
import sys
import random
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/bayes-opt/multi-task/")
import mk_kernel
from gen_correlated_rbfs import gen_correlated_rbfs
from helper_functions import expected_improvement

if __name__ == '__main__':
    low_x = 0
    high_x = 1
    num_pts = 100

    end_sample_count = 30 # this seems like a bad criteria...
    full_x = torch.linspace(0, 1, num_pts)
    n_tasks = 2
    full_y = gen_correlated_rbfs(full_x, _num_tasks=n_tasks)
    _num_tasks = full_y.shape[1]

    # get out starting points #
    n_start = 5
    start_samples = random.sample(range(num_pts), n_start)
    obs_x = full_x[start_samples]
    obs_y = full_y[start_samples, :]
    obs_y.shape
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

    lh = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    lh.log_noise
    lh.log_noise.data[0,0] = -8
    lh.log_noise

    lh2 = gpytorch.likelihoods.GaussianLikelihood()
    lh2.log_noise
    lh2.log_noise.data[0,0] = -8
    lh2.log_noise
    # fake_x = torch.Tensor([0.5]).type(torch.FloatTensor)
    # fake_y = torch.tensor([[10,10]]).type(torch.FloatTensor)
    model = MultitaskModel(obs_x, obs_y, lh)
    # lekern = model.covar_module(full_x)
    # lekern.diag()
    # print(lekern.evaluate())
    # print(lekern.diag())

    lh.eval();
    model.eval();
    pred = lh(model(full_x))
    pred.covariance_matrix
    pred.stddev
    import matplotlib.pyplot as plt
plt.plot(full_x.numpy(), pred.stddev[:, 0].detach().numpy())
plt.plot(obs_x.numpy(), obs_y[:, 0].numpy(), "b*")
plt.show()


    class MultitaskModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=_num_tasks
            )
            self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(),
                                                                num_tasks=2, rank=1)
            # self.covar_module = gpytorch.kernels.ScaleKernel(mk_kernel.multi_kernel())
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    model = MultitaskModel(obs_x, obs_y, lh)
    # lekern = model.covar_module(full_x)
    # print(lekern.diag())



    # model.train();
    # lh.train();
    #
    # optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=0.01)
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, model)
    #
    # n_iter = 20
    # for i in range(n_iter):
    #     optimizer.zero_grad()
    #     output = model(obs_x)
    #     loss = -mll(output, obs_y)
    #     loss.backward()
    #     optimizer.step()
    #
    # model.eval();
    # lh.eval();
    #
    # with torch.no_grad(), gpytorch.fast_pred_var():
    #     pred_model = lh(model(full_x))
    #
    # lower, upper = pred_model.confidence_region()
    # print(lower)

y = torch.randn([5])
list(y.numpy())
def bleh():
    return 1, 2
 _ , here = bleh()
