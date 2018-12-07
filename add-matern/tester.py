import torch
import math
import gpytorch
import scipy
kern = gpytorch.kernels.RBFKernel()
tt = gpytorch.kernels.rbf_kernel.RBFKernel
isinstance(kern, tt)

kern2 =gpytorch.kernels.MaternKernel()
isinstance(kern2, gpytorch.kernels.matern_kernel.MaternKernel)



## manually test function ##
rbf_len = 3.0
mat_len = 2.0
x1 = torch.linspace(1, 5, 5)
x2 = torch.linspace(1, 5, 5)

lmbd = math.sqrt(3.0)/2.0 * mat_len/rbf_len

diff = torch.rand((10, 5))
diff
lmbd
lmbd + diff
x1_, x2_ =  kern._create_input_grid(x1, x2)
diff = (x1_ - x2_).norm(2, dim=1)
scaled_diff = diff.mul(math.sqrt(3.0)).div(mat_len)

pre_term = math.sqrt(lmbd) * (math.pi/2.0)**(0.25) * math.exp(lmbd**2)
in_term = (2 * torch.cosh(scaled_diff) - scaled_diff.exp().mul(torch.erf(lmbd.expand_as(diff) + diff.div(rbf_len))) - scaled_diff.exp().mul(torch.erf(lmbd.expand_as(diff) - diff.div(rbf_len))))



test = torch.tensor(2.0)
test.exp()
test
test.mul(-1).exp()
