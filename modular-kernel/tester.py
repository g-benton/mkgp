import math
import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.kernels import RBFKernel, IndexKernel
#from gpytorch.kernels import IndexKernel
from gpytorch.lazy import LazyTensor, NonLazyTensor, KroneckerProductLazyTensor, BlockDiagLazyTensor



num_tasks = 3
n = 5
m = 3

test = torch.zeros((num_tasks**2, n, m))
test[0, :, :] = torch.ones((n, m))

torch.cat((test[0], test[1]), 1)

out = tuple([ test[ii] for ii in range(num_tasks)])
out[1]
test_out = torch.cat(out, 1)
3 * test_out
len(test)
