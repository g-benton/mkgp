import math
import torch
import gpytorch


temp_mat = torch.Tensor()
temp_mat = temp_mat.new_empty((3, 5, 5))

lazy_mat = gpytorch.lazy.BlockLazyTensor(temp_mat)

dim = 10
test_mat = torch.Tensor([i for i in range(dim**2)])
test_mat = test_mat.view(-1, dim)

l_inds = torch.Tensor([0,1,2,3]).long()
r_inds = torch.Tensor([0,1,2,3]).long()

n_indices = l_inds.numel()
