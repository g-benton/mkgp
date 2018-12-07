#! /usr/bin/env python3

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
import math
import torch
from torch.nn import ModuleList
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.kernels import RBFKernel, IndexKernel
#from gpytorch.kernels import IndexKernel
from gpytorch.lazy import LazyTensor, NonLazyTensor, KroneckerProductLazyTensor, BlockDiagLazyTensor

# def kronecker_product(t1, t2,size_t1,size_t2):
#     fusion_tensor = torch.bmm(t1.unsqueeze(2), t2.unsqueeze(1))
#     fusion_tensor = fusion_tensor.view(-1, size_t1 * size_t2)
#     return fusion_tensor
#
# def basis_vec(size, ind):
#     vec = torch.zeros((size))
#     vec[ind] = 1
#     return vec

class MultiKernel(Kernel):
    """
    Kernel supporting Kronecker style multitask Gaussian processes (where every data point is evaluated at every
    task) using :class:`gpytorch.kernels.IndexKernel` as a basic multitask kernel.

    Given a base covariance module to be used for the data, :math:`K_{XX}`, this kernel computes a task kernel of
    specified size :math:`K_{TT}` and returns :math:`K = K_{TT} \otimes K_{XX}`. as an
    :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.

    Args:
        data_covar_module (:obj:`gpytorch.kernels.Kernel`):
            Kernel to use as the data kernel.
        num_tasks (int):
            Number of tasks
        batch_size (int, optional):
            Set if the MultitaskKernel is operating on batches of data (and you want different
            parameters for each batch)
        rank (int):
            Rank of index kernel to use for task covariance matrix.
        task_covar_prior (:obj:`gpytorch.priors.Prior`):
            Prior to use for task kernel. See :class:`gpytorch.kernels.IndexKernel` for details.
    """

    def __init__(
            self,
            kernel_list = None,
            batch_size=1,
            task_covar_prior=None
            ):
        super(MultiKernel, self).__init__()
        self.num_tasks = len(kernel_list)
        self.output_scale_kernel = IndexKernel(num_tasks=self.num_tasks, batch_size=1,
                                               rank=self.num_tasks, prior=task_covar_prior)
        self.batch_size = batch_size
        self.in_task_covar = ModuleList(
            [
                kernel_list[ii] for ii in range(self.num_tasks)
            ]
        )
    def perm_matrices(self, x1, x2):
        m = self.num_tasks
        n = self.num_tasks
        r = x1.numel()
        q = x2.numel()

        l_perm = torch.zeros((r*m, r*m))
        for row in range(r*m):
            ind = ((row)%m)*r + math.floor((row)/m)
            # print("lperm row", row, " col", ind)
            l_perm[row, ind] = 1

        r_perm = torch.zeros((q*n, q*n))
        for row in range(q*n):
            ind = ((row)%q)*n + math.floor((row)/q)
            r_perm[row, ind] = 1

        return l_perm, r_perm

    def sq_exp_mix(self, x1, x2, length1, length2, **params):
        # length1 = math.exp(length1)
        # length2 = math.exp(length2)
        x1_, x2_ = self._create_input_grid(x1, x2, **params)
        diff = (x1_ - x2_).norm(2, dim=-1)
        # print(diff)
        # print(diff.pow(2).div(-1).exp_())
        pre_term = math.sqrt((2*length1 * length2)/(length1**2 + length2**2))
        return pre_term * diff.pow(2).div(-1*(length1**2 + length2**2)).exp_()

    def mat_rbf_covar(self, x1, x2, mat_len, rbf_len, **params):
        # set up #
        lmbd = math.sqrt(3.0)/2.0 * rbf_len/mat_len
        x1_, x2_ = self._create_input_grid(x1, x2, **params)
        diff = (x1_ - x2_).norm(2, dim=1)
        scaled_diff = diff.mul(math.sqrt(3.0)).div(mat_len)

        # compute #
        pre_term = math.sqrt(lmbd) * (math.pi/2.0)**(0.25) * math.exp(lmbd**2)
        in_term = (
                   2 * torch.cosh(scaled_diff) -
                   scaled_diff.exp().mul(torch.erf(lmbd + diff.div(rbf_len))) -
                   scaled_diff.mul(-1).exp().mul(torch.erf(lmbd - diff.div(rbf_len)))
                  )
        # print(pre_term * in_term)
        return pre_term * in_term
        # return pre_term * in_term

    def mat_mat_covar(self, x1, x2, len1, len2, **params):
        x1_, x2_ = self._create_input_grid(x1, x2, **params)
        dff = (x1_ - x2_).norm(2, dim=1)

        # pre_term = 2 * math.sqrt(len1 * len2)/(len1**2 - len2**2)
        term1 = diff.mul(-math.sqrt(3.0)/len1).exp().mul(len1)
        term2 = diff.mul(-math.sqrt(3.0)/len2).exp().mul(len2)

        # return pre_term*(term1 - term2)
        return (term1 - term2)
    def get_inter_covar(self, x1, x2, kern1, kern2, **params):
        """
        Takes in two kernels and returns inter-task covariance
        """
        # just helpful to keep things clean #
        rbf_kern = gpytorch.kernels.rbf_kernel.RBFKernel
        mat_kern = gpytorch.kernels.matern_kernel.MaternKernel
        # print("is it catching either?")
        # print(kern1)
        # print(isinstance(kern1, rbf_kern))
        # print(isinstance(kern1, mat_kern))
        # case 1: rbf x rbf
        if(isinstance(kern1, mat_kern) and isinstance(kern2, rbf_kern)):
            return self.mat_rbf_covar(x1, x2, kern1.lengthscale.item(), kern2.lengthscale.item(), **params)
        elif(isinstance(kern1, rbf_kern) and isinstance(kern2, mat_kern)):
            return self.mat_rbf_covar(x1, x2, kern2.lengthscale.item(), kern1.lengthscale.item(), **params)
        elif(isinstance(kern1, rbf_kern) and isinstance(kern2, rbf_kern)):
            return self.sq_exp_mix(x1, x2, kern1.lengthscale.item(), kern2.lengthscale.item(), **params)
        elif(isinstance(kern1, mat_kern) and isinstance(kern2, mat_kern)):
            return self.mat_mat_covar(x1, x2, kern1.lengthscale.item(), kern2.lengthscale.item(), **params)
        else:
            raise RuntimeError("Make sure kernels are either RBF or Matern 3/2")

    def size(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariancn matrix.
        """
        non_batch_size = (self.num_tasks * x1.size(-2), self.num_tasks * x2.size(-2))
        if x1.ndimension() == 3:
            return torch.Size((x1.size(0),) + non_batch_size)
        else:
            return torch.Size(non_batch_size)


    def forward(self, x1, x2, **params):
        n = torch.numel(x1)
        m = torch.numel(x2)
        # print("n = ", n)

        block_tens = torch.zeros((self.num_tasks**2, n, m))

        scale_mat = self.output_scale_kernel.covar_matrix.evaluate()
        # print(scale_mat[0][0][1])
        inder = -1
        for t1 in range(self.num_tasks):
            for t2 in range(self.num_tasks):
                inder += 1
                ## within task covariance
                if t1 == t2:
                    block_tens[inder] = scale_mat[0][t1][t2] * self.in_task_covar[t1].forward(x1, x2, **params)
                    # print(block_tens[inder])
                ## between tasks ##
                else:
                    # TODO: figure out the indexing to avoid redundant computations
                    # block_tens[inder] = scale_mat[0][t1][t2] * self.sq_exp_mix(x1, x2, self.in_task_covar[t1].lengthscale.item(),
                    #                 self.in_task_covar[t2].lengthscale.item(), **params)
                    block_tens[inder] = scale_mat[0][t1][t2] * self.get_inter_covar(x1, x2,
                                                            self.in_task_covar[t1], self.in_task_covar[t2], **params)
                    # print(block_tens[inder])


        ## COMBINE INTO COVARIANCE ##
        rows = torch.zeros((self.num_tasks, n, m*self.num_tasks))

        # make tensor of rows to concatenate #
        scale_inder = -1
        for row_ind in range(len(rows)):
            scale_inder += 1
            to_cat = tuple([block_tens[row_ind*self.num_tasks + ii]
                                for ii in range(self.num_tasks)])
            rows[row_ind] = torch.cat(to_cat, 1)

        # concatenate rows #
        to_cat = tuple([rows[ii] for ii in range(rows.shape[0])])
        multi_task_mat = torch.cat(to_cat, 0)

        ## permute so we are represented the way gpytorch wants ##
        l_perm, r_perm = self.perm_matrices(x1, x2)
        temp = l_perm.mm(multi_task_mat)
        # print(r_perm)
        multi_task_mat = temp.mm(r_perm)
        # print(NonLazyTensor(multi_task_mat).tensor)
        return multi_task_mat
