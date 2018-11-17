from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import math
import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.kernels import RBFKernel, IndexKernel
#from gpytorch.kernels import IndexKernel
from gpytorch.lazy import LazyTensor, NonLazyTensor, KroneckerProductLazyTensor, BlockDiagLazyTensor

def kronecker_product(t1, t2,size_t1,size_t2):
    fusion_tensor = torch.bmm(t1.unsqueeze(2), t2.unsqueeze(1))
    fusion_tensor = fusion_tensor.view(-1, size_t1 * size_t2)
    return fusion_tensor

def basis_vec(size, ind):
    vec = torch.zeros((size))
    vec[ind] = 1
    return vec

class MultitaskRBFKernel(Kernel):
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
            num_tasks,
            rank=1,
            batch_size=1,
            task_covar_prior=None):

        super(MultitaskRBFKernel, self).__init__()
        #self.task_covar_module = IndexKernel(
        #    num_tasks=num_tasks, batch_size=batch_size, rank=rank, prior=task_covar_prior
        #)
        # self.within_covar_module = gpytorch.kernels.RBFKernel()
        self.output_scale_kernel = IndexKernel(num_tasks=num_tasks, batch_size=1,
                                               rank=num_tasks, prior=task_covar_prior)
        # self.in_task_covar = [gpytorch.kernels.RBFKernel() for _ in range(num_tasks)]
        self.in_task1 = gpytorch.kernels.RBFKernel()
        self.in_task2 = gpytorch.kernels.RBFKernel()
        self.num_tasks = num_tasks
        self.batch_size = 1
        # We need gpytorch to know about the lengthscales - copied this from Kernel

        # self.register_parameter(
        #     name="log_task_lengthscales",
        #     parameter=torch.nn.Parameter(log_task_lengthscales)
        # )

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

    def sq_exp_fun(self, x1, x2, length, **params):
        length = math.exp(length)
        x1_ = x1.div(length)
        x2_ = x2.div(length)
        x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

        diff = (x1_ - x2_).norm(2, dim=-1)
        return diff.pow(2).div_(-2).exp_()

    def sq_exp_mix(self, x1, x2, length1, length2, **params):
        # length1 = math.exp(length1)
        # length2 = math.exp(length2)
        x1_, x2_ = self._create_input_grid(x1, x2, **params)
        diff = (x1_ - x2_).norm(2, dim=-1)
        # print(diff)
        # print(diff.pow(2).div(-1).exp_())
        pre_term = math.sqrt((2*length1 * length2)/(length1**2 + length2**2))
        return pre_term * diff.pow(2).div(-1*(length1**2 + length2**2)).exp_()

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

        # for kern in self.in_task_covar:
        #     print(kern.forward(x1, x2, **params))

        mat_list = [None for _ in range(self.num_tasks**2)]

        inder = -1
        for t1 in range(self.num_tasks):
            for t2 in range(self.num_tasks):
                inder += 1
                if t1 == t2:
                    if t1 == 0:
                        mat_list[inder] = self.in_task1.forward(x1, x2, **params)
                    else:
                        mat_list[inder] = self.in_task2.forward(x1, x2, **params)

                else:
                    mat_list[inder] = self.sq_exp_mix(x1, x2, self.in_task1.lengthscale.item(),
                                                      self.in_task2.lengthscale.item(), **params)
                    #____ MAT ___ = mix kernel mat

        ## combine matrices ##
        ## ONLY WORKS FOR TWO TASKS, FIX LATER ##
        # print(mat_list[0])

        scale_mat = self.output_scale_kernel.covar_matrix.evaluate().view(-1)

        for ind, scale in enumerate(scale_mat):
            mat_list[ind] = scale * mat_list[ind]

        row1 = torch.cat([mat_list[0], mat_list[1]], 2)
        row2 = torch.cat([mat_list[2], mat_list[3]], 2)

        multi_task_mat = torch.cat([row1, row2], 1)[0]

        l_perm, r_perm = self.perm_matrices(x1, x2)
        temp = l_perm.mm(multi_task_mat)
        # print(r_perm)
        multi_task_mat = temp.mm(r_perm)
        # print(NonLazyTensor(multi_task_mat).tensor)
        return multi_task_mat
