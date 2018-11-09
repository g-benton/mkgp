from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import math
import torch
from gpytorch.kernels import RBFKernel
from torch.autograd import Variable
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.lazy import BlockLazyTensor, LazyTensor, NonLazyTensor

import sys
sys.path.append("/Users/greg/Google Drive/Fall 18/ORIE6741/mkgp/explore/")

from multi_kernel_tensor import MultiKernelTensor

class multi_kernel(Kernel):
    def __init__(self, lengthscales=None, rank=1, batch_size=1, task_covar_prior=None):
        super(multi_kernel, self).__init__()
        self.num_tasks = len(lengthscales)
        self.lengthscales = lengthscales
        # self.has_custom_exact_predictions = True

    # @property
    # def has_custom_exact_predictions(self):
    #     return True


    def sq_exp_fun(self, x1, x2, length, **params):
        x1_ = x1.div(length)
        x2_ = x2.div(length)
        x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

        diff = (x1_ - x2_).norm(2, dim=-1)
        return diff.pow(2).div_(-2).exp_()

    def sq_exp_mix(self, x1, x2, length1, length2, **params):

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

        mat_list = [None for _ in range(self.num_tasks**2)]

        inder = -1
        for t1 in range(self.num_tasks):
            for t2 in range(self.num_tasks):
                inder += 1
                if t1 == t2:
                    mat_list[inder] = self.sq_exp_fun(x1, x2, self.lengthscales[t1], **params)
                else:
                    mat_list[inder] = self.sq_exp_mix(x1, x2, self.lengthscales[t1], self.lengthscales[t2], **params)
                    #____ MAT ___ = mix kernel mat

        ## combine matrices ##
        ## ONLY WORKS FOR TWO TASKS, FIX LATER ##
        # print(mat_list[0])
        row1 = torch.cat([mat_list[0], mat_list[1]], 2)
        row2 = torch.cat([mat_list[2], mat_list[3]], 2)

        multi_task_mat = torch.cat([row1, row2], 1)
        # print(NonLazyTensor(multi_task_mat).tensor)
        return multi_task_mat

    # def forward(self, x1, x2, **params):
    #     # print("HIT KERNEL GENERATION")
    #     # print(x1)
    #     # print(x2)
    #     x1_len = len(x1[0])
    #     x2_len = len(x2[0])
    #
    #     x1 = x1.div(self.lengthscales[0])
    #     x2 = x2.div(self.lengthscales[1])
    #     x1, x2 = self._create_input_grid(x1, x2, **params)
    #
    #     diff_mat = (x1 - x2).norm(2, dim=-1) # gives the matrix of ||x - x'|| for all pairs
    #
    #     # TODO: get block matrix of kernels
    #     multi_task_mat = torch.Tensor()
    #     multi_task_mat = multi_task_mat.new_empty((1, self.num_tasks*x1_len, self.num_tasks*x2_len))
    #     # multi_task_mat = multi_task_mat.new_empty((self.num_tasks**2, dat_len, dat_len))
    #
    #     inder = -1
    #     for t1 in range(self.num_tasks):
    #         for t2 in range(self.num_tasks):
    #             inder += 1
    #
    #             row_start = t1*x1_len
    #             col_start = t2*x2_len
    #
    #             # print(row_start, row_end, col_start, col_end)
    #             if t1 == t2:
    #                 multi_task_mat[0][row_start:(row_start+x1_len), col_start:(col_start+x2_len)] = self.sq_exp_fun(x1, x2, self.lengthscales[t1])
    #                 # multi_task_mat[inder] = self.sq_exp_fun(x1, x2, self.lengthscales[t1])
    #
    #             else:
    #                 multi_task_mat[0][row_start:(row_start+x1_len), col_start:(col_start+x2_len)] = self.sq_exp_mix(x1, x2,
    #                                     self.lengthscales[t1], self.lengthscales[t2])
    #                 # multi_task_mat[inder] = self.sq_exp_mix(x1, x2,
    #                                             # self.lengthscales[t1], self.lengthscales[t2])
    #     return NonLazyTensor(multi_task_mat)



        """
        I was confused about where exact_predictive_mean/covar were getting called,
        everything below should be unnecessary
        """

        # def exact_predictive_mean(full_covar, full_mean, train_labels, num_train, likelihood,
        #                           precomputed_cache=None, non_batch_train=False):
        #
        #     print("DID WE GET HOME")
        #     """
        #     Custom method for doing MK/MT predictions,
        #     passed inputs from lazy_evaluated_kernel_tensor:
        #         full_covar: lazy_evaluated_kernel_tensor holding info to generate covar.
        #         full_mean: mean of X's (where is this computed) ** BROKEN
        #         train_labels: concatenated vector of training Y's
        #         num_train: number of points corresponding to training
        #         likelihood: Likelihood Module (Mutlitask Gaussian Likelihood)
        #         precomputed_cache: None ** worry about this later
        #     """
        #     print("in multi_kernel: full_covar.kernel = ", full_covar.kernel)
        #     return pred_mean
        #
        # def exact_predictive_covar(num_train, likelihood,
        #                            precomputed_cache=None, non_batch_train=False):
        #
        #     return pred_covar

    # def size(self, x1, x2):
    #
    #     return torch.Size(something??)

def tester():

    t_num = 5
    for t1 in range(t_num):
        for t2 in range(t1, t_num):
            print(t1, t2)
    import torch
    mat = torch.Tensor()
    mat = mat.new_empty((5,5))

    test = torch.linspace(1, 5, 50)
    test.shape

    temp_mat = torch.linspace(0, 100**2, steps=100**2)
    temp_mat = temp_mat.view(100, 100)

    test_mat = torch.Tensor()
    test_mat = test_mat.new_empty((1, 200, 200))
    dat_len = 100
    for t1 in range(2):
        for t2 in range(2):
            row_start = t1*dat_len
            col_start = t2*dat_len
            if (t1 + t2)%2 == 0:
                test_mat[0][row_start:(row_start + dat_len), col_start:(col_start + dat_len)] = temp_mat
            else:
                test_mat[0][row_start:(row_start + dat_len), col_start:(col_start + dat_len)] = torch.transpose(temp_mat, 0, 1)
    import matplotlib.pyplot as plt
    import numpy as np
    data = np.random.random((10,10))
    plt.imshow(test_mat[0].numpy())


    return None
