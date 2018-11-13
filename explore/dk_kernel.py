from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.kernels import RBFKernel
#from gpytorch.kernels import IndexKernel
from gpytorch.lazy import LazyTensor, NonLazyTensor, KroneckerProductLazyTensor, BlockDiagLazyTensor


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
            data_covar_module,
            num_tasks,
            rank=1,
            batch_size=1,
            task_covar_prior=None,
            log_task_outputscales=None):
        """
        """
        super(MultitaskRBFKernel, self).__init__()
        self.within_covar_module = gpytorch.kernels.RBFKernel()
        self.num_tasks = num_tasks
        self.batch_size = 1

    def forward(self, x1, x2, diag=False, batch_dims=None, **params):
        if batch_dims == (0, 2):
            raise RuntimeError("MultitaskRBFKernel does not accept the batch_dims argument.")

        covar_x1 = self.within_covar_module(x1, x2, **params)
        covar_x2 = self.within_covar_module(x1, x2, **params)
        for_diag = torch.stack(
            (covar_x1.evaluate_kernel()[0].evaluate(),
             covar_x2.evaluate_kernel()[0].evaluate()))
        res = BlockDiagLazyTensor(NonLazyTensor(for_diag))

        if diag:
            return res.diag()
        else:
            return res

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
