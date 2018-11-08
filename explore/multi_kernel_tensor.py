import math
import torch
import gpytorch


class MultiKernelTensor(gpytorch.lazy.LazyTensor):
    """
    This is just a placeholder class for now to get things runningself.

    covar_mat contains block matrix:
    [K(X, X), K(X*, X);  K(X, X*), K(X*, X*)]

    MKGP function a bit differently than current implementation of multi-task method,
    so this was necessary to implement custom exact prediction methods
    """
    def __init__(self, covar_mat):
        super(MultiKernelTensor, self).__init__(covar_mat)
        self.covar_mat = covar_mat

    def _size(self):

        return self.covar_mat.size()

    def _get_indices(self, left_indices, right_indices):
        """
        What does this do?
        """
        # print("left_inds = ", left_indices)
        return self.covar_mat[left_indices, right_indices]

    def _transpose_nonbatch(self):

        return torch.t(self.covar_mat)

    def _matmul(self, rhs):
        # print("covar mat shape = ", self.covar_mat.shape)
        # print("rhs shape = ", rhs.shape)
        rtn = torch.mm(self.covar_mat, rhs)
        # print("returning something shape = ", rtn.shape)
        return rtn

    ## HOPEFULLY WONT USE THE STUFF BELOW HERE ##
    def exact_predictive_mean(self, full_mean, train_labels, num_train,
                              likelihood, precomputed_cache=None):
        """
        inputs:
            self: contains covar_mat
            full_mean: mean of X's (where is this computed) ** BROKEN
            train_labels: concatenated vector of training Y's
            num_train: number of points corresponding to training
            likelihood: Likelihood Module (Mutlitask Gaussian Likelihood)
            precomputed_cache: None


        returns:
            predictive mean:(same dimension as ??)
            precomputed_cache: train_train_covar.inv_matmul(train_labels_offset)
        Just boring old predictive inference, nothing computationally fancy for now
        """
        print(self.covar_mat.shape)
        # print(full_mean.shape)
        pred_mean = 1

        return pred_mean


    def exact_predictive_covar(self, num_train, likelihood, precomputed_cache=None):
        """
        Just regular covariance prediction, nothing fancy
        """
        return pred_covar
