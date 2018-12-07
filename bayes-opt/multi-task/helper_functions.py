import torch
import gpytorch
import math
import random

def expected_improvement(means, vars, current_max):
    """
    Computes the expected improvement in the standard way
    """

    std_vals = (means - current_max).div(vars)
    std_normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    in_term = std_vals.mul(std_normal.cdf(std_vals)) + std_normal.log_prob(std_vals).exp()

    return vars.mul(in_term)


# means = torch.Tensor([ii for ii in range(1, 11)])
# vars = torch.Tensor(torch.randn(means.shape).abs())
# vars = torch.Tensor([ii for ii in range(1, 11)])
# means
# vars
# std_vals = (means - 2.0).div(vars)
# std_vals
# std_normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
# std_normal.variance
# std_normal.cdf(std_vals)
