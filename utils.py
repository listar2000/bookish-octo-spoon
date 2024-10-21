"""
Utility functions for calculating SURE, etc.
"""
from scipy.optimize import minimize_scalar
import numpy as np


def _minimize_lbfgs(fun, args=(), bounds=None):
    return minimize_scalar(fun, bounds=bounds, method='bounded', args=args).x


# metrics
def get_mse(y_true: np.ndarray, esimates: np.ndarray) -> float:
    # Calculate the mean squared error between the true values and the estimates.
    return np.mean((y_true - esimates) ** 2)


if __name__ == '__main__':
    # create a toy function to test the minimization
    def fun(x):
        return (x - 2) ** 2
    
    # test the minimization
    assert _minimize_lbfgs(fun, bounds=(0, 4)) == 2.0, "Minimization failed"
