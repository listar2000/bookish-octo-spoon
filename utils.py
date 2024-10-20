"""
Utility functions for calculating SURE, etc.
"""
from scipy.optimize import minimize_scalar

def _minimize_lbfgs(fun, args=(), bounds=None):
    return minimize_scalar(fun, bounds=bounds, method='bounded', args=args).x


if __name__ == '__main__':
    # create a toy function to test the minimization
    def fun(x):
        return (x - 2) ** 2
    
    # test the minimization
    print(_minimize_lbfgs(fun, bounds=(0, 4)))
