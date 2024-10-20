"""
All the estimators are represented as (pure) functions that in principle exerts no side-effects.
In the case that this shall be violated, define a higher-order function.
"""
import numpy as np
from datasets.dataset import PPIEmpBayesDataset
from utils import _minimize_lbfgs
from typing import Union

# Trivial estimators
def get_mle_estimators(data: PPIEmpBayesDataset) -> np.ndarray:
    return np.array([y.mean() for y in data.y_labelled])


def get_pred_mean_estimators(data: PPIEmpBayesDataset) -> np.ndarray:
    return np.array([pred.mean() for pred in data.pred_labelled])


# PPI estimators
def _get_generic_ppi_estimators(f_x_tilde: np.ndarray, f_x: np.ndarray, y: np.ndarray, lambda_: Union[float, np.ndarray]) -> np.ndarray:
    ppi = []
    for i in range(len(f_x_tilde)):
        ppi_i = f_x_tilde[i].mean() + lambda_ * (y[i].mean() - f_x[i].mean())
        ppi.append(ppi_i)
    return np.array(ppi)


def get_vanilla_ppi_estimators(data: PPIEmpBayesDataset) -> np.ndarray:
    return _get_generic_ppi_estimators(data.pred_unlabelled, data.pred_labelled, data.y_labelled, 1.0)


def get_power_tuned_ppi_estimators(data: PPIEmpBayesDataset, get_lambdas: bool = False) \
    -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    lambdas = []
    for i in range(data.M):
        n, N = data.ns[i], data.Ns[i]
        var_bar = np.concatenate([data.pred_labelled[i], data.pred_unlabelled[i]]).var()
        cov_bar = np.cov(data.pred_labelled[i], data.y_labelled[i])[0, 1]
        # compute the lambda for each problem
        lambda_i = (N / (n + N)) * cov_bar / var_bar
        lambda_i = np.clip(lambda_i, 0, 1)
        lambdas.append(lambda_i)

    ppi = _get_generic_ppi_estimators(data.pred_unlabelled, data.pred_labelled, data.y_labelled, np.array(lambdas))
    return ppi if not get_lambdas else (ppi, np.array(lambdas))


# Empirical Bayes (Shrinkage) estimators

# SURE estimator with heteroscedastic Var[Y_i] for each i
def get_eb_sure_estimators(data: PPIEmpBayesDataset, get_lambdas: bool = False, hetero_var_y: bool = False) \
    -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    # prepare observations for solving the minimum lambda

    var_y, f_x_tilde_bar, y_bar = [], [], []
    for i in range(data.M):
        n = data.ns[i]
        if hetero_var_y:
            var_y.append(data.pred_labelled[i].var() / n)
        f_x_tilde_bar.append(data.pred_unlabelled[i].mean())
        y_bar.append(data.y_labelled[i].mean())

    if not hetero_var_y:
        var_y = np.concatenate(data.pred_labelled).var()
        var_y = var_y / data.ns

    f_x_tilde_bar, y_bar, var_y = np.array(f_x_tilde_bar), np.array(y_bar), np.array(var_y)

    def sure_fn(lambda_: float) -> float:
        return np.sum((var_y / (var_y + lambda_) ** 2) \
                      * (var_y * (f_x_tilde_bar - y_bar) ** 2 + lambda_ ** 2 - var_y ** 2))
    
    # calculate upper search bound for lambda
    # TODO: make this a parameter
    cutoff = 0.99 
    lbd_upper = cutoff / (1 - cutoff) * var_y.max()
    optimal_lbd = _minimize_lbfgs(sure_fn, bounds=(0, lbd_upper))

    print(f"Optimal lambda: {optimal_lbd}, Cutoff: {cutoff}, Upper bound: {lbd_upper}")

    lambdas = optimal_lbd / (var_y + optimal_lbd)
    sure_estimates = lambdas * y_bar + (1 - lambdas) * f_x_tilde_bar

    return sure_estimates if not get_lambdas else (sure_estimates, lambdas)


# testing
if __name__ == '__main__':
    from datasets.amazon_review import AmazonReviewDataset

    review_ds = AmazonReviewDataset(verbose=True)
    print(get_mle_estimators(review_ds))

    sure_est, lambdas = get_eb_sure_estimators(review_ds, get_lambdas=True, hetero_var_y=True)
    print(sure_est)
    print(lambdas)
    