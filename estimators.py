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
    """ Obtain the MLE estimator, i.e. the mean response for each problem.
    """
    return np.array([y.mean() for y in data.y_labelled])


def get_pred_mean_estimators(data: PPIEmpBayesDataset) -> np.ndarray:
    """ Obtain the prediction mean estimator for each problem.
    """
    return np.array([pred.mean() for pred in data.pred_unlabelled])


# PPI estimators
def _get_generic_ppi_estimators(f_x_tilde: np.ndarray, f_x: np.ndarray, y: np.ndarray, lambda_: Union[float, np.ndarray]) -> np.ndarray:
    """ Helper function to compute the PPI estimator for the PPI problem.

    Args:
        f_x_tilde (np.ndarray): the prediction mean of the unlabelled data for each product.

        f_x (np.ndarray): the prediction mean of the labelled data for each product.

        y (np.ndarray): the mean response (MLE) of the labelled data for each product.

        lambda (Union[float, np.ndarray]): the power-tuning parameter λ_i for each product.

    Returns:
        ppi: the PPI estimator for each product.
    """
    ppi = []
    flag = isinstance(lambda_, np.ndarray)
    for i in range(len(f_x_tilde)):
        lbd = lambda_[i] if flag else lambda_
        ppi_i = y[i].mean() + lbd * (f_x_tilde[i].mean() - f_x[i].mean())
        ppi.append(ppi_i)
    return np.array(ppi)


def get_vanilla_ppi_estimators(data: PPIEmpBayesDataset) -> np.ndarray:
    """ Obtain the vanilla PPI estimator for the PPI problem. This estimator is **non-compound**.
    
    The vanilla PPI estimator is given by:

    θ_i^PPI = θ_i + (μ_i - κ_i)

    where `θ_i` is the MLE, `μ_i`/`κ_i` are the prediction mean of the unlabelled/labelled data for the i^th problem.

    Args:
        data (PPIEmpBayesDataset): the dataset object.

    Returns:
        ppi_estimates: the vanilla PPI estimator for each product.

    References:
        [1] A. N. Angelopoulos, J. C. Duchi, and T. Zrnic, “PPI++: Efficient Prediction-Powered Inference”.
    """
    return _get_generic_ppi_estimators(data.pred_unlabelled, data.pred_labelled, data.y_labelled, 1.0)


def get_power_tuned_ppi_estimators(data: PPIEmpBayesDataset, get_lambdas: bool = False) \
    -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """ Obtain the power-tuned PPI estimator for the PPI problem. This estimator is **non-compound**.
    
    The power-tuning parameter λ_i for the i^th problem is given by:

    λ_i = (N_i / (n_i + N_i)) * Cov(Y_i, f(X_i)) / Var(f(X_i))

    where `Cov(Y_i, f(X_i))` is the sample covariance of the `n_i` paired labelled data and `Var(f(X_i))` \
        is the sample variance calculated from `n_i + N_i` unlabelled data. The final estimator is given by:

    θ_i^PPI = θ_i + λ_i * (μ_i - κ_i)

    where `θ_i` is the MLE, `μ_i`/`κ_i` are the prediction mean of the unlabelled/labelled data for the i^th problem.

    Args:
        data (PPIEmpBayesDataset): the dataset object.

        get_lambdas (bool): whether to return the power-tuning parameter λ_i. Default to `False`.
    
    Returns:
        ppi_estimates: the power-tuned PPI estimator for each product. If `get_lambdas` is `True`, the power-tuning parameters will \
        also be returned.

    References:
        [1] A. N. Angelopoulos, J. C. Duchi, and T. Zrnic, “PPI++: Efficient Prediction-Powered Inference”.
    """
    lambdas = []
    for i in range(data.M):
        n, N = data.ns[i], data.Ns[i]
        var_bar = np.concatenate([data.pred_labelled[i], data.pred_unlabelled[i]]).var(ddof=1)
        cov_bar = np.cov(data.pred_labelled[i], data.y_labelled[i], ddof=1)[0, 1]
        # compute the lambda for each problem
        lambda_i = (N / (n + N)) * cov_bar / var_bar
        lambda_i = np.clip(lambda_i, 0, 1)
        lambdas.append(lambda_i)

    ppi_estimates = _get_generic_ppi_estimators(data.pred_unlabelled, data.pred_labelled, data.y_labelled, np.array(lambdas))
    return ppi_estimates if not get_lambdas else (ppi_estimates, np.array(lambdas))


# Empirical Bayes (Shrinkage) estimators

# SURE estimator
def get_eb_sure_estimators(data: PPIEmpBayesDataset, get_lambdas: bool = False, hetero_var_y: bool = False, cutoff: float = 0.99) \
    -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """ Obtain the SURE-minimizing shrinkage estimator for the PPI problem.

    Let A_i denotes Var(Y_i) for the i^th product, θ_i be the MLE and μ_i be the prediction mean (to shrink towards).
    SURE(λ) = Σ_i (A_i / (A_i + λ)^2) * (A_i * (μ_i - θ_i)^2 + λ^2 - A_i^2)

    To minimize the above, we will resort to numerical optimization (via `scipy.optim`). The final estimator is given by:

    θ_i^SURE = λ_i * θ_i + (1 - λ_i) * μ_i, where λ_i = λ^* / (A_i + λ^*)

    Args:
        data (PPIEmpBayesDataset): the dataset object

        get_lambdas (bool): whether to return the shrinkage factor λ_i. Default to `False`.

        hetero_var_y (bool): whether to consider heteroscedastic A_i for each i. If `True`, A_i will be estimated based on \
            each sample variance of the observed Y_i. Otherwise, a global variance (across products) will be first computed \
            and then divided by the n_i to get A_i for each problem. Default to `False`.

        cutoff (float): the cutoff value for the maximum value of λ_i. This is an *ad-hoc* way of restraining the search space \
            for the optimal λ since λ_i = λ^* / (A_i + λ^*), i.e. we can calculate the upper bound of λ^* based on the \
            cutoff value and the maximum A_i. Default to `0.99`.

    Returns:
        sure_estimates: the SURE-minimizing shrinkage estimator for each product. If `get_lambdas` is `True`, the shrinkage factors will \
        also be returned.

    References:
        [1] X. Xie, S. C. Kou, and L. D. Brown, “SURE Estimates for a Heteroscedastic Hierarchical Model”.
    """
    # prepare observations for minimizing SURE
    var_y, f_x_tilde_bar, y_bar = [], [], []
    for i in range(data.M):
        n = data.ns[i]
        # N = data.Ns[i]
        if hetero_var_y:
            # var_y.append(data.pred_labelled[i].var(ddof=1) / N)
            var_y.append(data.y_labelled[i].var(ddof=1) / n)
        f_x_tilde_bar.append(data.pred_unlabelled[i].mean())
        y_bar.append(data.y_labelled[i].mean())

    if not hetero_var_y:
        # var_y = np.concatenate(data.pred_labelled).var(ddof=1)
        var_y = np.concatenate(data.y_labelled).var(ddof=1)
        var_y = var_y / data.ns

    f_x_tilde_bar, y_bar, var_y = np.array(f_x_tilde_bar), np.array(y_bar), np.array(var_y)

    def sure_fn(lambda_: float) -> float:
        return np.sum((var_y / (var_y + lambda_) ** 2) \
                      * (var_y * (f_x_tilde_bar - y_bar) ** 2 + lambda_ ** 2 - var_y ** 2))
    
    # calculate upper search bound for lambda
    assert 0 < cutoff < 1, "Cutoff must be in (0, 1)"
    lbd_upper = cutoff / (1 - cutoff) * var_y.max()
    optimal_lbd = _minimize_lbfgs(sure_fn, bounds=(0, lbd_upper))

    lambdas = optimal_lbd / (var_y + optimal_lbd)
    sure_estimates = lambdas * y_bar + (1 - lambdas) * f_x_tilde_bar

    return sure_estimates if not get_lambdas else (sure_estimates, lambdas)


def get_shrink_var_ppi_estimators(data: PPIEmpBayesDataset, get_lambdas: bool = False):
    """ Obtain the shrinkage variance PPI estimator for the PPI problem.

    This estimator is very similar to the `power-tuned PPI` estimator, with the difference that we estimate the sample variance differently.
    The power-tuning parameter λ_i for the i^th problem is now given by:

    λ_i = (N_i / (n_i + N_i)) * Cov(Y_i, f(X_i)) / V^*_i

    V^*_i = λ_v * V_m + (1 - λ_v) * Var(f(X_i))

    and V_m is the median of the vector V = (SV(f(X_1))), ..., (SV(f(X_M))), where SV(f(X_1)) denotes the sample variance of the prediction for the i^th problem.
    The variance shrinkage parameter λ_v is estimated by:

    λ_v = min(1, sum(SV2_i) / sum( (SV(f(X_i)) - V_m) ** 2 )

    where SV2_i is the `sample variance` of the `sample variance` of the prediction for the i^th problem. See [1] for more details.

    Args:
        data (PPIEmpBayesDataset): the dataset object.
        get_lambdas (bool): whether to return the power-tuning parameter λ_i after variance shrinkage. Default to `False`.

    Returns:
        shrink_var_ppi_estimates: the shrinkage variance PPI estimator for each product. If `get_lambdas` is `True`, the power-tuning parameters will \
        also be returned. Default to `False`.

    References:
        [1] R. Opgen-Rhein and K. Strimmer, “Accurate Ranking of Differentially Expressed Genes by a Distribution-Free Shrinkage Approach”.
    """
    # precedure of estimating the shrinkage variance parameter λ_v
    cov_bar = []
    # unbiased sample variance and sample variance of the sample variance
    sv, sv2 = [], []
    for i in range(data.M):
        N = data.Ns[i]
        all_pred = np.concatenate([data.pred_labelled[i], data.pred_unlabelled[i]])
        
        bar_f_x = all_pred.mean()
        ws = (all_pred - bar_f_x) ** 2
        bar_ws = ws.mean()

        sv.append( (N / (N - 1)) * bar_ws )
        sv2.append( N / (N - 1) ** 3 * np.sum((ws - bar_ws) ** 2) )

        cov_bar.append(np.cov(data.pred_labelled[i], data.y_labelled[i], ddof=1)[0, 1])
    
    # calculate the optimal λ_v
    sv, sv2 = np.array(sv), np.array(sv2)
    v_m = np.median(sv)
    lambda_v = np.sum(sv2) / np.sum((sv - v_m) ** 2)
    lambda_v = min(1, lambda_v)

    # calculate the individual λ_i from λ_v
    N, n = np.array(data.Ns), np.array(data.ns)
    lambdas = (N / (n + N)) * np.array(cov_bar) / (lambda_v * v_m + (1 - lambda_v) * sv)
    lambdas = np.clip(lambdas, 0, 1)

    shrink_var_ppi_estimates = _get_generic_ppi_estimators(data.pred_unlabelled, data.pred_labelled, data.y_labelled, lambdas)

    return shrink_var_ppi_estimates if not get_lambdas else (shrink_var_ppi_estimates, lambdas)


def get_compound_ppi_estimators(data: PPIEmpBayesDataset, get_lambda: bool = False):
    """ Obtain the compound PPI estimator for the PPI problem.

    TODO: add the description of the compound PPI estimator.

    Args:
        data (PPIEmpBayesDataset): the dataset object.

        get_lambda (bool): whether to return the power-tuning parameter λ_i. Default to `False`.
    
    Returns:
        ppi_estimates: the compound PPI estimator for each product. If `get_lambda` is `True`, the power-tuning parameters will \
        also be returned.
    """
    # step 1: aggregate the sum of covariance and variance terms
    numerator, denominator = 0, 0
    for i in range(data.M):
        n, N = data.ns[i], data.Ns[i]
        var_bar = np.concatenate([data.pred_labelled[i], data.pred_unlabelled[i]]).var(ddof=1)
        cov_bar = np.cov(data.pred_labelled[i], data.y_labelled[i], ddof=1)[0, 1]
        # compute the lambda for each problem
        numerator += cov_bar / n
        denominator += var_bar * ((N + n) / (N * n))

    lambda_ = numerator / denominator
    ppi_estimates = _get_generic_ppi_estimators(data.pred_unlabelled, data.pred_labelled, data.y_labelled, lambda_)
    return ppi_estimates if not get_lambda else (ppi_estimates, lambda_)


def get_mixed_compound_pt_ppi_estimators(data: PPIEmpBayesDataset, get_w: bool = False):
    """ TODO: add the description of the mixed compound-pt PPI estimator.
    """
    compound_ppi_estimates, lambda_cp = get_compound_ppi_estimators(data, get_lambda=True)
    pt_ppi_estimates, lambda_pt = get_power_tuned_ppi_estimators(data, get_lambdas=True)

    bias_square = np.array([(y.mean() - pred.mean()) ** 2 for y, pred in \
                            zip(data.y_labelled, data.pred_labelled)])

    w_numerator = np.sum((lambda_cp * (lambda_cp - lambda_pt)) * bias_square)
    w_denominator = np.sum((lambda_cp - lambda_pt) ** 2 * bias_square)
    w = w_numerator / w_denominator
    w = np.clip(w, 0, 1)

    mixed_ppi_estimates = w * pt_ppi_estimates + (1 - w) * compound_ppi_estimates
    return mixed_ppi_estimates if not get_w else (mixed_ppi_estimates, w)


# Aliases for the estimators
ALL_ESTIMATORS = {
    "mle": get_mle_estimators,
    # "pred_mean": get_pred_mean_estimators,
    "vanilla_ppi": get_vanilla_ppi_estimators,
    "power_tuned_ppi": get_power_tuned_ppi_estimators,
    "eb_sure": get_eb_sure_estimators,
    "shrink_var_ppi": get_shrink_var_ppi_estimators,
    "compound_ppi": get_compound_ppi_estimators,
    "mixed_compound_pt_ppi": get_mixed_compound_pt_ppi_estimators
}