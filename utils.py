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


"""
Some visualization functions
"""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_estimators_over_problems(estimators: list[np.ndarray], names: list[str] = None, links: list[tuple[int, int]] = None):
    """
    Use seaborn to plot line plots of the estimators over the problems (id from 1 to len(n_problems)). Choose different colors and markers
    for each estimator. Link the 
    """
    n_estimators, n_problems = len(estimators), len(estimators[0])

    if names is None:
        names = [f'Estimator {i}' for i in range(len(estimators))]
    
    problem_idx = np.arange(1, n_problems + 1)
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'x']
    colors = sns.color_palette('husl', n_estimators)

    fig = plt.figure(figsize=(10, 6))
    for i, estimator in enumerate(estimators):
        sns.lineplot(x=problem_idx, y=estimator, label=names[i], marker=markers[i], color=colors[i], linestyle='')

    for link in links:
        assert len(link) == 2, 'Link should be a tuple of two integers'
        assert 0 <= link[0] <= n_problems and 1 <= link[1] <= n_problems, 'Link should be a tuple of two integers'
        # plot vertical lines between estimators in `link` for each problem
        for problem in problem_idx:
            plt.vlines(problem, estimators[link[0] - 1][problem - 1], estimators[link[1] - 1][problem - 1], colors='gray', linestyles='dotted')
        
    # show legend
    plt.legend()
    plt.xlabel('Problem ID')
    plt.ylabel('Estimated Value')
    plt.show()
    # return the plot so that it can be saved
    return fig


if __name__ == '__main__':
    fig = plot_estimators_over_products([np.random.rand(10), np.random.rand(10)], ['Estimator 1', 'Estimator 2'], [(1, 2)])
    fig.savefig('test.png')
