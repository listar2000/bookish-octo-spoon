"""
Experiment with Amazon Review dataset
"""
from datasets.amazon_review import AmazonReviewDataset
from estimators import ALL_ESTIMATORS
from utils import get_mse
from tqdm import tqdm # for progress bar
import pandas as pd

def benchmark_amazon_review(trials: int = 100, summary: bool = True) -> pd.DataFrame:
    """ Benchmark the Amazon Review dataset.
    """
    mse_results = pd.DataFrame(columns=ALL_ESTIMATORS.keys())
    data = AmazonReviewDataset(split_seed=42)

    for i in tqdm(range(trials)):
        # randomize the dataset (train-test split)
        data.reload_data(split_seed=i + 42)
        true_theta = data.true_theta

        for estimator_name, estimator_func in ALL_ESTIMATORS.items():
            theta_hat = estimator_func(data)
            mse = get_mse(theta_hat, true_theta)
            mse_results.loc[i, estimator_name] = mse
    
    if summary:
        # print the mean + sd of each estimator (column)
        print("Mean of MSE:\n", mse_results.mean())
        print("SD of MSE:\n", mse_results.std())
        
    return mse_results


def visualize_benchmark_amazon_review(mse_results: pd.DataFrame):
    """ Visualize the benchmark results.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=mse_results, orient="v")
    plt.xlabel("MSE")
    plt.title("Benchmark results on Amazon Review dataset")
    plt.savefig("benchmark_amazon_review_results.png")


if __name__ == "__main__":
    mse_results = benchmark_amazon_review(trials=100, summary=True)
    visualize_benchmark_amazon_review(mse_results)