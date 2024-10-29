"""
Experiment with Amazon Review dataset
"""
from datasets.amazon_review import AmazonReviewDataset, AMAZON_FOOD_REVIEW_FILE_PATH
from estimators import ALL_ESTIMATORS
from utils import get_mse
from tqdm import tqdm # for progress bar
import pandas as pd

def benchmark_amazon_review(trials: int = 100, summary: bool = True, file_path: str = AMAZON_FOOD_REVIEW_FILE_PATH) -> pd.DataFrame:
    """ Benchmark the Amazon Review dataset.
    """
    mse_results = pd.DataFrame(columns=ALL_ESTIMATORS.keys())
    data = AmazonReviewDataset(file_path=file_path)

    for i in tqdm(range(trials)):
        # randomize the dataset (train-test split)
        data.reload_data(split_seed=i + 12345)
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
    from pathlib import Path
    bad_review_path = Path('/net/scratch2/listar2000/kaggle-amazon-review/src/datasets/amazon_review_bad.h5')
    mse_results = benchmark_amazon_review(trials=500, summary=True, file_path=bad_review_path)
    visualize_benchmark_amazon_review(mse_results)