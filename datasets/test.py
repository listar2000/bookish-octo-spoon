from amazon_review import AmazonReviewDataset

if __name__ == "__main__":
    dataset = AmazonReviewDataset(verbose=True)
    print(dataset.pred_labelled[0].shape, dataset.y_labelled[0].shape)
    print(dataset.pred_unlabelled[0].shape, dataset.y_unlabelled[0].shape)
    print(dataset.true_theta.shape)