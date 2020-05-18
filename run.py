from algoml.linear_regression.linear_regression import LinearRegressionPipeline

def run_linear_regession_pipeline():
    r = LinearRegressionPipeline()
    df = r.load_data()
    training, testing = r.split_data(0.2)
    training_data, training_labels = r.split_estimator_label()
    r.preprocess()
    r.train(r.data, training_labels)
    r.score(r.data, training_labels)

if __name__ == '__main__':
    run_linear_regession_pipeline()
