from algoml.binary_classification.binary_classification import BinaryClassificationPipeline
from algoml.linear_regression.linear_regression import LinearRegressionPipeline

def run_binary_classification_pipeline():
    c = BinaryClassificationPipeline()
    data, target = c.load_data()
    x_train, x_test, y_train, y_test = c.split_data()
    c.train(x_train, y_train)
    print(c.cm_score(x_train, y_train))
    c.save_model("binary_classification_model.pkl")

def run_linear_regession_pipeline():
    r = LinearRegressionPipeline()
    df = r.load_data()
    training, testing = r.split_data(0.2)
    training_data, training_labels = r.split_estimator_label()
    r.preprocess()
    r.train(r.data, training_labels)
    r.score(r.data, training_labels)
    r.save_model("linear_regression_model.pkl")

if __name__ == '__main__':
    run_binary_classification_pipeline()
