from algoml.binary_classification.binary_classification import BinaryClassificationPipeline
from algoml.linear_regression.linear_regression import LinearRegressionPipeline
from algoml.multiclass_classification.multiclass_classification import MulticlassClassificationPipeline
from algoml.multilabel_classification.multilabel_classification import MultilabelClassificationPipeline

def run_linear_regession_pipeline():
    r = LinearRegressionPipeline()
    r.run()

def run_binary_classification_pipeline():
    c = BinaryClassificationPipeline()
    c.run()

def run_multiclass_classification_pipeline():
    c = MulticlassClassificationPipeline()
    c.run()

def run_multilabel_classification_pipeline():
    c = MultilabelClassificationPipeline()
    c.run(save_model=True)

if __name__ == '__main__':
    run_multilabel_classification_pipeline()
