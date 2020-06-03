from algoml.classification import BinaryClassificationPipeline
from algoml.classification import MulticlassClassificationPipeline
from algoml.classification import MultilabelClassificationPipeline
from algoml.regression import LinearRegressionPipeline
from algoml.svm import LinearSVMPipeline

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
    c.run()

def run_linear_svm_pipeline():
    s = LinearSVMPipeline()
    s.run()

if __name__ == '__main__':
    run_linear_svm_pipeline()
