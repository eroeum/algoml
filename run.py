from algoml.classification import BinaryClassificationPipeline
from algoml.classification import MulticlassClassificationPipeline
from algoml.classification import MultilabelClassificationPipeline
from algoml.decision_tree import DecisionTreeClassificationPipeline
from algoml.decision_tree import DecisionTreeRegressionPipeline
from algoml.ensembles import VotingClassifierPipeline
from algoml.regression import LinearRegressionPipeline
from algoml.regression import LogisticRegressionPipeline
from algoml.svm import LinearSVMPipeline
from algoml.svm import NonlinearSVMPipeline

def run_linear_regession_pipeline():
    r = LinearRegressionPipeline()
    r.run()

def run_logistic_regession_pipeline():
    r = LogisticRegressionPipeline()
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

def run_nonlinear_svm_pipeline():
    s = NonlinearSVMPipeline()
    s.run()

def run_decision_tree_classification_pipeline():
    d = DecisionTreeClassificationPipeline()
    d.run()

def run_decision_tree_regression_pipeline():
    d = DecisionTreeRegressionPipeline()
    d.run()

def run_voting_classifier_pipeline():
    e = VotingClassifierPipeline()
    e.run()

if __name__ == '__main__':
    run_voting_classifier_pipeline()
