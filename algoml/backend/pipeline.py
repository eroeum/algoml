import abc

class Pipeline(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return(hasattr(subclass, '__init__') and
               callable(subclass.__init__) and
               hasattr(subclass, 'fetch_data') and
               callable(subclass.fetch_data) and
               hasattr(subclass, 'ingest_data') and
               callable(subclass.ingest_data) and
               hasattr(subclass, 'split_data') and
               callable(subclass.split_data) and
               hasattr(subclass, 'preprocess') and
               callable(subclass.preprocess) and
               hasattr(subclass, 'train') and
               callable(subclass.train) and
               hasattr(subclass, 'evaluate') and
               callable(subclass.evaluate) and
               hasattr(subclass, 'run') and
               callable(subclass.run) and
               hasattr(subclass, 'save_model') and
               callable(subclass.save_model) and
               hasattr(subclass, 'load_model') and
               callable(subclass.load_model) or
               NotImplemented)

    @abc.abstractmethod
    def __init__(self) -> 'Model':
        """Initialize data values and model type"""
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_data(self, url, path, name) -> 'data/location':
        """Fetches data to local disk space"""
        raise NotImplementedError

    @abc.abstractmethod
    def ingest_data(self, path, name) -> 'Full data set':
        """Load data into local memory"""
        raise NotImplementedError

    @abc.abstractmethod
    def split_data(self, test_ratio) -> ('x_train', 'y_train', 'x_test', 'y_test'):
        """Split data into training and testing sets"""
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self) -> 'Transformed data':
        """Preprocess data for training and inference"""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self) -> None:
        """Train model on training data"""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self) -> ('scores'):
        """Evaluate model"""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> None:
        """Run pipeline"""
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, name) -> None:
        """Save model for future use"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, name) -> None:
        """Load model that has been saved"""
        raise NotImplementedError
