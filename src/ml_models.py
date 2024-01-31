from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from dvc_func import add_to_dvc


class Model(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class LogisticRegressionModel(Model):
    def __init__(self, **params):
        self.model = LogisticRegression(**params)

    def train(self, X, y):
        self.model.fit(X, y)
        add_to_dvc(X, 'train_data')
        add_to_dvc(y, 'train_target')

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.get_params()

class DecisionTreeModel(Model):
    def __init__(self, **params):
        self.model = DecisionTreeClassifier(**params)

    def train(self, X, y):
        self.model.fit(X, y)
        add_to_dvc(X, 'train_data')
        add_to_dvc(y, 'train_target')

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.get_params()