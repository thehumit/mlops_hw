from ml_models import LogisticRegressionModel, DecisionTreeModel

import boto3
# from sklearn.datasets import make_classification
import numpy as np
from dvc_func import add_to_dvc



class ModelManager:
    def __init__(self):
        """
        Инициализация ModelManager.

        Создает пустой словарь для хранения моделей (`self.models`) и определяет
        доступные типы моделей в словаре `self.AVAILABLE_MODELS`.
        """
        self.models = {}
        self.s3_client = boto3.client('s3',
                         endpoint_url='http://127.0.0.1:9000',
                         aws_access_key_id='minioadmin',
                         aws_secret_access_key='minioadmin')
        self.AVAILABLE_MODELS = {
                                "logistic_regression": LogisticRegressionModel,
                                "decision_tree": DecisionTreeModel
                        }

    def get_available_models(self):
        """
        Возвращает список доступных типов моделей.

        :return: Список строк, представляющих ключи доступных моделей.
        """
        return list(self.AVAILABLE_MODELS.keys())

    def model_factory(self, model_type, **params):
        """
        Фабричный метод для создания экземпляра модели по указанному типу и параметрам.

        :param model_type: Тип модели для создания.
        :param params: Параметры, передаваемые в конструктор модели.
        :return: Экземпляр модели указанного типа.
        :raises ValueError: Если тип модели не поддерживается.
        """
        if model_type in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_type](**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def add_model(self, key, model):
        """
        Добавляет модель в словарь управляемых моделей.

        :param key: Ключ для идентификации модели.
        :param model: Экземпляр модели для добавления.
        """
        self.models[key] = model

    def retrain_model(self, key, model_type, params, X_train, y_train):
        """
        Повторно обучает модель с указанными параметрами и обновляет ее в словаре моделей.

        :param key: Ключ модели для повторного обучения.
        :param model_type: Тип модели для обучения.
        :param params: Параметры для обучения модели.
        :param X_train: Обучающие данные.
        :param y_train: Целевые значения.
        """
        new_model = self.model_factory(model_type, **params)
        new_model.train(X_train, y_train)
        self.models[key] = new_model
        add_to_dvc(X_train,  'train_data')
        add_to_dvc(y_train,  'train_target')

    def delete_model(self, key):
        """
        Удаляет модель из словаря управляемых моделей.

        :param key: Ключ модели для удаления.
        :raises ValueError: Если модель с указанным ключом не найдена.
        """
        if key in self.models:
            del self.models[key]
        else:
            raise ValueError(f"No model found with key {key}")

    def get_model(self, key):
        """
        Возвращает модель по ее ключу.

        :param key: Ключ модели для извлечения.
        :return: Модель, связанная с данным ключом, или None, если модель не найдена.
        """
        return self.models.get(key)

    def predict(self, key, X):
        """
        Выполняет прогнозирование с использованием указанной модели.

        :param key: Ключ модели для использования в прогнозировании.
        :param X: Данные для прогнозирования.
        :return: Результат прогнозирования модели.
        :raises ValueError: Если модель с указанным ключом не найдена.
        """
        model = self.get_model(key)
        if model is not None:
            return model.predict(X)
        else:
            raise ValueError(f"Model with key {key} not found")
    