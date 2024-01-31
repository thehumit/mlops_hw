from pydantic import BaseModel, Field
import numpy as np

class ModelParams(BaseModel):
    model_type: str
    params: dict


class TrainData(BaseModel):
    X_train: np.ndarray
    y_train: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class PredictData(BaseModel):
    X: np.ndarray

    class Config:
        arbitrary_types_allowed = True