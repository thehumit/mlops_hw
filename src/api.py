from pydantic_models import ModelParams, PredictData, TrainData
from fastapi import FastAPI, HTTPException
import uvicorn
from model_manager import ModelManager

######
from sklearn.datasets import make_classification
from ml_models import LogisticRegressionModel
######

app = FastAPI()

X_train, y_train = make_classification(n_samples=1000, n_features=20, n_informative=2, 
                                       n_redundant=2, n_repeated=0, n_classes=2, 
                                       n_clusters_per_class=2, weights=None, flip_y=0.01, 
                                       class_sep=1.0, hypercube=True, shift=0.0, 
                                       scale=1.0, shuffle=True, random_state=None)


X_train_new, y_train_new = make_classification(n_samples=1500, n_features=30, n_informative=5, 
                                               n_redundant=3, n_repeated=0, n_classes=2, 
                                               n_clusters_per_class=2, weights=None, flip_y=0.05, 
                                               class_sep=1.5, hypercube=True, shift=0.0, 
                                               scale=1.0, shuffle=True, random_state=None)


model_manager = ModelManager()
model_key = "my_model"
model_manager.add_model(model_key, LogisticRegressionModel(max_iter=100))
model_manager.get_model(model_key).train(X_train, y_train)

# Переобучение модели
model_manager.retrain_model(model_key, "logistic_regression", {"max_iter": 200}, X_train_new, y_train_new)

# Удаление модели
# model_manager.delete_model(model_key)
# app = model_manager.get_app()
print(model_manager.models)

@app.post("/add_model/{key}")
def add_model(key: str, model_params: ModelParams):
    model_manager.add_model(key, model_manager.model_factory(model_params.model_type, **model_params.params))

@app.post("/retrain_model/{key}")
def retrain_model(key: str, model_params: ModelParams, train_data: TrainData):
    model_manager.retrain_model(key, model_params.model_type, model_params.params, 
                       train_data.X_train, train_data.y_train)

@app.delete("/delete_model/{key}")
def delete_model(key: str):
    try:
        model_manager.delete_model(key)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/get_model/{key}")
def get_model(key: str):
    model = model_manager.get_model(key)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with key {key} not found")
    return {"model": model.get_params()}

@app.get("/get_availiable_models")
def get_availiable_models():
    models = model_manager.get_available_models()
    return {"model": models}
    # if model is None:
        # raise HTTPException(status_code=404, detail=f"Model with key {key} not found")
    # return {"model": model.get_params()}

@app.post("/predict/{key}")
def predict(key: str, predict_data: PredictData):
    try:
        return model_manager.predict(key, predict_data.X)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)