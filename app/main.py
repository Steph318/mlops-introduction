from fastapi import FastAPI, HTTPException, Path, BackgroundTasks
from contextlib import asynccontextmanager
from typing import Annotated
import pandas as pd
import pickle
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field 
from sklearn.datasets import load_iris

from evidently.report import Report

from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import DataDriftPreset
from fastapi.responses import HTMLResponse



import asyncio



load_dotenv()


class IrisData(BaseModel):
    sepal_length: float = Field(default=1.1, gt=0, lt=10, description="Sepal length is in range (0,10)")
    sepal_width: float = Field(default=3.1, gt=0, lt=10, description="Sepal length is in range (0,10)")
    petal_length: float = Field(default=2.1, gt=0, lt=10, description="Sepal length is in range (0,10)")
    petal_width: float = Field(default=4.1, gt=0, lt=10, description="Sepal length is in range (0,10)")



ml_models = {} # Global dictionary to hold the models.
def load_model(path: str):
    model = None
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models when the app starts
    ml_models["logistic_model"] = load_model(os.getenv("LOGISTIC_MODEL"))
    ml_models["rf_model"] = load_model(os.getenv("RF_MODEL"))

    yield
    # This code will be executed after the application finishes handling requests, right before the shutdown
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Models loaded and FastAPI is ready!"}


@app.get("/models")
async def list_models():
    # Return the list of available models' names
    return {"available_models": list(ml_models.keys())}


@app.post("/predict/{model_name}")
async def predict(
    model_name: Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")],
    iris_data: IrisData,
    background_tasks: BackgroundTasks,
):
    input_data = [
        [
            iris_data.sepal_length,
            iris_data.sepal_width,
            iris_data.petal_length,
            iris_data.petal_width,
        ]
    ]

    if model_name not in ml_models.keys():
        raise HTTPException(status_code=404, detail="Model not found.")

    model = ml_models[model_name]
    prediction = model.predict(input_data)

    background_tasks.add_task(log_data, input_data[0], int(prediction[0]))

    return {"model": model_name, "prediction": int(prediction[0])}




def log_data(iris_data: list, prediction: int):
    global DATA_LOG
    iris_data.append(prediction)
    DATA_LOG.append(iris_data)



DATA_WINDOW_SIZE = 45

def load_train_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df


# loads our latest predictions
def load_last_predictions():
    prediction_data = pd.DataFrame(
        DATA_LOG[-DATA_WINDOW_SIZE:],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
            "species",
        ],
    )
    return prediction_data


def generate_dashboard() -> str:
    data_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ],
    )

    reference_data = load_train_data()
    current_data = load_last_predictions()

    data_report.run(reference_data=reference_data, current_data=current_data)

    return data_report.get_html()


@app.get("/monitoring", tags=["Other"])
def monitoring():
    if len(DATA_LOG) == 0:
        return {"msg": "No data."}
    dashboard = generate_dashboard()
    return HTMLResponse(dashboard)


# Cas d'utilisation plus courant et complet

# from fastapi import FastAPI
# from databases import Database
# import aioredis

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Initialisation
#     # 1. Connexion à la base de données
#     app.state.database = Database("postgresql://user:pass@localhost/db")
#     await app.state.database.connect()
    
#     # 2. Connexion à Redis
#     app.state.redis = await aioredis.create_redis_pool('redis://localhost')
    
#     # 3. Chargement des modèles ML
#     app.state.ml_models = {
#         'model1': load_model("path/to/model1"),
#         'model2': load_model("path/to/model2")
#     }
    
#     # 4. Initialisation du cache
#     app.state.cache = {}

#     yield  # L'application s'exécute

#     # Nettoyage
#     # 1. Fermeture de la base de données
#     await app.state.database.disconnect()
    
#     # 2. Fermeture de Redis
#     app.state.redis.close()
#     await app.state.redis.wait_closed()
    
#     # 3. Nettoyage des modèles ML
#     app.state.ml_models.clear()
    
#     # 4. Vidage du cache
#     app.state.cache.clear()

# app = FastAPI(lifespan=lifespan)