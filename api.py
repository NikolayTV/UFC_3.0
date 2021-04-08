from fastapi import Body, FastAPI, HTTPException
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from core.catboost_v0_0 import Catboost_v0_0
import ast
from datetime import datetime

app = FastAPI()

Catboost_model_v0 = Catboost_v0_0()

@app.get("/predict_fight")
def predict_fight(
        fighter1_name: str = 'Donald Cerrone',
        fighter2_name: str = 'Jim Miller',
        event_date: datetime = datetime.utcnow()
):
    y_proba_catboost_v0 = Catboost_model_v0.predict_fight(fighter1_name=fighter1_name, fighter2_name=fighter2_name, event_date=event_date)

    output = {}
    output['y_proba_catboost_v0'] = y_proba_catboost_v0

    return output



