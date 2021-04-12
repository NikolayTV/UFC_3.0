from fastapi import Body, FastAPI, HTTPException
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from core.catboost_v0_0 import Catboost_v0_0
from core.catboost_v1_0 import Catboost_v1_0

import ast
from datetime import date

app = FastAPI()

# Catboost_model_v0 = Catboost_v0_0()
Catboost_model_v1_0 = Catboost_v1_0()

@app.get("/predict_fight")
def predict_fight(
        f1_id: int = 70,
        f2_id: int = 426,
        f1_odd: float = 1.35,
        f2_odd: float = 3,
        weightCategory_id: str = '7',
        city: str = 'Las Vegas',
        country: str = 'USA',
        event_date: date = date.today(),
        event_name: str = 'UFC Fight Night',
        time_zone: str = 'America/Denver',
):
    response = Catboost_model_v1_0.predict_fight(f1_id=f1_id, f2_id=f2_id, event_date=event_date,
                                                              f1_odd=f1_odd, f2_odd=f2_odd,
                                                              weightCategory_id=weightCategory_id, city=city, country=country,
                                                              event_name=event_name, time_zone=time_zone)

    output = {'y_proba_catboost_v1_0': response[:2], 'X_df_values': list(response[2]), 'X_df_columns': list(response[3])}

    return output



