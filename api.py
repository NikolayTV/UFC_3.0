from fastapi import Body, FastAPI, HTTPException
from core.catboost_v1 import Catboost_v1_0, Catboost_v1_1, Catboost_v1_2
import pandas as pd
import ast
from datetime import date

app = FastAPI()

Catboost_model_v1_0 = Catboost_v1_0()
Catboost_model_v1_1 = Catboost_v1_1()
Catboost_model_v1_2 = Catboost_v1_2()

# Check if fighter ID is in the base
fighters_df = pd.read_csv("./Notebooks/Catboost_v1_0/data_models/fighters_df.csv", index_col=0)
fighters_list = fighters_df.index

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

    output = {'success': True}
    if f1_id not in fighters_list:
        output['missing_fighterId_1'] = True
        output['success'] = False

    if f2_id not in fighters_list:
        output['missing_fighterId_2'] = True
        output['success'] = False
    if not output['success']:
        return output

    try:
        response_1_0 = Catboost_model_v1_0.predict_fight(f1_id=f1_id, f2_id=f2_id, event_date=event_date,
                                                         f1_odd=f1_odd, f2_odd=f2_odd,
                                                         weightCategory_id=weightCategory_id, city=city, country=country,
                                                         event_name=event_name, time_zone=time_zone)

        response_1_1 = Catboost_model_v1_1.predict_fight(f1_id=f1_id, f2_id=f2_id, event_date=event_date,
                                                         f1_odd=f1_odd, f2_odd=f2_odd,
                                                         weightCategory_id=weightCategory_id, city=city, country=country,
                                                         event_name=event_name, time_zone=time_zone)

        response_1_2 = Catboost_model_v1_2.predict_fight(f1_id=f1_id, f2_id=f2_id, event_date=event_date,
                                                         f1_odd=f1_odd, f2_odd=f2_odd,
                                                         weightCategory_id=weightCategory_id, city=city, country=country,
                                                         event_name=event_name, time_zone=time_zone)
    except Exception as exp:
        output['success'] = False
        output['exp'] = exp
        return output


    output['y_proba_catboost_v1_0'] = response_1_0[0]
    output['values']  = str(response_1_0[1])
    output['columns'] = str(response_1_0[2])
    output['y_proba_catboost_v1_1'] = response_1_1
    output['y_proba_catboost_v1_2'] = response_1_2
    # output = {'y_proba_catboost_v1_0': response_1_0[:1][0], 'X_df_values': list(response[1]),
    #           'X_df_columns': list(response[2])}

    return output
