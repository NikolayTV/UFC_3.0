from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import ast

from core.utils import load_fighters, difference


class Catboost_v0_0:
    '''
    Catboost model v0. Cumulative sum stats
    '''

    def __init__(self):
        self.clf_v0 = CatBoostClassifier()
        self.clf_v0.load_model('models/Catboost_v0/catboost_v0_0_05.04.2021.cat')
        self.model_cols_v0 = self.clf_v0.feature_names_

        self.f_stats_events_cumulative = pd.read_csv('data/f_stats_events_cumulative_05.04.2021.csv', index_col=0)
        self.ready_cols = ast.literal_eval(open('data/ready_cols_05.04.2021.txt', 'r').read())
        self.fighters_df, self.f_name_dict = load_fighters()


    def predict_fight(self, fighter1_name, fighter2_name, event_date):

        f1_birthDate = self.fighters_df[self.fighters_df['name'] == fighter1_name]['dateOfBirth']
        f1_age = ((pd.to_datetime(event_date, utc=True) - pd.to_datetime(f1_birthDate, utc=True)) / 365).dt.days.values[
            0]

        f2_birthDate = self.fighters_df[self.fighters_df['name'] == fighter2_name]['dateOfBirth']
        f2_age = ((pd.to_datetime(event_date, utc=True) - pd.to_datetime(f2_birthDate, utc=True)) / 365).dt.days.values[
            0]

        fighter1_stats = self.f_stats_events_cumulative[self.f_stats_events_cumulative['fighterName'] == fighter1_name]
        fighter1_stats = pd.DataFrame(
            fighter1_stats[fighter1_stats[self.ready_cols].isna().sum(axis=1) < 10].iloc[-1]).T.reset_index(drop=True)

        fighter2_stats = self.f_stats_events_cumulative[self.f_stats_events_cumulative['fighterName'] == fighter2_name]
        fighter2_stats = pd.DataFrame(
            fighter2_stats[fighter2_stats[self.ready_cols].isna().sum(axis=1) < 10].iloc[-1]).T.reset_index(drop=True)

        # Create df for prediction
        X_df = pd.DataFrame(index=[0])

        X_df = X_df.join(
            fighter1_stats[self.ready_cols].add_prefix("f1_"))  # , on="id"

        X_df = X_df.join(
            fighter2_stats[self.ready_cols].add_prefix("f2_"))

        X_df.loc[0, ['f1_age', 'f2_age']] = f1_age, f2_age

        X_df_combined = difference(X_df, self.ready_cols)
        X_df_combined_reversed = X_df_combined * -1

        y_proba_straight = self.clf_v0.predict_proba(X_df_combined[self.model_cols_v0])[:, 1]
        y_proba_reversed = self.clf_v0.predict_proba(X_df_combined_reversed[self.model_cols_v0])[:, 0]
        y_proba = np.mean([y_proba_straight, y_proba_reversed])

        return y_proba

