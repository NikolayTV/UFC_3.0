from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import ast
import json

from core.utils import load_fighters, difference


class Catboost_v0_0:
    '''
    Catboost model v0. Cumulative sum stats
    '''

    def __init__(self):
        self.clf_v0 = CatBoostClassifier()
        self.clf_v0.load_model('models/Catboost_v1_0/catboost_v0_0_05.04.2021.cat')
        self.model_cols_v0 = self.clf_v0.feature_names_

        self.f_stats_events_cumulative = pd.read_csv('data/Catboost_v1_0/PROD_f_stats_events_cumulative_06.04.2021.csv',
                                                     index_col=0)
        self.all_new_cols = json.load(open('../../data/Catboost_v1_0/all_new_cols_06.04.2021.txt', 'r'))
        self.fighters_df, self.f_name_dict = load_fighters()

    def predict_fight(self, fighter1_name, fighter2_name, event_date, f1_odd, f2_odd,
                      weightCategory_name, city, country
                      ):

        all_fightCols_list = self.all_new_cols['new_cumsum_colnames'] + self.all_new_cols['new_accuracy_cols'] + \
                             self.all_new_cols['new_percent_cols'] + self.all_new_cols['new_PM_cols'] \
                             + self.all_new_cols['new_streak_cols'] + \
                             ['count_of_fights', 'age', 'odds']

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

        X_df.loc[0, ['f1_age', 'f2_age', 'f1_odds', 'f2_age_odds']] = f1_age, f2_age, f1_odd, f2_odd
        X_df[['weightCategory.name', 'city', 'country']] = weightCategory_name, city, country

        X_df_combined = difference(X_df, all_fightCols_list)
        X_df_combined_reversed = X_df_combined.copy()
        X_df_combined_reversed[all_fightCols_list] = X_df_combined_reversed[all_fightCols_list] * -1

        y_proba_straight = self.clf_v0.predict_proba(X_df_combined[self.model_cols_v0])[:, 1]
        y_proba_reversed = self.clf_v0.predict_proba(X_df_combined_reversed[self.model_cols_v0])[:, 0]
        y_proba = np.mean([y_proba_straight, y_proba_reversed])

        return y_proba
