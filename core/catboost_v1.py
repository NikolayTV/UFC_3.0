from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import ast
import json
import datetime


def load_fighters():
    # fighters_df = pd.read_csv("data/Catboost_v1_0/fighters_df.csv", index_col=0)
    fighters_df = pd.read_csv("Notebooks/Catboost_v1_0/data_models/fighters_df.csv", index_col=0)

    fighters_df["dateOfBirth"] = pd.to_datetime(fighters_df["dateOfBirth"])
    fighters_cols = [
        # "id",
        "name",
        "weight",
        "height",
        "armSpan",
        "legSwing",
        "weightCategory.id",
        "weightCategory.name",
        "dateOfBirth",
        "country",
        "city",
        "timezone",
    ]
    fighters_df = fighters_df[fighters_cols]
    # fighters_df.set_index("id", inplace=True)
    f_name_dict = fighters_df['name'].to_dict()

    return fighters_df, f_name_dict


class Catboost_v1_0:
    '''
    Catboost model v0. Cumulative sum stats
    '''

    def __init__(self):
        self.clf1 = CatBoostClassifier()
        self.clf1.load_model(
            'Notebooks/Catboost_v1_0/data_models/catboost_v1_0_13.04.2021_1.cat')
        self.model_cols1 = self.clf1.feature_names_

        self.clf2 = CatBoostClassifier()
        self.clf2.load_model(
            'Notebooks/Catboost_v1_0/data_models/catboost_v1_0_13.04.2021_2.cat')
        self.model_cols2 = self.clf2.feature_names_

        # TODO replace fighters df to f_stats_events_cumulative
        self.f_stats_events_cumulative = pd.read_csv(
            'Notebooks/Catboost_v1_0/data_models/PROD_f_stats_events_cumulative_prod_08.04.2021.csv',
            index_col=0)
        self.f_stats_events_cumulative['eventDate.date'] = pd.to_datetime(
            self.f_stats_events_cumulative['eventDate.date'])

        self.generated_features = json.load(
            open('Notebooks/Catboost_v1_0/data_models/generated_features_08.04.2021.txt',
                 'r'))
        self.num_cols = [i[3:] for i in self.generated_features['fighter1_stats']][:-8]

        self.static_cols = ['country', 'city', 'armSpan', 'height', 'legSwing', 'timezone']
        self.f1_static_cols = ['f1_' + col for col in self.static_cols]
        self.f2_static_cols = ['f2_' + col for col in self.static_cols]

        self.fighters_df, self.f_name_dict = load_fighters()

    def predict_fight(self, f1_id, f2_id, event_date, f1_odd, f2_odd,
                      weightCategory_id, city, country, event_name, time_zone,
                      ):
        is_fight_night = 'fight night' in str(event_name).lower()

        ### Static stats
        f1_birthDate = self.fighters_df.loc[int(f1_id), ['dateOfBirth']]
        f1_static_stats = self.fighters_df.loc[int(f1_id), self.static_cols].values
        f1_age = ((pd.to_datetime(event_date) - pd.to_datetime(f1_birthDate)) / 365).dt.days.values[0]

        f2_birthDate = self.fighters_df.loc[int(f2_id), ['dateOfBirth']]
        f2_static_stats = self.fighters_df.loc[int(f2_id), self.static_cols].values
        f2_age = ((pd.to_datetime(event_date) - pd.to_datetime(f2_birthDate)) / 365).dt.days.values[0]

        ### Dynamic stats
        fighter1_stats = self.f_stats_events_cumulative[
            (self.f_stats_events_cumulative['fighterId'] == int(f1_id)) &
            (self.f_stats_events_cumulative['eventDate.date'] < pd.to_datetime(event_date))]
        fighter1_stats_NaNs = (fighter1_stats[self.num_cols].isna().sum(axis=1) / fighter1_stats.shape[1])
        fighter1_stats = pd.DataFrame(fighter1_stats[fighter1_stats_NaNs < 0.2].iloc[-1]).T.reset_index(drop=True)

        fighter2_stats = self.f_stats_events_cumulative[
            (self.f_stats_events_cumulative['fighterId'] == int(f2_id)) &
            (self.f_stats_events_cumulative['eventDate.date'] < pd.to_datetime(event_date))]
        fighter2_stats_NaNs = (fighter2_stats[self.num_cols].isna().sum(axis=1) / fighter2_stats.shape[1])
        fighter2_stats = pd.DataFrame(fighter2_stats[fighter2_stats_NaNs < 0.2].iloc[-1]).T.reset_index(drop=True)

        # Create prediction vector
        X_df = pd.DataFrame(index=[0])

        X_df = X_df.join(fighter1_stats[self.num_cols].add_prefix("f1_"))
        X_df = X_df.join(fighter2_stats[self.num_cols].add_prefix("f2_"))

        X_df.loc[0, ['f1_age', 'f2_age', 'f1_odds', 'f2_odds']] = f1_age, f2_age, f1_odd, f2_odd

        X_df[['weightCategory.id', 'city', 'country', 'is_fight_night', 'timezone']] = \
            weightCategory_id, city, country, is_fight_night, time_zone

        X_df[self.f1_static_cols] = f1_static_stats
        X_df[self.f2_static_cols] = f2_static_stats

        binary_fighter_cols = []
        for prefix in ["f1_", "f2_"]:
            for key in ["isHomeCity", "isHomeCountry", "isHomeTimezone"]:
                binary_fighter_cols.append(prefix + key)

        binary_stats = []
        binary_cols = ['city', 'country', 'timezone']
        for prefix in ["f1_", "f2_"]:
            for col in binary_cols:
                binary_stats.append(int(X_df.loc[0, prefix + col] == X_df.loc[0, col]))

        X_df[binary_fighter_cols] = binary_stats
        X_df[self.f1_static_cols + self.f2_static_cols] = X_df[self.f1_static_cols + self.f2_static_cols].fillna(
            'unknown')

        # Difference
        fighter1_stat_cols = self.generated_features['fighter1_stats']
        for col in fighter1_stat_cols:
            new_col_name = col[3:] + '_difference'
            X_df[new_col_name] = X_df['f1_' + col[3:]].astype(float) - X_df['f2_' + col[3:]].astype(float)

        # Make reversed prediction vector
        X_df_reversed = X_df.copy()

        reversed_cols = []
        for col in X_df.columns:
            if 'f2' in col:
                new_col_name = col.replace('f2', 'f1')
            elif 'f1' in col:
                new_col_name = col.replace('f1', 'f2')
            else:
                new_col_name = col
            reversed_cols.append(new_col_name)

        X_df_reversed.columns = reversed_cols
        fighter1_stat_cols = self.generated_features['fighter1_stats']
        for col in fighter1_stat_cols:
            new_col_name = col[3:] + '_difference'
            X_df_reversed[new_col_name] = X_df_reversed['f1_' + col[3:]].astype(float) - X_df_reversed[
                'f2_' + col[3:]].astype(float)

        y_proba1 = self.clf1.predict_proba(X_df[self.model_cols1])[:, 1]
        y_proba2 = self.clf2.predict_proba(X_df_reversed[self.model_cols2])[:, 0]

        y_proba = (y_proba1 + y_proba2) / 2
        return float(y_proba), [str(x) for x in X_df_reversed[self.model_cols2].values.tolist()], self.model_cols2


class Catboost_v1_1:
    """
    Catboost model v0. Cumulative sum stats
    """

    def __init__(self):
        self.clf1 = CatBoostClassifier()
        self.clf1.load_model(
            'Notebooks/Catboost_v1_1/data_models/catboost_v1_1_13.04.2021_1.cat')
        self.model_cols1 = self.clf1.feature_names_

        self.clf2 = CatBoostClassifier()
        self.clf2.load_model(
            'Notebooks/Catboost_v1_1/data_models/catboost_v1_1_13.04.2021_2.cat')
        self.model_cols2 = self.clf2.feature_names_

        # TODO replace fighters df to f_stats_events_cumulative
        self.f_stats_events_cumulative = pd.read_csv(
            'Notebooks/Catboost_v1_1/data_models/PROD_f_stats_events_cumulative_prod_08.04.2021.csv',
            index_col=0)
        self.f_stats_events_cumulative['eventDate.date'] = pd.to_datetime(
            self.f_stats_events_cumulative['eventDate.date'])

        self.generated_features = json.load(
            open('Notebooks/Catboost_v1_1/data_models/generated_features_08.04.2021.txt',
                 'r'))
        self.num_cols = [i[3:] for i in self.generated_features['fighter1_stats']][:-8]

        self.static_cols = ['country', 'city', 'armSpan', 'height', 'legSwing', 'timezone']
        self.f1_static_cols = ['f1_' + col for col in self.static_cols]
        self.f2_static_cols = ['f2_' + col for col in self.static_cols]

        self.fighters_df, self.f_name_dict = load_fighters()

    def predict_fight(self, f1_id, f2_id, event_date, f1_odd, f2_odd,
                      weightCategory_id, city, country, event_name, time_zone,
                      ):
        is_fight_night = 'fight night' in str(event_name).lower()

        ### Static stats
        f1_birthDate = self.fighters_df.loc[int(f1_id), ['dateOfBirth']]
        f1_static_stats = self.fighters_df.loc[int(f1_id), self.static_cols].values
        f1_age = ((pd.to_datetime(event_date) - pd.to_datetime(f1_birthDate)) / 365).dt.days.values[0]

        f2_birthDate = self.fighters_df.loc[int(f2_id), ['dateOfBirth']]
        f2_static_stats = self.fighters_df.loc[int(f2_id), self.static_cols].values
        f2_age = ((pd.to_datetime(event_date) - pd.to_datetime(f2_birthDate)) / 365).dt.days.values[0]

        ### Dynamic stats
        fighter1_stats = self.f_stats_events_cumulative[
            (self.f_stats_events_cumulative['fighterId'] == int(f1_id)) &
            (self.f_stats_events_cumulative['eventDate.date'] < pd.to_datetime(event_date))]
        fighter1_stats_NaNs = (fighter1_stats[self.num_cols].isna().sum(axis=1) / fighter1_stats.shape[1])
        fighter1_stats = pd.DataFrame(fighter1_stats[fighter1_stats_NaNs < 0.2].iloc[-1]).T.reset_index(drop=True)

        fighter2_stats = self.f_stats_events_cumulative[
            (self.f_stats_events_cumulative['fighterId'] == int(f2_id)) &
            (self.f_stats_events_cumulative['eventDate.date'] < pd.to_datetime(event_date))]
        fighter2_stats_NaNs = (fighter2_stats[self.num_cols].isna().sum(axis=1) / fighter2_stats.shape[1])
        fighter2_stats = pd.DataFrame(fighter2_stats[fighter2_stats_NaNs < 0.2].iloc[-1]).T.reset_index(drop=True)

        # Create prediction vector
        X_df = pd.DataFrame(index=[0])

        X_df = X_df.join(fighter1_stats[self.num_cols].add_prefix("f1_"))
        X_df = X_df.join(fighter2_stats[self.num_cols].add_prefix("f2_"))

        X_df.loc[0, ['f1_age', 'f2_age', 'f1_odds', 'f2_odds']] = f1_age, f2_age, f1_odd, f2_odd

        X_df[['weightCategory.id', 'city', 'country', 'is_fight_night', 'timezone']] = \
            weightCategory_id, city, country, is_fight_night, time_zone

        X_df[self.f1_static_cols] = f1_static_stats
        X_df[self.f2_static_cols] = f2_static_stats

        binary_fighter_cols = []
        for prefix in ["f1_", "f2_"]:
            for key in ["isHomeCity", "isHomeCountry", "isHomeTimezone"]:
                binary_fighter_cols.append(prefix + key)

        binary_stats = []
        binary_cols = ['city', 'country', 'timezone']
        for prefix in ["f1_", "f2_"]:
            for col in binary_cols:
                binary_stats.append(int(X_df.loc[0, prefix + col] == X_df.loc[0, col]))

        X_df[binary_fighter_cols] = binary_stats
        X_df[self.f1_static_cols + self.f2_static_cols] = X_df[self.f1_static_cols + self.f2_static_cols].fillna(
            'unknown')

        # Difference
        fighter1_stat_cols = self.generated_features['fighter1_stats']
        for col in fighter1_stat_cols:
            new_col_name = col[3:] + '_difference'
            X_df[new_col_name] = X_df['f1_' + col[3:]].astype(float) - X_df['f2_' + col[3:]].astype(float)

        # Make reversed prediction vector
        X_df_reversed = X_df.copy()

        reversed_cols = []
        for col in X_df.columns:
            if 'f2' in col:
                new_col_name = col.replace('f2', 'f1')
            elif 'f1' in col:
                new_col_name = col.replace('f1', 'f2')
            else:
                new_col_name = col
            reversed_cols.append(new_col_name)

        X_df_reversed.columns = reversed_cols
        fighter1_stat_cols = self.generated_features['fighter1_stats']
        for col in fighter1_stat_cols:
            new_col_name = col[3:] + '_difference'
            X_df_reversed[new_col_name] = X_df_reversed['f1_' + col[3:]].astype(float) - X_df_reversed[
                'f2_' + col[3:]].astype(float)

        y_proba1 = self.clf1.predict_proba(X_df[self.model_cols1])[:, 1]
        y_proba2 = self.clf2.predict_proba(X_df_reversed[self.model_cols2])[:, 0]

        y_proba = (y_proba1 + y_proba2) / 2
        return float(y_proba), [str(x) for x in X_df_reversed[self.model_cols2].values.tolist()], self.model_cols2


class Catboost_v1_2:
    """
    Catboost model v0. Cumulative sum stats
    """

    def __init__(self):
        self.clf1 = CatBoostClassifier()
        self.clf1.load_model(
            'Notebooks/Catboost_v1_2/data_models/catboost_v1_2_13.04.2021_1.cat')
        self.model_cols1 = self.clf1.feature_names_

        self.clf2 = CatBoostClassifier()
        self.clf2.load_model(
            'Notebooks/Catboost_v1_2/data_models/catboost_v1_2_13.04.2021_2.cat')
        self.model_cols2 = self.clf2.feature_names_

        # TODO replace fighters df to f_stats_events_cumulative
        self.f_stats_events_cumulative = pd.read_csv(
            'Notebooks/Catboost_v1_2/data_models/PROD_f_stats_events_cumulative_prod_08.04.2021.csv',
            index_col=0)
        self.f_stats_events_cumulative['eventDate.date'] = pd.to_datetime(
            self.f_stats_events_cumulative['eventDate.date'])

        self.generated_features = json.load(
            open('Notebooks/Catboost_v1_2/data_models/generated_features_08.04.2021.txt',
                 'r'))
        self.num_cols = [i[3:] for i in self.generated_features['fighter1_stats']][:-8]

        self.static_cols = ['country', 'city', 'armSpan', 'height', 'legSwing', 'timezone']
        self.f1_static_cols = ['f1_' + col for col in self.static_cols]
        self.f2_static_cols = ['f2_' + col for col in self.static_cols]

        self.fighters_df, self.f_name_dict = load_fighters()

    def predict_fight(self, f1_id, f2_id, event_date, f1_odd, f2_odd,
                      weightCategory_id, city, country, event_name, time_zone,
                      ):
        is_fight_night = 'fight night' in str(event_name).lower()

        ### Static stats
        f1_birthDate = self.fighters_df.loc[int(f1_id), ['dateOfBirth']]
        f1_static_stats = self.fighters_df.loc[int(f1_id), self.static_cols].values
        f1_age = ((pd.to_datetime(event_date) - pd.to_datetime(f1_birthDate)) / 365).dt.days.values[0]

        f2_birthDate = self.fighters_df.loc[int(f2_id), ['dateOfBirth']]
        f2_static_stats = self.fighters_df.loc[int(f2_id), self.static_cols].values
        f2_age = ((pd.to_datetime(event_date) - pd.to_datetime(f2_birthDate)) / 365).dt.days.values[0]

        ### Dynamic stats
        fighter1_stats = self.f_stats_events_cumulative[
            (self.f_stats_events_cumulative['fighterId'] == int(f1_id)) &
            (self.f_stats_events_cumulative['eventDate.date'] < pd.to_datetime(event_date))]
        fighter1_stats_NaNs = (fighter1_stats[self.num_cols].isna().sum(axis=1) / fighter1_stats.shape[1])
        fighter1_stats = pd.DataFrame(fighter1_stats[fighter1_stats_NaNs < 0.2].iloc[-1]).T.reset_index(drop=True)

        fighter2_stats = self.f_stats_events_cumulative[
            (self.f_stats_events_cumulative['fighterId'] == int(f2_id)) &
            (self.f_stats_events_cumulative['eventDate.date'] < pd.to_datetime(event_date))]
        fighter2_stats_NaNs = (fighter2_stats[self.num_cols].isna().sum(axis=1) / fighter2_stats.shape[1])
        fighter2_stats = pd.DataFrame(fighter2_stats[fighter2_stats_NaNs < 0.2].iloc[-1]).T.reset_index(drop=True)

        # Create prediction vector
        X_df = pd.DataFrame(index=[0])

        X_df = X_df.join(fighter1_stats[self.num_cols].add_prefix("f1_"))
        X_df = X_df.join(fighter2_stats[self.num_cols].add_prefix("f2_"))

        X_df.loc[0, ['f1_age', 'f2_age', 'f1_odds', 'f2_odds']] = f1_age, f2_age, f1_odd, f2_odd

        X_df[['weightCategory.id', 'city', 'country', 'is_fight_night', 'timezone']] = \
            weightCategory_id, city, country, is_fight_night, time_zone

        X_df[self.f1_static_cols] = f1_static_stats
        X_df[self.f2_static_cols] = f2_static_stats

        binary_fighter_cols = []
        for prefix in ["f1_", "f2_"]:
            for key in ["isHomeCity", "isHomeCountry", "isHomeTimezone"]:
                binary_fighter_cols.append(prefix + key)

        binary_stats = []
        binary_cols = ['city', 'country', 'timezone']
        for prefix in ["f1_", "f2_"]:
            for col in binary_cols:
                binary_stats.append(int(X_df.loc[0, prefix + col] == X_df.loc[0, col]))

        X_df[binary_fighter_cols] = binary_stats
        X_df[self.f1_static_cols + self.f2_static_cols] = X_df[self.f1_static_cols + self.f2_static_cols].fillna(
            'unknown')

        # Difference
        fighter1_stat_cols = self.generated_features['fighter1_stats']
        for col in fighter1_stat_cols:
            new_col_name = col[3:] + '_difference'
            X_df[new_col_name] = X_df['f1_' + col[3:]].astype(float) - X_df['f2_' + col[3:]].astype(float)

        # Make reversed prediction vector
        X_df_reversed = X_df.copy()

        reversed_cols = []
        for col in X_df.columns:
            if 'f2' in col:
                new_col_name = col.replace('f2', 'f1')
            elif 'f1' in col:
                new_col_name = col.replace('f1', 'f2')
            else:
                new_col_name = col
            reversed_cols.append(new_col_name)

        X_df_reversed.columns = reversed_cols
        fighter1_stat_cols = self.generated_features['fighter1_stats']
        for col in fighter1_stat_cols:
            new_col_name = col[3:] + '_difference'
            X_df_reversed[new_col_name] = X_df_reversed['f1_' + col[3:]].astype(float) - X_df_reversed[
                'f2_' + col[3:]].astype(float)

        y_proba1 = self.clf1.predict_proba(X_df[self.model_cols1])[:, 1]
        y_proba2 = self.clf2.predict_proba(X_df_reversed[self.model_cols2])[:, 0]

        y_proba = (y_proba1 + y_proba2) / 2
        return float(y_proba), [str(x) for x in X_df_reversed[self.model_cols2].values.tolist()], self.model_cols2
