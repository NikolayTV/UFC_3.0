import pandas as pd
import json
from pandas import json_normalize
from tqdm.notebook import tqdm

import numpy as np
import ast
from typing import Dict, List

import sys
import os

sys.path.append(os.path.join(sys.path[0], '../../'))

from core.fighters_utils import replace_null_height_to_arm_span, replace_null_arm_span_to_height
from core.events_utils import parse_odds, get_fighter_stats_cols, sum_round_stats, \
    parse_fight_data_attack, parse_fight_data_defence, balance_target, add_age, get_territorial_cols, \
    fill_territorial_cols


def prepare_fighters(csv_path):
    fighters_df = pd.read_csv(csv_path, index_col=0)
    fighters_df["dateOfBirth"] = pd.to_datetime(fighters_df["dateOfBirth"])
    fighters_cols = [
        "id",
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
    fighters_df.set_index("id", inplace=True)
    f_name_dict = fighters_df['name'].to_dict()

    ### Исправляем поле `country` для бойцов из США
    # У некоторых бойцов из США в поле `country` указан штат, а не страна. \
    # Также заменяем написание `United States` на `USA`, чтобы название соответствовало данным из таблицы с боями.

    usa_state_names = [
        "Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut",
        "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho",
        "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine",
        "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota",
        "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma",
        "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
        "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia",
        "Wyoming",
    ]

    fighters_df.loc[fighters_df["country"] == "United States", "country"] = "USA"
    fighters_df.loc[fighters_df["country"].isin(usa_state_names), "country"] = "USA"

    ### Выбросы размаха ног меняем на NaN, для дальнейшей обработки
    # fighters_df.replace(fighters_df.legSwing.max(), np.nan, inplace=True)
    # fighters_df.replace(fighters_df.legSwing.min(), np.nan, inplace=True)

    ### Замена пустых значений роста на размах рук
    fighters_df['height'] = fighters_df.apply(
        lambda row: replace_null_height_to_arm_span(row),
        axis=1)

    ### Замена пустых значений размаха рук на рост
    fighters_df['armSpan'] = fighters_df.apply(
        lambda row: replace_null_arm_span_to_height(row),
        axis=1)

    ### Замена остальных пустых значений
    numeric_cols = fighters_df.select_dtypes(include=['int', 'float']).columns
    obj_cols = fighters_df.select_dtypes(include=['object']).columns

    fighters_df[obj_cols] = fighters_df[obj_cols].fillna('unknown')
    # fighters_df[numeric_cols] = fighters_df[numeric_cols].fillna(9999)
    fighters_df['dateOfBirth'] = fighters_df['dateOfBirth'].fillna(fighters_df['dateOfBirth'].mean())

    ### Убираем пустые значения размаха ног, средним по колонке
    # fighters_df['legSwing'].fillna(np.round(fighters_df['legSwing'].mean(), 1), inplace=True)
    fighters_df["dateOfBirth"] = pd.to_datetime(fighters_df["dateOfBirth"])

    # fighters_df.to_csv('./data_models/fighters_df.csv')
    return fighters_df


def prepare_events(csv_path):
    events_df = pd.read_csv(csv_path, index_col=0)
    # Shuffle fighter1 and fighter2 positions
    events_df = balance_target(events_df)
    # Define winner as True when fighterId1 wins
    events_df['winner'] = (events_df['winnerId'] == events_df['fighterId_1'])

    # Cleaning
    # Убираем строки с незавершенными боями и боями, где отсутствует `winnerId`
    events_df = events_df[events_df["completed"] == True]
    events_df = events_df[~events_df["winnerId"].isna()]

    # Убираем строки, где `winnerId` не совпадает с айди ни одного из бойцов
    events_df = events_df[
        ~((events_df["winnerId"] != events_df["fighterId_1"]) & (events_df["winnerId"] != events_df["fighterId_2"]))]

    # Удаляем лишние колонки
    events_df.drop(columns=["completed", "eventDate.timezone_type", "link"],
                   inplace=True)

    # Извлекаем данные из колонок `avgOdds` и `fighters`
    events_df[["f1_odds", "f2_odds"]] = events_df[["avgOdds", "fighterId_1", "fighterId_2"]] \
        .apply(lambda row: parse_odds(row), axis=1)

    # events_df = events_df.drop(columns="avgOdds")

    # Парсим колонку `fighters`
    fighter_attack_stats_cols, fighter_def_stats_cols = get_fighter_stats_cols()

    events_df[fighter_attack_stats_cols] = events_df[
        ["fighters", "fighterId_1", "fighterId_2"]
    ].apply(lambda row: parse_fight_data_attack(row), axis=1)

    events_df[fighter_def_stats_cols] = events_df[
        ["fighters", "fighterId_1", "fighterId_2"]
    ].apply(lambda row: parse_fight_data_defence(row), axis=1)

    events_df.drop(columns="fighters", inplace=True)

    events_df['eventDate.date'] = pd.to_datetime(events_df['eventDate.date'])

    return events_df


def join_fighters_and_events(fighters_df, events_df):
    fighter_data_cols = fighters_df.drop(columns=["weightCategory.id", "weightCategory.name"]).columns

    events_df = events_df.join(fighters_df[fighter_data_cols].add_prefix("f1_"),
                               on="fighterId_1")

    events_df = events_df.join(fighters_df[fighter_data_cols].add_prefix("f2_"),
                               on="fighterId_2")

    events_df[["f1_age", "f2_age"]] = events_df[["eventDate.date", "f1_dateOfBirth", "f2_dateOfBirth"]] \
        .apply(lambda row: add_age(row), axis=1)

    # Добавляем признаки `isHomeCity`, `isHomeCountry`, `isHomeTimezone`
    # Возможные значения переменных: 0 и 1 \
    # `isHomeCity` - боец дерется в родном городе \
    # `isHomeCountry` - боец дерется в родной стране \
    # `isHomeTimezone` - боец дерется в своем часовом поясе

    territorial_cols = get_territorial_cols()
    events_df[territorial_cols] = events_df.apply(
        lambda row: fill_territorial_cols(row), axis=1
    )
    events_df = events_df.reset_index()

    return events_df
