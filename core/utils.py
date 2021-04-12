import pandas as pd


def load_fighters():
    # fighters_df = pd.read_csv("data/Catboost_v1_0/fighters_df.csv", index_col=0)
    fighters_df = pd.read_csv("/home/nikolay/workspace/UFC_betting/UFC_3.0/data/Catboost_v1_0/fighters_df.csv", index_col=0)


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


def difference(events_df_joined, cols):
    df = pd.DataFrame()
    # цикл заменяет столбцы характеристик каждого бойца столбцами разницы этих характеристик
    for col in cols:
        df[col + '_difference'] = events_df_joined['f1_' + col].astype(float) - events_df_joined['f2_' + col].astype(
            float)

    df['age_difference'] = events_df_joined['f1_age'] - events_df_joined[
        'f2_age']  # не стал удалять столбцы с возрастом

    return df

