import pandas as pd
import numpy as np


def process_features_single(df: pd.DataFrame):
    df['linqmap_subtype'] = np.where(pd.isna(df['linqmap_subtype']), df['linqmap_type'] + "_NO_SUBTYPE",
                                     df['linqmap_subtype'])
    df['light_rail'] = np.where(df['linqmap_reportDescription'] == 'אתר התארגנות - הקו הירוק של הרכבת הקלה', 1, 0)
    df['update_date'] = df['update_date'].astype("datetime64[ns]")
    df['pubDate'] = df['pubDate'].astype("datetime64[ns]")
    df['time_since_pub'] = df['update_date'] - df['pubDate']
    df['minutes_since_pub'] = np.where(df['time_since_pub'] < pd.Timedelta(1, "d"),
                                       df['time_since_pub'] / np.timedelta64(1, 'm'), 0)
    df['days_since_pub'] = np.where(df['time_since_pub'] >= pd.Timedelta(1, "d"),
                                    df['time_since_pub'] / np.timedelta64(1, 'D'), 0)

    nan_count = [0]
    def replace_street_name(street):
        if not pd.isna(street):
            return street
        nan_count[0] += 1
        return "nan" + str(nan_count[0])

    df['linqmap_street'] = df['linqmap_street'].apply(replace_street_name)

    df = df.drop(columns=['linqmap_reportDescription', 'time_since_pub', 'update_date'])
    return df


def process_features_combined(df: pd.DataFrame):
    for i in range(1, 5):
        df = pd.get_dummies(df, columns=[f"linqmap_type_{i}",
                                         f"linqmap_subtype_{i}",
                                         f"linqmap_roadType_{i}"])

    return df
