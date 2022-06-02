import pandas as pd
import numpy as np

Z_SCORE_THRESHOLD = 2.5

global_hours_dict = {}


def get_hours_dict(df: pd.DataFrame):
    hours_df = pd.DataFrame(pd.to_datetime(df['pubDate']).dt.hour)
    hours_df = hours_df.value_counts().reset_index()
    hours_df = hours_df.sort_values(by=[0], ascending=False)
    hours_dict = dict(zip(hours_df['pubDate'], hours_df[0]))
    return hours_dict


def add_timeslots(df: pd.DataFrame):
    i = 0
    while i < 23:
        df[f"{i}-{i + 2}"] = ((df['hour'] == i) | (df['hour'] == i + 1)).astype(int)
        i += 2
    return df


def process_features_single(df: pd.DataFrame, isTest: bool = False):
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

    if not isTest:
        global_hours_dict = get_hours_dict(df)
    df['hour'] = pd.to_datetime(df['pubDate']).dt.hour
    df['traffic'] = df['hour'].apply(lambda x: global_hours_dict[x])
    df = add_timeslots(df)

    df['sin_magvar'] = np.sin((df['linqmap_magvar'] * np.pi) / 180)
    df['cos_magvar'] = np.cos((df['linqmap_magvar'] * np.pi) / 180)

    df = df.drop(columns=['linqmap_reportDescription', 'time_since_pub', 'update_date', 'linqmap_magvar'])
    return df


def get_2_most_prominent_streets(df: pd.Series):
    streets = dict()
    for i in range(1, 5):
        if df[f"linqmap_street_{i}"] in streets:
            streets[df[f"linqmap_street_{i}"]] += 1
        else:
            streets[df[f"linqmap_street_{i}"]] = 1
    most_prominent_streets = []
    for key, value in streets.items():
        if value >= 3:
            most_prominent_streets.insert(0, value)
        elif value == 2:
            most_prominent_streets.append(value)
    result = pd.Series()
    result["most_prominent_street"] = most_prominent_streets[0] if len(
        most_prominent_streets) > 0 else 0
    result["second_most_prominent_street"] = most_prominent_streets[1] if len(
        most_prominent_streets) > 1 else 0

    for row_i in range(1, 5):
        result[f"{row_i}_in_most_prominent_street"] = 1 \
            if len(most_prominent_streets) > 0 and \
               df[f"linqmap_street_{row_i}"] == most_prominent_streets[
                   0] else 0
        result[f"{row_i}_in_second_most_prominent_street"] = 1 \
            if len(most_prominent_streets) > 1 and \
               df[f"linqmap_street_{row_i}"] == most_prominent_streets[
                   1] else 0
    return result


def combine_time(df):
    result = pd.DataFrame(columns=[f"duration_{i}" for i in range(2, 5)])
    for i in range(2, 5):
        result[f"duration_{i}"] = (df[f"pubDate_{i}"] - df[
            f"pubDate_{i - 1}"]).total_seconds()


def get_location_mean_features(df: pd.Series):
    coordinates = np.ndarray([4, 2])
    for i in range(1, 5):
        coordinates[i - 1] = df[f"x_{i}"], df[f"y_{i}"]
    mean_location = coordinates.mean(axis=0)
    dist_from_mean = np.linalg.norm(coordinates - mean_location, axis=1)
    std = np.std(coordinates, axis=0).mean()
    z_score = dist_from_mean / (std if std > 0 else 1)
    result = pd.Series()
    divisor = 4
    for i in range(1, 5):
        result[f"z_score_{i}"] = z_score[i - 1]
        if z_score[i - 1] > Z_SCORE_THRESHOLD:
            dist_from_mean[i - 1] = 0
            result[f"{i}_used_in_mean"] = 0
            divisor -= 1
        else:
            result[f"{i}_used_in_mean"] = 1
    coordinate_sum = np.sum(coordinates, axis=0) / (
        divisor if divisor > 0 else 1)
    result["mean_x"] = coordinate_sum[0]
    result["mean_y"] = coordinate_sum[1]
    return result


def process_features_combined(df: pd.DataFrame):
    row_range = range(1, 5)
    # make type and subtype one-hot
    for i in row_range:
        df = pd.get_dummies(df, columns=[f"linqmap_type_{i}",
                                         f"linqmap_subtype_{i}"])

    # replace street names with 2 columns of most prominent streets (how many
    # occurrences are in these streets) and for each occurrence, boolean of
    # which street it is on
    streets = df[[f"linqmap_street_{i}" for i in row_range]]
    new_streets_features = streets.apply(get_2_most_prominent_streets,
                                         axis=1).reindex(df.index)
    df = pd.concat([df, new_streets_features], axis=1)
    df.drop([f"linqmap_street_{i}" for i in row_range], axis=1, inplace=True)

    # add z score of x and y and mean coordinates of the closest points
    locations = pd.concat([df[[f"x_{i}" for i in row_range]],
                           df[[f"y_{i}" for i in row_range]]], axis=1)
    new_location_features = locations.apply(get_location_mean_features,
                                            axis=1).reindex(df.index)
    df = pd.concat([df, new_location_features], axis=1)

    return df
