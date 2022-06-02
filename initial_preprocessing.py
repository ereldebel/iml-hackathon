import pandas as pd
from sklearn.model_selection import train_test_split

from process_features import process_features_single, process_features_combined


def clean_data(df):
    for feature in ["OBJECTID", "linqmap_expectedBeginDate",
                    "linqmap_reportMood",
                    "linqmap_expectedEndDate", "linqmap_nearby", "nComments"]:
        df = df.drop(feature, 1)
    return df


TLV_STRING = ('תל אביב - יפו', 'TLV')


# def get_fifths(full_data: pd.DataFrame):
#     row_i = [0]
#
#     def create_fifth(row: pd.DataFrame):
#         row_i[0] = row_i[0] + 1
#         if (row_i[0] == 5):
#
#
#         row_i[0] = row_i[0]%5
#
#     # full_data.apply()


def get_fifths(df: pd.DataFrame):
    columns = ['x_label', 'y_label', 'linqmap_type_label', 'linqmap_subtype_label'] + [column + "_" + str(index) for
                                                                                       index in (1, 2, 3, 4) for
                                                                                       column in df.columns if
                                                                                       column != 'index']
    new_df = pd.DataFrame(columns=columns)
    rows_list = df.to_dict('records')
    for index, row in enumerate(rows_list):
        if index % 5 != 4:
            continue
        row_1, row_2, row_3, row_4, row_5 = rows_list[index - 4], \
                                            rows_list[index - 3], rows_list[index - 2], rows_list[index - 1], rows_list[
                                                index]
        dict1 = {key + "_1": value for (key, value) in row_1.items() if key != 'index'}
        dict2 = {key + "_2": value for (key, value) in row_2.items() if key != 'index'}
        dict3 = {key + "_3": value for (key, value) in row_3.items() if key != 'index'}
        dict4 = {key + "_4": value for (key, value) in row_4.items() if key != 'index'}
        dict_label = {key + "_label": value for (key, value) in row_5.items() if
                      key in ['x', 'y', 'linqmap_type', 'linqmap_subtype']}
        ndic = {**dict1, **dict2, **dict3, **dict4, **dict_label}
        new_df = new_df.append(ndic, ignore_index=True)
    return new_df


def write_data(df: pd.DataFrame, city_string=TLV_STRING):
    """gets RAW df and producues data ready for learning."""
    df['update_date'] = pd.to_datetime(df['update_date'], unit='ms')
    df = df.sort_values(by=['update_date'])
    df = clean_data(df)
    df.to_csv(f"datasets/original_data_cleaned.csv")
    df = process_features_single(df)
    df_city = df[df['linqmap_city'] == city_string[0]].reset_index()
    df_fifths = get_fifths(df_city)
    df_fifths_with_combined_features = process_features_combined(df_fifths)

    train_set, test_set = train_test_split(df_fifths, train_size=2 / 3,
                                           shuffle=False)

    train_set.to_csv(f"datasets/train_set_{city_string[1]}.csv")
    test_set.to_csv(f"datasets/test_set_{city_string[1]}.csv")
    return train_set, test_set


if __name__ == '__main__':
    file_path = "Mission 1 - Waze/waze_data.csv"
    df = pd.read_csv(file_path, parse_dates=['pubDate', 'update_date'])
    train_set, test_set = write_data(df, city_string=TLV_STRING)
    pass
