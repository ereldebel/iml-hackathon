import pandas as pd
from sklearn.model_selection import train_test_split

from process_features import process_features_single, process_features_combined


def get_processed_test_set():
	test_path = "Mission 1 - Waze/waze_take_features.csv"
	df = pd.read_csv(test_path)
	columns = [column + "_" + str(index) for index in (1, 2, 3, 4) for column in df.columns if column != 'index']
	new_df = pd.DataFrame(columns=columns)
	rows_list = df.to_dict('records')
	for index, row in enumerate(rows_list):
		if index % 4 != 3:
			continue
		row_1, row_2, row_3, row_4 = rows_list[index - 3], rows_list[index - 2], rows_list[index - 1], rows_list[index]
		dict1 = {key + "_1": value for (key, value) in row_1.items() if
				 key != 'index'}
		dict2 = {key + "_2": value for (key, value) in row_2.items() if
				 key != 'index'}
		dict3 = {key + "_3": value for (key, value) in row_3.items() if
				 key != 'index'}
		dict4 = {key + "_4": value for (key, value) in row_4.items() if
				 key != 'index'}
		ndic = {**dict1, **dict2, **dict3, **dict4}
		new_df = new_df.append(ndic, ignore_index=True)
	df_combined = process_features_combined(new_df)
	df_combined.drop(
		columns=[f"pubDate_{i}" for i in range(1, 5)], inplace=True)
	df_combined.drop(
		columns=[f"linqmap_city_{i}" for i in range(1, 5)], inplace=True)
	return df_combined


def clean_data(df):
	features = ["OBJECTID", "linqmap_expectedBeginDate",
				"linqmap_reportMood",
				"linqmap_expectedEndDate", "linqmap_nearby", "nComments"]
	return df.drop(features, axis=1)


TLV_STRING = ('תל אביב - יפו', 'TLV')


def get_fifths(df: pd.DataFrame):
	columns = ['x_label', 'y_label', 'linqmap_type_label',
			   'linqmap_subtype_label'] + [column + "_" + str(index) for
										   index in (1, 2, 3, 4) for
										   column in df.columns if
										   column != 'index']
	new_df = pd.DataFrame(columns=columns)
	rows_list = df.to_dict('records')
	for index, row in enumerate(rows_list):
		if index % 5 != 4:
			continue
		row_1, row_2, row_3, row_4, row_5 = rows_list[index - 4], \
											rows_list[index - 3], rows_list[
												index - 2], rows_list[
												index - 1], rows_list[
												index]
		# sort 4 rows by publish date
		rows_by_pub = [row_1, row_2, row_3, row_4]
		rows_by_pub.sort(key=lambda dict: dict['pubDate'])
		row_1, row_2, row_3, row_4 = rows_by_pub

		dict1 = {key + "_1": value for (key, value) in row_1.items() if
		         key != 'index'}
		dict2 = {key + "_2": value for (key, value) in row_2.items() if
		         key != 'index'}
		dict3 = {key + "_3": value for (key, value) in row_3.items() if
		         key != 'index'}
		dict4 = {key + "_4": value for (key, value) in row_4.items() if
		         key != 'index'}

		dict_label = {key + "_label": value for (key, value) in row_5.items()
		              if
		              key in ['x', 'y', 'linqmap_type', 'linqmap_subtype']}
		ndic = {**dict1, **dict2, **dict3, **dict4, **dict_label}
		new_df = new_df.append(ndic, ignore_index=True)
	return new_df


def write_data(df: pd.DataFrame, city_string=TLV_STRING):
	"""gets RAW df and produces data ready for learning."""
	df['update_date'] = pd.to_datetime(df['update_date'], unit='ms')
	df = df.sort_values(by=['update_date'])
	df = clean_data(df)
	df.to_csv(f"datasets/original_data_cleaned.csv")
	df = process_features_single(df)
	df_city = df[df['linqmap_city'] == city_string[0]].reset_index()
	df_fifths = get_fifths(df_city)

	df_fifths_with_combined_features = process_features_combined(df_fifths)
	df_fifths_with_combined_features.drop(
		columns=[f"pubDate_{i}" for i in range(1, 5)], inplace=True)
	df_fifths_with_combined_features.drop(
		columns=[f"linqmap_city_{i}" for i in range(1, 5)], inplace=True)

	train_set, test_set = train_test_split(
		df_fifths_with_combined_features, train_size=2 / 3,
		shuffle=False)

	label_columns = [column for column in train_set.columns if
	                 column.endswith("label")]
	train_set_X = train_set.drop(columns=label_columns, inplace=False)
	train_set_y = train_set[label_columns]
	test_set_X = test_set.drop(columns=label_columns, inplace=False)
	test_set_y = test_set[label_columns]

	train_set_X.to_csv(f"datasets/train_set_X_{city_string[1]}.csv",
	                 index=False)
	test_set_X.to_csv(f"datasets/test_set_X_{city_string[1]}.csv",
	                index=False)

	train_set_y.to_csv(f"datasets/train_set_y_{city_string[1]}.csv",
	                 index=False)
	test_set_y.to_csv(f"datasets/test_set_y_{city_string[1]}.csv",
	                index=False)

	return train_set, test_set


if __name__ == '__main__':
	file_path = "Mission 1 - Waze/waze_data.csv"
	df = pd.read_csv(file_path, parse_dates=['pubDate', 'update_date'])
	train_set, test_set = write_data(df, city_string=TLV_STRING)

	pass
