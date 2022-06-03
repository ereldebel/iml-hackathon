import pandas as pd
from process_features import ProcessFeatures


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


def get_train_data_all_cities(df: pd.DataFrame, processor: ProcessFeatures,
                              export: bool = False):
	"""gets RAW df and produces data ready for learning."""
	df['update_date'] = pd.to_datetime(df['update_date'], unit='ms')
	df = df.sort_values(by=['update_date'])
	df = clean_data(df)
	if export:
		df.to_csv(f"datasets/original_train_data_cleaned.csv", index=False)
	df = processor.process_features_single(df)

	df_fifths = pd.DataFrame()
	cities = df['linqmap_city'].unique()
	for city in cities:
		df_city = df[df['linqmap_city'] == city].reset_index()
		if df_city.shape[0] < 100:
			continue
		df_fifth_single_city = get_fifths(df_city)
		df_fifth_single_city['is_tlv'] = 1 if (city == 'תל אביב - יפו') else 0
		df_fifths = df_fifths.append(df_fifth_single_city)

	processed_data = processor.process_features_combined(df_fifths)
	processed_data.drop(
		columns=[f"pubDate_{i}" for i in range(1, 5)], inplace=True)
	processed_data.drop(
		columns=[f"linqmap_city_{i}" for i in range(1, 5)], inplace=True)

	label_columns = [column for column in processed_data.columns if
	                 column.endswith("label")]
	train_X = processed_data.drop(columns=label_columns, inplace=False)
	train_y = processed_data[label_columns]
	if export:
		train_X.to_csv(f"datasets/train_X_all_cities.csv", index=False)
		train_y.to_csv(f"datasets/train_y_all_cities.csv", index=False)
	return train_X, train_y


def get_fourths(df: pd.DataFrame):
	columns = [column + "_" + str(index) for index in (1, 2, 3, 4) for column
	           in df.columns if column != 'index']
	new_df = pd.DataFrame(columns=columns)
	rows_list = df.to_dict('records')
	for index, row in enumerate(rows_list):
		if index % 4 != 3:
			continue
		row_1, row_2, row_3, row_4 = rows_list[index - 3], rows_list[
			index - 2], rows_list[index - 1], rows_list[index]
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

		ndic = {**dict1, **dict2, **dict3, **dict4}
		new_df = new_df.append(ndic, ignore_index=True)
	return new_df


def get_test_data_all_cities(df: pd.DataFrame, processor: ProcessFeatures,
                             export: bool = False):
	"""gets RAW df and produces data ready for learning."""
	df['update_date'] = pd.to_datetime(df['update_date'], unit='ms')
	df = df.sort_values(by=['update_date'])
	df = clean_data(df)
	if export:
		df.to_csv(f"datasets/original_test_data_cleaned.csv")
	df = processor.process_features_single(df, True)
	df_fourths = get_fourths(df)
	df_fourths['is_tlv'] = 1
	test_X = processor.process_features_combined(df_fourths)
	test_X.drop(
		columns=[f"pubDate_{i}" for i in range(1, 5)], inplace=True)
	test_X.drop(
		columns=[f"linqmap_city_{i}" for i in range(1, 5)], inplace=True)

	if export:
		test_X.to_csv(f"datasets/test_X.csv", index=False)
	return test_X


def match_columns(df_columns: pd.DataFrame, df_other: pd.DataFrame):
	"""Matches the columns of df_other to the columns of df_columns"""
	columns = df_columns.columns
	# Get missing columns in the training test
	missing_cols = set(columns) - set(df_other.columns)
	# Add a missing column in test set with default value equal to 0
	for col in missing_cols:
		df_other[col] = 0
	df_other = df_other.copy()
	# Ensure the order of column in the test set is in the same order as in train set
	df_other = df_other[columns]
	return df_other


def get_data(data_set_path, test_set_path, export: bool = False):
	processor = ProcessFeatures()
	full_data = pd.read_csv(data_set_path, parse_dates=['pubDate',
	                                                'update_date']).drop_duplicates()
	train_X, train_y = get_train_data_all_cities(full_data, processor, export)
	full_test_data = pd.read_csv(test_set_path,
	                             parse_dates=['pubDate', 'update_date'])
	test_X = get_test_data_all_cities(full_test_data, processor, export)
	train_X = match_columns(test_X, train_X)
	test_X = match_columns(train_X, test_X)
	return train_X, train_y, test_X


if __name__ == '__main__':
	get_data(True)
