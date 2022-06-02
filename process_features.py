import pandas as pd


def process_features_single(df: pd.DataFrame):
	return df


def get_2_most_prominent_streets(df: pd.DataFrame):
	streets = dict()
	for i in range(1, 5):
		if df[f"linqmap_street_{i}"] in streets:
			streets[df[f"linqmap_street_{i}"]] += 1
		else:
			streets[df[f"linqmap_street_{i}"]] = 1
	most_prominent_streets = []
	for key, value in streets.items():
		if value >= 3:
			most_prominent_streets.insert(0, key)
		elif value == 2:
			most_prominent_streets.append(key)
	result = pd.DataFrame(
		columns=["most_prominent_street", "second_most_prominent_street"])
	result["most_prominent_street"] = most_prominent_streets[0] if len(
		most_prominent_streets) > 0 else 0
	result["second_most_prominent_street"] = most_prominent_streets[1] if len(
		most_prominent_streets) > 1 else 0

	for row_i in range(1, 5):
		result[f"{i}_in_most_prominent_street"] = 1 if len(
			most_prominent_streets) > 0 and df[f"linqmap_street_{i}"] == \
		                                               most_prominent_streets[
			                                               0] else 0
		result[f"{i}_in_second_most_prominent_street"] = 1 if len(
			most_prominent_streets) > 1 and df[f"linqmap_street_{i}"] == \
		                                                      most_prominent_streets[
			                                                      1] else 0
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
	df = pd.concat([df, streets.apply(get_2_most_prominent_streets, axis=1)],
	               axis=1)
	df.drop([f"linqmap_street_{i}" for i in row_range], axis=1, inplace=True)

	locations = pd.concat([df[[f"x_{i}" for i in row_range]],
	                      df[[f"y_{i}" for i in row_range]]], axis=1)

	return df
