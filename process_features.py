import pandas as pd


def process_features_single(df: pd.DataFrame):
	return df


def process_features_combined(df: pd.DataFrame):
	for i in range(1, 5):
		df = pd.get_dummies(df, columns=[f"linqmap_type_{i}",
		                                 f"linqmap_subtype_{i}",
		                                 f"linqmap_roadType_{i}"])

	return df
