import pandas as pd
import numpy as np


def is_holiday(date: pd.datetime):
	holiday_date = '2022-06-05'
	return date == pd.to_datetime(holiday_date, format='%Y-%m-%d')


def predict_csv_table(df: pd.DataFrame, date_str: str):
	"""predict csv table in required format, on given date."""
	FACTOR_MORNING, FACTOR_AFTERNOON, FACTOR_EVENING = 0.4, 0.65, 0.6
	aggregated_df = pd.DataFrame()
	date = pd.to_datetime(date_str, format='%Y-%m-%d')
	df = df.drop(columns=['hour'])
	for type in ['ACCIDENT', 'JAM', 'ROAD_CLOSED', 'WEATHERHAZARD']:
		df_typed = df[df['linqmap_type'] == type]
		prediction = df_typed.groupby('day').agg('sum').agg(lambda col: sum(col) / sum(np.where(col == 0, 0, 1)))
		prediction = prediction * [FACTOR_MORNING, FACTOR_AFTERNOON,
								   FACTOR_EVENING] if date.dayofweek == 5 or is_holiday(date) else prediction
		aggregated_df = aggregated_df.append(prediction, ignore_index=True)
	return aggregated_df.transpose()


def add_timeslots(df: pd.DataFrame):
	"""add columns to df indicating what timeslot the event occurred in"""
	for timeslot in [(i, i + 2) for i in [8, 12, 18]]:
		df[f"{timeslot[0]}-{timeslot[1]}"] = (
				(df['hour'] == timeslot[0]) | (df['hour'] == timeslot[0] + 1)).astype(int)
	return df


def load_data():
	"""load original data and process its event distributions."""
	data_path = "datasets/original_data.csv"
	df = pd.read_csv(data_path)
	df = df.drop_duplicates()
	df['update_date'] = pd.to_datetime(df['update_date'], unit='ms')
	df['pubDate'] = df['pubDate'].astype('datetime64[ns]')
	df = df[["update_date", "linqmap_type"]]

	# update_date or pubDate??
	# reference days in data: ?? for june 5th, AVERAGE for june 7th, 15th may for june 9th.
	df['hour'] = pd.to_datetime(df['update_date']).dt.hour
	df['day'] = pd.to_datetime(df['update_date']).dt.day
	df = add_timeslots(df)
	return df


def main(dates):
	df = load_data()
	for date in dates:
		prediction = predict_csv_table(df, date)
		prediction.to_csv("prediction" + date + ".csv", index=False, header=False)


if __name__ == "__main__":
	main(["2022-06-05", "2022-06-07", "2022-06-09"])
