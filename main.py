from typing import List
import pandas as pd

from task_1.task_1 import task_1


def main(data_set_path: str, train_set_1: str, date_list: List[pd.date]):
	try:
		task_1(data_set_path,train_set_1)
	except(Exception):
		pass
	try:
		task_2(date_list)
	except(Exception):
		pass


if __name__ == '__main__':
	main("./Mission 1 - Waze/waze_data.csv",
	     "./Mission 1 - Waze/waze_take_features.csv",
	     ["2022-06-05", "2022-06-07", "2022-06-09"])
