from typing import List, Any

from task_1_ import task_1_main
from task_2 import task_2_main


def main(data_set_path: str, train_set_1: str, date_list: List[Any]):
	try:
		task_1_main(data_set_path, train_set_1)
	except Exception:
		pass
	try:
		task_2_main(data_set_path, date_list)
	except Exception:
		pass

if __name__ == '__main__':
	main("./Mission 1 - Waze/waze_data.csv",
	     "./Mission 1 - Waze/waze_take_features.csv",
	     ["2022-06-05", "2022-06-07", "2022-06-09"])
