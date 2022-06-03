import pandas as pd

from preprocess import get_data
from unified_estimator import UnifiedEstimator


def task_1_main(data_set_path, test_set_path):
	train_X, train_y, test_X = get_data(data_set_path, test_set_path)
	model = UnifiedEstimator().fit(train_X, train_y)
	predictions = pd.DataFrame(model.predict(test_X))
	predictions.columns = ["linqmap_type", "linqmap_subtype", "x", "y"]
	predictions.to_csv(f"predictions.csv", index=False)


if __name__ == '__main__':
	task_1_main("../Mission 1 - Waze/waze_data.csv",
	            "../Mission 1 - Waze/waze_take_features.csv")
