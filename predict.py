import pandas as pd
from unified_estimator import UnifiedEstimator

if __name__ == '__main__':
	train_X_path = "datasets/train_set_X_all_cities.csv"
	train_X = pd.read_csv(train_X_path)
	train_y_path = "datasets/train_set_y_all_cities.csv"
	train_y = pd.read_csv(train_y_path)

	test_X_path = "datasets/test_set_X_all_cities.csv"
	test_X = pd.read_csv(test_X_path)
	test_y_path = "datasets/test_set_y_all_cities.csv"
	test_y = pd.read_csv(test_y_path)

	model = UnifiedEstimator().fit(test_X, test_y)
	model.loss(train_X, train_y)
