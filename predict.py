import pandas as pd
from unified_estimator import UnifiedEstimator

if __name__ == '__main__':
	for data_name in ("TLV","all_cities"):
		print (data_name)

		train_X_path = f"datasets/train_set_X_{data_name}.csv"
		train_X = pd.read_csv(train_X_path)
		train_y_path = f"datasets/train_set_y_{data_name}.csv"
		train_y = pd.read_csv(train_y_path)

		test_X_path = f"datasets/test_set_X_{data_name}.csv"
		test_X = pd.read_csv(test_X_path)
		test_y_path = f"datasets/test_set_y_{data_name}.csv"
		test_y = pd.read_csv(test_y_path)

		model = UnifiedEstimator().fit(train_X, train_y)
		model.loss(test_X, test_y)