import pandas as pd
from task_1.preprocess import get_data
from unified_estimator import UnifiedEstimator

if __name__ == '__main__':
	train_X, train_y, test_X = get_data(True)
	model = UnifiedEstimator().fit(train_X, train_y)
	predictions = pd.DataFrame(model.predict(test_X))
	predictions.columns = ["linqmap_type", "linqmap_subtype", "x", "y"]
	predictions.to_csv(f"predictions.csv", index=False)
