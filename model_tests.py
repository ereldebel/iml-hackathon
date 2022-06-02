import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def match_columns(df_columns: pd.DataFrame, df_other: pd.DataFrame):
	"matches the columns of df_other to the columns of df_columns"
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


def train_and_predict(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, label):
	y_train, y_test = y_train[label], y_test[label]
	if label not in ['x_label', 'y_label']:
		model = ExtraTreesClassifier()
	else:
		model = make_pipeline(StandardScaler(), LinearRegression())
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	if label not in ['x_label', 'y_label']:
		print(f"label={label} performance: {f1_score(y_test, y_pred, average='macro')}")
	else:
		print(f"label={label} performance: {mean_squared_error(y_test, y_pred)}")


def evaluate_train_set(X_train, y_train):
	test_X_path = "datasets/test_set_X_TLV.csv"
	X_test = pd.read_csv(test_X_path)
	test_y_path = "datasets/test_set_y_TLV.csv"
	y_test = pd.read_csv(test_y_path)

	X_test = match_columns(X_train, X_test)
	y_test = match_columns(y_train, y_test)

	train_and_predict(X_train, y_train, X_test, y_test, label='linqmap_type_label')
	train_and_predict(X_train, y_train, X_test, y_test, label='linqmap_subtype_label')
	train_and_predict(X_train, y_train, X_test, y_test, label='x_label')
	train_and_predict(X_train, y_train, X_test, y_test, label='y_label')


if __name__ == '__main__':
	train_X_path = "datasets/train_set_X_all_cities.csv"
	train_y_path = "datasets/train_set_y_all_cities.csv"
	X_train = pd.read_csv(train_X_path)
	y_train = pd.read_csv(train_y_path)
	evaluate_train_set(X_train, y_train)

	print()

	train_X_path = "datasets/train_set_X_TLV.csv"
	train_y_path = "datasets/train_set_y_TLV.csv"
	X_train = pd.read_csv(train_X_path)
	y_train = pd.read_csv(train_y_path)
	evaluate_train_set(X_train, y_train)
