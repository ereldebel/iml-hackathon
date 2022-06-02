import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from process_features import process_features_single, process_features_combined
from sklearn.metrics import f1_score, confusion_matrix


def train_and_predict(X: pd.DataFrame, y: pd.DataFrame):
	y = y["linqmap_type_label"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2 / 3)
	model = ExtraTreesClassifier().fit(X_train, y_train)
	y_pred = model.fit(X_train)
	print(f1_score(y_train, y_pred))
	print(confusion_matrix(y_train, y_pred, labels=y_train.unique()))


if __name__ == '__main__':
	test_X_path = "datasets/train_set_X_TLV.csv"
	X = pd.read_csv(test_X_path)
	test_y_path = "datasets/train_set_y_TLV.csv"
	y = pd.read_csv(test_y_path)
	train_and_predict(X, y)
