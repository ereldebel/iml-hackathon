import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


def train_and_predict(X: pd.DataFrame, y: pd.DataFrame):
	y = y["linqmap_type_label"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2 / 3)
	model = ExtraTreesClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print(f1_score(y_test, y_pred, average='macro'))
	print(f1_score(y_test, y_pred, average='micro'))
	print(confusion_matrix(y_test, y_pred, labels=y_train.unique()))


if __name__ == '__main__':
	test_X_path = "datasets/train_set_X_TLV.csv"
	X = pd.read_csv(test_X_path)
	test_y_path = "datasets/train_set_y_TLV.csv"
	y = pd.read_csv(test_y_path)
	train_and_predict(X, y)
