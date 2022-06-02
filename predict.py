import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from process_features import process_features_single, process_features_combined
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def train_and_predict(X: pd.DataFrame, y: pd.DataFrame):
	y = y["linqmap_subtype_label"]
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2 / 3)
	model = ExtraTreesClassifier().fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print(f1_score(y_test, y_pred, average="macro"))
	ConfusionMatrixDisplay.from_estimator(
		model, X_test, y_test)
	plt.show()


if __name__ == '__main__':
	test_X_path = "datasets/train_set_X_TLV.csv"
	X = pd.read_csv(test_X_path)
	test_y_path = "datasets/train_set_y_TLV.csv"
	y = pd.read_csv(test_y_path)
	train_and_predict(X, y)
