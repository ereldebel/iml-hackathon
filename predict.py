import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from unified_estimator import UnifiedEstimator
from process_features import process_features_single, process_features_combined
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def train_and_predict(X: pd.DataFrame, y: pd.DataFrame):
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2 / 3)
	model = UnifiedEstimator().fit(X_train, y_train)
	model.loss(X_test, y_test)


if __name__ == '__main__':
	test_X_path = "datasets/train_set_X_TLV.csv"
	X = pd.read_csv(test_X_path)
	test_y_path = "datasets/train_set_y_TLV.csv"
	y = pd.read_csv(test_y_path)
	train_and_predict(X, y)
