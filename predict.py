import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
from sklearn.metrics import f1_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from typing import Callable, Tuple

from unified_estimator import UnifiedEstimator

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def train_and_predict(X: pd.DataFrame, y: pd.DataFrame):
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2 / 3)
	values = np.ndarray([19, 2], float)
	k_range = range(1, 20)
	for k in k_range:
		values[k - 1] = cross_validate(
			ExtraTreesClassifier(max_depth=k, random_state=0), X_train,
			y_train, type_loss,
			"linqmap_type_label")

	maximizer = np.argmax(values[:, 1])
	fig = go.Figure(
		[go.Scatter(x=list(k_range), y=values[:, 0], mode="lines+markers",
		            name="train score"),
		 go.Scatter(x=list(k_range), y=values[:, 1], mode="lines+markers",
		            name="validation score"),
		 go.Scatter(x=[maximizer], y=[values[maximizer + 1, 1]],
		            mode="markers",
		            name="validation score maximizer")
		 ], layout=go.Layout(
			title=rf"$\text{{Mean Train and Validation score Using 5-fold"
			      rf" Cross Validation.}}$"))
	fig.show()
	model = ExtraTreesClassifier(max_depth=maximizer + 1, random_state=0)
	model.fit(X_train, y_train)
	type_loss(y_test, model.predict(X_test))


def x_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame
           ) -> np.ndarray:
	return mean_squared_error(y_true, y_pred)


def y_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame
           ) -> np.ndarray:
	return mean_squared_error(y_true, y_pred)


def type_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame
              ) -> np.ndarray:
	return f1_score(y_true, y_pred, average="macro")


def subtype_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame
                 ) -> np.ndarray:
	return f1_score(y_true, y_pred, average="macro")


def cross_validate(estimator, X: np.ndarray, y: np.ndarray,
                   scoring,
                   column: str, cv: int = 5) -> Tuple[float, float]:
	X_parts = np.array_split(X, cv)
	y_col_parts = np.array_split(y[column], cv)
	train_sum, validation_sum = 0, 0
	for k in range(cv):
		X_k_fold = np.concatenate(
			[part for j, part in enumerate(X_parts) if k != j])
		y_col_k_fold = np.concatenate(
			[part for j, part in enumerate(y_col_parts) if k != j])
		estimator.fit(X_k_fold, y_col_k_fold)
		train_sum += scoring(estimator.predict(X_k_fold), y_col_k_fold)
		validation_sum += scoring(estimator.predict(X_parts[k]),
		                          y_col_parts[k])
	return train_sum / cv, validation_sum / cv


if __name__ == '__main__':
	test_X_path = "datasets/train_set_X_TLV.csv"
	X = pd.read_csv(test_X_path)
	test_y_path = "datasets/train_set_y_TLV.csv"
	y = pd.read_csv(test_y_path)
	train_and_predict(X, y)

	test_X_path = "datasets/train_set_X_all_cities.csv"
	X = pd.read_csv(test_X_path)
	test_y_path = "datasets/train_set_y_all_cities.csv"
	y = pd.read_csv(test_y_path)
	train_and_predict(X, y)
