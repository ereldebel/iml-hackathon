import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from typing import Tuple


class UnifiedEstimator:
	"""
	estimator class, include all the estimators use to fit and predict
	the relevant labels.
	"""
	def __init__(self,
				 type_model=ExtraTreesClassifier(max_depth=18, random_state=9),
	             subtype_model=ExtraTreesClassifier(max_depth=22, random_state=0),
	             x_model=RandomForestRegressor(max_depth=16, random_state=0),
	             y_model=RandomForestRegressor(max_depth=24, random_state=3)):
		# classifier to predict type
		self._type_model = type_model
		# classifier to predict subtype
		self._subtype_model = subtype_model
		# regressor to predict x_coordinate
		self._x_model = x_model
		# regressor to predict y_coordinate
		self._y_model = y_model
		self._fitted = False

	def fit(self, X: pd.DataFrame, y: pd.DataFrame):
		"""
		fit all the estimators
		:param X:
		:param y:
		:return:
		"""
		self._type_model.fit(X, y["linqmap_type_label"])
		self._subtype_model.fit(X, y["linqmap_subtype_label"])
		self._x_model.fit(X, y["x_label"])
		self._y_model.fit(X, y["y_label"])
		self._fitted = True
		return self

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		"""
		predict all the labels
		:param X:
		:return:
		"""
		if not self._fitted:
			print("not fitted")
		type_pred = self._type_model.predict(X).reshape(-1, 1)
		subtype_pred = self._subtype_model.predict(X).reshape(-1, 1)
		x_pred = self._x_model.predict(X).reshape(-1, 1)
		y_pred = self._y_model.predict(X).reshape(-1, 1)
		return np.column_stack([type_pred, subtype_pred, x_pred, y_pred])

	def loss(self, X: pd.DataFrame, y_true: pd.DataFrame) -> Tuple[float, float, float, float]:
		"""
		calculate loss, using the relevant loss functions, print and return them
		:param X:
		:param y_true:
		:return:
		"""
		y_pred = self.predict(X)
		y_pred_type = y_pred[:, 0]
		y_pred_subtype = y_pred[:, 1]
		y_pred_x = y_pred[:, 2]
		y_pred_y = y_pred[:, 3]

		y_true_type = y_true["linqmap_type_label"].values
		y_true_subtype = y_true["linqmap_subtype_label"].values
		y_true_x = y_true["x_label"].values
		y_true_y = y_true["y_label"].values

		# print
		type_loss = f1_score(y_true_type, y_pred_type, average="macro")
		print("type f1 macro:", type_loss)
		ConfusionMatrixDisplay.from_estimator(self._type_model, X, y_true_type)
		plt.show()
		subtype_loss = f1_score(y_true_subtype, y_pred_subtype, average="macro")
		print("subtype f1 macro:", subtype_loss)
		ConfusionMatrixDisplay.from_estimator(self._subtype_model, X, y_true_subtype)
		plt.show()
		x_loss = mean_squared_error(y_true_x, y_pred_x)
		print("x MSE: ", x_loss)
		y_loss = mean_squared_error(y_true_y, y_pred_y)
		print("y MSE: ", y_loss)
		return type_loss, subtype_loss, x_loss, y_loss

	def x_loss(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> np.ndarray:
		"""
		x loss function
		:param y_pred:
		:param y_true:
		:return:
		"""
		y_pred_x = y_pred[:, 0]
		y_true_x = y_true["x_label"].values
		return mean_squared_error(y_true_x, y_pred_x)

	def y_loss(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> np.ndarray:
		"""
		y loss function
		:param y_pred:
		:param y_true:
		:return:
		"""
		y_pred_x = y_pred[:, 1]
		y_true_x = y_true["y_label"].values
		return mean_squared_error(y_true_x, y_pred_x)

	def type_loss(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> np.ndarray:
		"""
		type loss function
		:param y_pred:
		:param y_true:
		:return:
		"""
		y_pred_type = y_pred[:, 2]
		y_true_type = y_true["linqmap_type_label"].values
		return f1_score(y_true_type, y_pred_type, average="macro")

	def subtype_loss(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> np.ndarray:
		"""
		subtype loss function
		:param y_pred:
		:param y_true:
		:return:
		"""
		y_pred_subtype = y_pred[:, 3]
		y_true_subtype = y_true["linqmap_subtype_label"].values
		return f1_score(y_true_subtype, y_pred_subtype, average="macro")
