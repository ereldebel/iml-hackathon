import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from process_features import process_features_single, process_features_combined
from sklearn.metrics import f1_score, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from typing import NoReturn


class UnifiedEstimator:
	# def __init__(self, type_model=ExtraTreesClassifier(),
	# 			 subtype_model=ExtraTreesClassifier(),
	# 			 x_model = GradientBoostingRegressor(),
	# 			 y_model = GradientBoostingRegressor()):
	#
	# 	self._type_model = type_model
	# 	self._subtype_model =subtype_model
	# 	self._x_model = x_model
	# 	self._y_model = y_model
	# 	self._fitted = False

	def __init__(self,x_model = GradientBoostingRegressor(),
				 y_model = GradientBoostingRegressor()):

		self._x_model = x_model
		self._y_model = y_model
		self._fitted = False

	def fit(self, X: pd.DataFrame, y: pd.DataFrame):
		# self._type_model.fit(X, y["linqmap_type_label"])
		# self._subtype_model.fit(X, y["linqmap_subtype_label"])
		self._x_model.fit(X, y["x_label"])
		self._y_model.fit(X, y["y_label"])
		self._fitted = True
		return self

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		if not self._fitted:
			print("not fitted")
		# type_pred = self._type_model.predict(X).reshape(-1, 1)
		# subtype_pred = self._subtype_model.predict(X).reshape(-1, 1)
		x_pred = self._x_model.predict(X).reshape(-1, 1)
		y_pred = self._y_model.predict(X).reshape(-1, 1)
		# return np.column_stack([x_pred, y_pred, type_pred, subtype_pred])
		return np.column_stack([x_pred, y_pred])

	def loss(self, X: pd.DataFrame, y_true: pd.DataFrame) -> np.ndarray:
		y_pred = self.predict(X)
		y_pred_x = y_pred[:, 0]
		y_pred_y = y_pred[:, 1]
		# y_pred_type = y_pred[:, 2]
		# y_pred_subtype = y_pred[:, 3]

		y_true_x = y_true["x_label"].values
		y_true_y = y_true["y_label"].values
		# y_true_type = y_true["linqmap_type_label"].values
		# y_true_subtype = y_true["linqmap_subtype_label"].values

		# print("type f1 macro:",
		#       f1_score(y_true_type, y_pred_type, average="macro"))
		# ConfusionMatrixDisplay.from_estimator(self._type_model, X, y_true_type)
		# plt.show()
		# print("subtype f1 macro:",
		      # f1_score(y_true_subtype, y_pred_subtype, average="macro"))
		# ConfusionMatrixDisplay.from_estimator(self._subtype_model, X,
		#                                       y_true_subtype)
		# plt.show()
		# print("x MSE: ", mean_squared_error(y_true_x, y_pred_x))
		# print("y MSE: ", mean_squared_error(y_true_y, y_pred_y))
		return mean_squared_error(y_true_x, y_pred_x), \
			   mean_squared_error(y_true_y, y_pred_y)
		# return y_pred

	def x_loss(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> np.ndarray:
		y_pred_x = y_pred[:, 0]
		y_true_x = y_true["x_label"].values
		return mean_squared_error(y_true_x, y_pred_x)

	def y_loss(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> np.ndarray:
		y_pred_x = y_pred[:, 1]
		y_true_x = y_true["y_label"].values
		return mean_squared_error(y_true_x, y_pred_x)

	def type_loss(self, y_pred: pd.DataFrame,
	              y_true: pd.DataFrame) -> np.ndarray:
		y_pred_type = y_pred[:, 2]
		y_true_type = y_true["linqmap_type_label"].values
		return f1_score(y_true_type, y_pred_type, average="macro")

	def subtype_loss(self, y_pred: pd.DataFrame,
	                 y_true: pd.DataFrame) -> np.ndarray:
		y_pred_subtype = y_pred[:, 3]
		y_true_subtype = y_true["linqmap_subtype_label"].values
		return f1_score(y_true_subtype, y_pred_subtype, average="macro")
