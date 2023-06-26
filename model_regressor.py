from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def estimator_model(estimator, X, y, X_v, y_v):
	"""A function to train and validate multiple sklearn-estimator and returns the value of R2, MAE, MAX_Error for the training and validation data.
	It also returns the predicted values for training and validation.
	estimator: Sklearn regressor, for example LinearRegression() or RandomForestRegressor
	X : The training data vector (features), type: np.array and np.shape (n_entries, n_features)
	y : The training data a one dimensional array, type: np.array
	X_v: The validation data (features)m type: np.array
	y_v: The validation data a one dimensional array, type: np.array and np.shape: (n_entries,)"""
	model = estimator
	model.fit(X, y)
	r_2, y_pre = model.score(X, y), model.predict(X)
	mae, max_ = mean_absolute_error(y, y_pre), max_error(y, y_pre)
	y_pre_v = model.predict(X_v)
	r_2_v = model.score(X_v, y_v)
	mae_v, max_v = mean_absolute_error(y_v, y_pre_v), max_error(y_v, y_pre_v)
	return [r_2, mae, max_, r_2_v, mae_v, max_v], [y, y_pre, y_v, y_pre_v], model