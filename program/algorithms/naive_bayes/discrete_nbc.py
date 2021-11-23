from time import time
from measuring_quality import MeasuringQuality
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import logging


class DiscreteNBC:
	"""
	Algorytm DiscreteNBC
	"""
	def __init__(self, data):
		self.data = data
		self.classifier = DiscreteNBCImpl()
		self.name = "DiscreteNBC"
		self.description = "DiscreteNBC"
		logging.info("Algorithm initialized")

	def start(self):

		B = 256
		X_train = discretize(self.data.X_train, B)
		X_test = discretize(self.data.X_test, B)
		m, n = X_train.shape

		self.classifier.config(domain_sizes=np.ones(n).astype("int32") * B, laplace=True, logarithm=True)

		logging.info("Training started")
		train_time_start = time()
		self.classifier.fit(X_train, self.data.y_train)
		train_time_stop = time()
		logging.info("Training completed")

		logging.info("Prediction started")
		predict_time_start = time()
		y_pred = self.classifier.predict(X_test)
		predict_time_stop = time()
		logging.info("Prediction completed")
		y_score = self.classifier.predict_proba(X_test)

		mq = MeasuringQuality(self.name, self.description, train_time_stop-train_time_start, predict_time_stop-predict_time_start)
		mq.calculate_algorithm(self.data.y_test, y_pred, y_score)
		logging.info("Prediction results: " + str(mq))
		return mq


def discretize(X, B, X_train_ref=None):
	if X_train_ref is None:
		X_train_ref = X
	mins = np.min(X_train_ref, axis=0)
	maxes = np.max(X_train_ref, axis=0)
	X = np.floor(((X - mins) / (maxes - mins)) * B).astype("int32")
	X = np.clip(X, 0, B - 1)
	return X


class DiscreteNBCImpl(BaseEstimator, ClassifierMixin):

	def __init__(self):
		self.laplace_ = None
		self.logarithm = None
		self.class_labels_ = None
		self.PY_ = None
		self.P_ = None
		self.domain_sizes_ = None

	def config(self, domain_sizes, laplace=False, logarithm=False):
		self.laplace_ = laplace
		self.logarithm = logarithm
		self.domain_sizes_ = domain_sizes

	def fit(self, X, y):
		m, n = X.shape
		self.class_labels_ = np.unique(y)

		y_normalized = np.zeros(m).astype("int32")
		for yy, label in enumerate(self.class_labels_):
			indexes = y == label
			y_normalized[indexes] = yy

		self.PY_ = np.zeros(self.class_labels_.size)
		for yy, label in enumerate(self.class_labels_):
			self.PY_[yy] = (y == label).sum() / m
		self.P_ = np.empty((self.class_labels_.size, n), dtype=object)
		for yy, label in enumerate(self.class_labels_):
			for j in range(n):
				self.P_[yy, j] = np.zeros(self.domain_sizes_[j])
		for i in range(m):
			for j in range(n):
				self.P_[y_normalized[i], j][X[i, j]] += 1
		if not self.laplace_:
			for yy, label in enumerate(self.class_labels_):
				for j in range(n):
					self.P_[yy, j] /= self.P_[yy, j].sum()
		else:
			for yy, label in enumerate(self.class_labels_):
				for j in range(n):
					self.P_[yy, j] = (self.P_[yy, j] + 1) / (self.P_[yy, j].sum() + self.domain_sizes_[j])
		if self.logarithm:
			for yy, label in enumerate(self.class_labels_):
				for j in range(n):
					self.P_[yy, j] = np.log(self.P_[yy, j])
			self.PY_ = np.log(self.PY_)

	def predict_proba(self, X):
		m, n = X.shape
		if self.logarithm:
			probs = np.zeros((m, self.class_labels_.size))
		else:
			probs = np.ones((m, self.class_labels_.size))
		for i in range(m):
			x = X[i]
			for y in range(self.class_labels_.size):
				if self.logarithm:
					for j in range(n):
						probs[i, y] += self.P_[y, j][X[i, j]]
					probs[i, y] += self.PY_[y]
					# probs[i] = np.exp(probs[i])
				else:
					for j in range(n):
						probs[i, y] *= self.P_[y, j][X[i, j]]
					probs[i, y] *= self.PY_[y]
			s = probs[i].sum()
			if s > 0:
				probs[i] /= s
		return probs

	def predict(self, X):
		return self.class_labels_[np.argmax(self.predict_proba(X), axis=1)]
