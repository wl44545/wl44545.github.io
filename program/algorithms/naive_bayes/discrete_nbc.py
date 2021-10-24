from time import time
from measuring_quality import MeasuringQuality
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class DiscreteNBC:
	"""
	Algorytm DiscreteNBC
	"""
	def __init__(self, data):
		self.data = data
		self.classifier = DiscreteNBCImpl()

	def start(self):

		m, n = self.data.X_train.shape
		B = 256
		self.classifier.config(domain_sizes=np.ones(n).astype("int32") * B, laplace=True, logarithm=True)

		train_time_start = time()
		self.classifier.fit(self.data.X_train, self.data.y_train)
		train_time_stop = time()

		predict_time_start = time()
		y_pred = self.classifier.predict(self.data.X_test)
		y_score = self.classifier.predict_proba(self.data.X_test)
		predict_time_stop = time()

		return MeasuringQuality("DiscreteNBC","DiscreteNBC", train_time_stop-train_time_start,predict_time_stop-predict_time_start,self.data.y_test,y_pred,y_score)


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
