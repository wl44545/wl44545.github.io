"""
Moduł zawierający liniowy algorytm KNeighbors
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from time import time
from measuring_quality import MeasuringQuality
import logging


class KNeighbors(object):
	"""
	Algorytm KNeighbors
	"""
	def __init__(self, data, n_neighbors=5):
		self.data = data
		self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
		self.name = "KNeighborsClassifier"
		self.description = "Classifier implementing the k-nearest neighbors vote"
		logging.info("Algorithm initialized")

	def start(self):
		logging.info("Training started")
		train_time_start = time()
		self.classifier.fit(self.data.X_train, self.data.y_train)
		train_time_stop = time()
		logging.info("Training completed")

		logging.info("Prediction started")
		predict_time_start = time()
		y_pred = self.classifier.predict(self.data.X_test)
		predict_time_stop = time()
		logging.info("Prediction completed")
		y_score = self.classifier.predict_proba(self.data.X_test)

		mq = MeasuringQuality(self.name, self.description, train_time_stop-train_time_start, predict_time_stop-predict_time_start)
		mq.calculate_algorithm(self.data.y_test, y_pred, y_score)
		logging.info("Prediction results: " + str(mq))
		return mq
