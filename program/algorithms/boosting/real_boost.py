from time import time
from measuring_quality import MeasuringQuality
import logging

class RealBoost(object):

	def __init__(self, data):
		self.data = data
		self.classifier = RealBoostImpl()
		self.name = "RealBoost"
		self.description = "An RealBoost classifier"
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


class RealBoostImpl(object):
	def __init__(self):
		pass

	def fit(self, X, y):
		pass

	def predict_proba(self, X):
		pass

	def predict(self, X):
		pass
