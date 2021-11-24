from time import time
import tensorflow.keras.applications as models
from algorithms.neural_networks.cnn import CNN
from measuring_quality import MeasuringQuality
import logging


class ResNet152(object):

	def __init__(self, data):
		self.name = "ResNet152"
		self.description = ""
		self.cnn = CNN(models.ResNet152, data, 32, 32, self.name)
		logging.info("Algorithm initialized")

	def start(self):
		logging.info("Training started")
		train_time_start = time()
		history = self.cnn.fit()
		train_time_stop = time()
		logging.info("Training completed")
		logging.info("Training results: " + str(history.params) + str(history.history))

		logging.info("Prediction started")
		predict_time_start = time()
		y_true, y_pred = self.cnn.predict()
		predict_time_stop = time()
		logging.info("Prediction completed")

		mq = MeasuringQuality(self.name, self.description, train_time_stop-train_time_start, predict_time_stop-predict_time_start)
		mq.calculate_neural_network(y_true, y_pred)
		logging.info("Prediction results: " + str(mq))
		return mq
