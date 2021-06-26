"""
Moduł zawierający liniowy algorytm KNeighbors
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from time import time
from measuring_quality import MeasuringQuality


class KNeighbors(object):
	"""
	Algorytm KNeighbors
	"""
	def __init__(self, data, n_neighbors=5):
		self.data = data
		self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

	def start(self):
		train_time_start = time()
		self.classifier.fit(self.data.X_train, self.data.y_train)
		train_time_stop = time()

		predict_time_start = time()
		y_pred = self.classifier.predict(self.data.X_test)
		y_score = self.classifier.predict_proba(self.data.X_test)
		predict_time_stop = time()

		return MeasuringQuality("KNeighborsClassifier","Classifier implementing the k-nearest neighbors vote.", train_time_stop-train_time_start,predict_time_stop-predict_time_start,self.data.y_test,y_pred,y_score)