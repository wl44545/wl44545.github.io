"""
Moduł zawierający liniowy algorytm Support Vector Machines
"""
from sklearn.svm import LinearSVC
from time import time
from measuring_quality import MeasuringQuality


class LinearSVM(object):
	"""
	Liniowy algorytm Support Vector Machines.
	"""
	def __init__(self, data):
		self.data = data
		self.classifier = LinearSVC()

	def start(self):
		train_time_start = time()
		self.classifier.fit(self.data.X_train, self.data.y_train)
		train_time_stop = time()

		predict_time_start = time()
		y_pred = self.classifier.predict(self.data.X_test)
		predict_time_stop = time()

		return MeasuringQuality("LinearSVM","Linear Support Vector Classification", train_time_stop-train_time_start,predict_time_stop-predict_time_start,self.data.y_test,y_pred)