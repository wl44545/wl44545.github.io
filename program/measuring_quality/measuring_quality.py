"""
Moduł zawierający miary jakości klasyfikacji.
"""
from sklearn.metrics import *

class MeasuringQuality:
	"""
	Miary jakości klasyfikacji.
	"""

	def __init__(self, method, description, train_time, predict_time, y_true, y_pred):
		"""
		Konstruktor.
		"""
		self.method = method
		self.description = description
		self.train_time = train_time
		self.predict_time = predict_time
		self.y_true = y_true
		self.y_pred = y_pred

		self.confusion_matrix = None
		self.roc_curve = None

		self.true_positive = 0
		self.true_negative = 0
		self.false_positive = 0
		self.false_negative = 0
		self.recall = 0.0
		self.sensitivity = 0.0
		self.specificity = 0.0
		self.fall_out = 0.0
		self.precision = 0.0
		self.accuracy = 0.0
		self.error = 0.0
		self.f1 = 0.0

		self.calculate()



	def calculate(self):
		"""
		Metoda obliczająca statystyki.
		"""
		self.confusion_matrix = multilabel_confusion_matrix(self.y_true, self.y_pred)

		self.true_positive = self.confusion_matrix[:, 1, 1]
		self.true_negative = self.confusion_matrix[:, 0, 0]
		self.false_positive = self.confusion_matrix[:, 0, 1]
		self.false_negative = self.confusion_matrix[:, 1, 0]

		self.recall = self.true_positive / (self.true_negative + self.true_positive)
		self.sensitivity = self.true_positive / (self.true_positive + self.false_negative )
		self.specificity = self.true_negative / (self.true_negative + self.false_positive)
		self.fall_out = self.false_negative / (self.false_positive + self.false_negative)
		self.precision = self.true_positive / (self.true_positive + self.false_positive)
		self.accuracy = (self.true_positive + self.true_negative) / (len(self.true_positive) + len(self.true_negative) +len(self.false_positive) + len(self.false_negative))
		self.error = (self.false_negative + self.false_positive) / (len(self.true_positive) + len(self.true_negative) +len(self.false_positive) + len(self.false_negative))
		self.f1 = (2 * self.recall * self.sensitivity) / (self.recall + self.sensitivity)
