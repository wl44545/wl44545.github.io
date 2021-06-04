"""
Moduł zawierający miary jakości klasyfikacji.
"""


class MeasuringQuality:
	"""
	Miary jakości klasyfikacji.
	"""
	def __init__(self):
		"""
		Konstruktor.
		"""
		self.true_positive = 0
		self.true_negative = 0
		self.false_positive = 0
		self.false_negative = 0
		self.recall = 0
		self.sensitivity = 0
		self.specificity = 0
		self.fall_out = 0
		self.precision = 0
		self.accuracy = 0
		self.error = 0
		self.f1 = 0

		self.confusion_matrix = None
		self.roc_curve = None

		self.actual_labels = None
		self.predicted_labels = None

	def update(self, actual, predicted):
		"""
		Metoda przesyłająca etykietki klas.
		"""
		self.actual_labels = actual
		self.predicted_labels = predicted

	def calculate(self):
		"""
		Metoda obliczająca statystyki.
		"""
		for i in range(len(self.actual_labels)):
			if self.actual_labels[i] == 0 and self.predicted_labels[i] == 0:
				self.true_negative += 1
			elif self.actual_labels[i] == 1 and self.predicted_labels[i] == 1:
				self.true_positive += 1
			elif self.actual_labels[i] == 1 and self.predicted_labels[i] == 0:
				self.false_negative += 1
			elif self.actual_labels[i] == 0 and self.predicted_labels[i] == 1:
				self.false_positive += 1

		if (self.true_negative + self.true_positive) != 0:
			self.recall = self.true_positive / (self.true_negative + self.true_positive)
		else:
			self.recall = 0

		if (self.true_positive + self.false_negative) != 0:
			self.sensitivity = self.true_positive / (self.true_positive + self.false_negative)
		else:
			self.sensitivity = 0

		if (self.false_positive + self.true_negative) != 0:
			self.specificity = self.true_negative / (self.false_positive + self.true_negative)
		else:
			self.specificity = 0

		if (self.false_positive + self.false_negative) != 0:
			self.fall_out = self.false_negative / (self.false_positive + self.false_negative)
		else:
			self.fall_out = 0

		if (self.true_positive + self.false_positive) != 0:
			self.precision = self.true_positive / (self.true_positive + self.false_positive)
		else:
			self.precision = 0

		self.accuracy = (self.true_positive + self.true_negative) / len(self.actual_labels)
		self.error = (self.false_negative + self.false_positive) / len(self.actual_labels)

		if (self.recall + self.sensitivity) != 0:
			self.f1 = (2 * self.recall * self.sensitivity) / (self.recall + self.sensitivity)
		else:
			self.f1 = 0
