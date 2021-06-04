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
