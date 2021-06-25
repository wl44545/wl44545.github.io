"""
Moduł zawierający miary jakości klasyfikacji.
"""
from measuring_quality import MeasuringQuality
import pandas as pd


class Statistics:
	"""
	Miary jakości klasyfikacji.
	"""

	def __init__(self):
		"""
		Konstruktor.
		"""
		self.data = []
		self.dataframe = None

	def insert(self, measuring_quality: MeasuringQuality):
		self.data.append([measuring_quality.name, measuring_quality.true_positive, measuring_quality.true_negative,
		                  measuring_quality.false_positive, measuring_quality.false_negative, measuring_quality.recall,
		                  measuring_quality.sensitivity, measuring_quality.specificity, measuring_quality.fall_out,
		                  measuring_quality.precision, measuring_quality.accuracy, measuring_quality.error,
		                  measuring_quality.f1, measuring_quality.confusion_matrix, measuring_quality.roc_curve])

	def create_statistics(self):
		self.dataframe = pd.DataFrame(self.data,
		                  columns=['name', 'true_positive', 'true_negative', 'false_positive', 'false_negative',
		                           'recall', 'sensitivity', 'specificity', 'fall_out', 'precision', 'accuracy', 'error',
		                           'f1', 'confusion_matrix', 'roc_curve'])

	def show(self):
		print(self.dataframe)

	def export_csv(self):
		self.dataframe.to_csv(r'resources\results\result.csv', index=False, header=True)

	def export_html(self):
		html = self.dataframe.to_html()
		text_file = open(r'resources\results\result.html', "w")
		text_file.write(html)
		text_file.close()
