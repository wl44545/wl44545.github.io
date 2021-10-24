"""
Moduł zawierający miary jakości klasyfikacji.
"""
from data import Data
from measuring_quality import MeasuringQuality
import pandas as pd
import os
import shutil



class Statistics:
	"""
	Miary jakości klasyfikacji.
	"""

	def __init__(self):
		"""
		Konstruktor.
		"""
		self.data_info = None
		self.data = []
		self.dataframe = None
		shutil.rmtree("resources/results/images")
		os.mkdir("resources/results/images")

	def insert(self, measuring_quality: MeasuringQuality):
		self.data.append([measuring_quality.method, measuring_quality.description, measuring_quality.train_time,
		                  measuring_quality.predict_time, measuring_quality.true_positive,
		                  measuring_quality.true_negative,
		                  measuring_quality.false_positive, measuring_quality.false_negative,
		                  measuring_quality.sensitivity, measuring_quality.specificity,
		                  measuring_quality.precision, measuring_quality.accuracy, measuring_quality.error,
		                  measuring_quality.f1, measuring_quality.confusion_matrix, measuring_quality.roc_curve])

		print(measuring_quality.method, measuring_quality.description, measuring_quality.train_time,
		                  measuring_quality.predict_time, measuring_quality.true_positive,
		                  measuring_quality.true_negative,
		                  measuring_quality.false_positive, measuring_quality.false_negative,
		                  measuring_quality.sensitivity, measuring_quality.specificity,
		                  measuring_quality.precision, measuring_quality.accuracy, measuring_quality.error,
		                  measuring_quality.f1)

	def create_statistics(self):
		self.dataframe = pd.DataFrame(self.data,
		                              columns=['method', 'description', 'train_time', 'predict_time', 'true_positive',
		                                       'true_negative', 'false_positive',
		                                       'false_negative',
		                                       'sensitivity', 'specificity', 'precision',
		                                       'accuracy', 'error',
		                                       'f1', 'confusion_matrix', 'roc_curve'])

	def update_data(self, data:Data):
		self.data_info = data.data_size, data.augmented_size, len(data.X_train), len(data.X_test)

	def show(self):
		print(self.dataframe)

	def export_csv(self):
		self.dataframe.to_csv(r'resources\results\result.csv', index=False, header=True)

	def export_html(self):
		datainfo = "<p2>Original data size: "+str(self.data_info[0])+"</p2><br>"+"<p2>Data size afer augmentation: "+str(self.data_info[1])+"</p2><br>"+"<p2>Train data size: "+str(self.data_info[2])+"</p2><br>"+"<p2>Test data size: "+str(self.data_info[3])+"</p2><br><br><br>"
		html = str(self.dataframe.to_html()).replace("&lt;","<").replace("&gt;",">")
		text_file = open(r'resources\results\result.html', "w", encoding="utf-8")
		text_file.write(datainfo)
		text_file.write(html)
		text_file.close()
