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
		                  measuring_quality.f1, measuring_quality.my_score, measuring_quality.confusion_matrix,
		                  measuring_quality.confusion_matrix_percentage, measuring_quality.roc_curve])

	def create_statistics(self):
		dataframe = pd.DataFrame(self.data,
		                              columns=['method', 'description', 'train_time', 'predict_time', 'true_positive',
		                                       'true_negative', 'false_positive',
		                                       'false_negative',
		                                       'sensitivity', 'specificity', 'precision',
		                                       'accuracy', 'error',
		                                       'f1', 'my_score', 'confusion_matrix',
		                                       'confusion_matrix_percentage', 'roc_curve'])
		self.dataframe = dataframe.sort_values("my_score", ascending=False)

	def update_data(self, data:Data):
		self.data_info = data.original_size, data.data_size

	def show(self):
		print(self.dataframe)

	def export_csv(self):
		self.dataframe.to_csv(r'resources\results\results.csv', index=False, header=True)

	def export_html(self):
		datainfo = """
		<p2><b>Original data size:</b></p2><br>
		<p2>Train Normal: {}</p2><br>
		<p2>Train COVID: {}</p2><br>
		<p2>Train SUM: {}</p2><br>
		<p2>Test Normal: {}</p2><br>
		<p2>Test COVID: {}</p2><br>
		<p2>Test SUM: {}</p2><br>

		<p2><b>Augmented data size:</b></p2><br>
		<p2>Train Normal: {}</p2><br>
		<p2>Train COVID: {}</p2><br>
		<p2>Train SUM: {}</p2><br>
		<p2>Test Normal: {}</p2><br>
		<p2>Test COVID: {}</p2><br>
		<p2>Test SUM: {}</p2><br>	
		
		""".format(
			self.data_info[0][0][0],
			self.data_info[0][0][1],
			self.data_info[0][0][0] + self.data_info[0][0][1],
			self.data_info[0][1][0],
			self.data_info[0][1][1],
			self.data_info[0][1][0] + self.data_info[0][1][1],
			self.data_info[1][0][0],
			self.data_info[1][0][1],
			self.data_info[1][0][0] + self.data_info[1][0][1],
			self.data_info[1][1][0],
			self.data_info[1][1][1],
			self.data_info[1][1][0] + self.data_info[1][1][1],
		)

		html = str(self.dataframe.to_html()).replace("&lt;","<").replace("&gt;",">")
		text_file = open(r'resources\results\results.html', "w", encoding="utf-8")
		text_file.write(datainfo)
		text_file.write(html)
		text_file.close()
