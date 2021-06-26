"""
Moduł zawierający miary jakości klasyfikacji.
"""
import uuid

from numpy import interp
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sn


class MeasuringQuality:
	"""
	Miary jakości klasyfikacji.
	"""

	def __init__(self, method, description, train_time, predict_time, y_true, y_pred, y_score):
		"""
		Konstruktor.
		"""
		self.method = method
		self.description = description

		self.train_time = train_time
		self.predict_time = predict_time

		self.y_true = y_true
		self.y_pred = y_pred
		self.y_score = y_score

		self.confusion_matrix = None
		self.roc_curve = None
		self.auc = None

		self.true_positive = None
		self.true_negative = None
		self.false_positive = None
		self.false_negative = None

		self.accuracy = None
		self.error = None
		self.precision = None
		self.specificity = None
		self.sensitivity = None
		self.f1 = None

		self.calculate()

	def calculate(self):
		"""
		Metoda obliczająca statystyki.
		"""
		self.confusion_matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
		self.true_negative, self.false_positive, self.false_negative, self.true_positive = self.confusion_matrix.ravel()

		if len(self.y_score.shape) == 1:
			tmp = self.y_score[:]
		elif len(self.y_score.shape) == 2:
			tmp = self.y_score[:, 1]

		self.roc_curve = metrics.roc_curve(self.y_true, tmp)
		self.auc = metrics.roc_auc_score(self.y_true, tmp)

		# self.roc_curve = metrics.roc_curve(self.y_true, self.y_score[:, 1])
		# self.auc = metrics.roc_auc_score(self.y_true, self.y_score[:, 1])

		self.accuracy = metrics.accuracy_score(self.y_true, self.y_pred)
		self.error = 1 - self.accuracy
		self.precision = metrics.precision_score(self.y_true, self.y_pred)
		self.specificity = self.true_negative / (self.true_negative + self.false_positive)
		self.sensitivity = self.true_positive / (self.true_positive + self.false_negative)
		self.f1 = metrics.f1_score(self.y_true, self.y_pred)

		fn1 = draw_roc(self.roc_curve)
		fn2 = draw_confusion(self.confusion_matrix)

		self.roc_curve = "<a href=\"./images/"+fn1+"\"><img src=\"./images/"+fn1+"\" width=200 height=200 /></a>"
		self.confusion_matrix = "<a href=\"./images/"+fn2+"\"><img src=\"./images/"+fn2+"\" width=200 height=200 /></a>"


def draw_confusion(confusion):
	plt.figure()
	df = pd.DataFrame(confusion, columns=["Normal", "COVID-19"],index=["Normal", "COVID-19"])
	sn.heatmap(df, annot=True, cmap="YlGnBu")
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	filename = str(uuid.uuid1())+".png"
	plt.savefig("resources/results/images/"+filename)
	return filename
	# plt.show()
	# df = pd.DataFrame(confusion/np.sum(confusion), columns=["Normal", "COVID-19"],index=["Normal", "COVID-19"])
	# sn.heatmap(df, annot=True, cmap="YlGnBu")
	# plt.xlabel("Predicted")
	# plt.ylabel("Actual")
	# filename = "./images/"+str(uuid.uuid1)+".png"
	# plt.savefig(filename)
	# return filename


def draw_roc(roc):
	plt.figure()
	plt.plot(roc[0], roc[1], color='darkorange', label='ROC curve')
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	filename = str(uuid.uuid1())+".png"
	plt.savefig("resources/results/images/"+filename)
	return filename
