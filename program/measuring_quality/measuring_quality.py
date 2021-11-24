"""
Moduł zawierający miary jakości klasyfikacji.
"""
import uuid

from sklearn import metrics
from numpy import interp
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sn


class MeasuringQuality:
	"""
	Miary jakości klasyfikacji.
	"""

	def __init__(self, method, description, train_time, predict_time):
		"""
		Konstruktor.
		"""
		self.method = method
		self.description = description

		self.train_time = train_time
		self.predict_time = predict_time

		self.confusion_matrix = None
		self.confusion_matrix_percentage = None
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

		self.my_score = None

	def __str__(self):
		return "method: {}, description: {},  train_time: {}, predict_time: {},  true_positive: {}, true_negative: {}, false_positive: {}, false_negative: {}, sensitivity: {},  specificity: {}, precision: {},  accuracy: {},  error: {}, f1: {}".format(self.method, self.description, self.train_time,self.predict_time, self.true_positive,self.true_negative,self.false_positive, self.false_negative,self.sensitivity, self.specificity,self.precision, self.accuracy, self.error, self.f1, self.my_score)

	def calculate_algorithm(self, y_true, y_pred, y_score):
		"""
		Metoda obliczająca statystyki.
		"""

		self.confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
		[[self.true_negative, self.false_positive], [self.false_negative, self.true_positive]] = self.confusion_matrix

		if len(y_score.shape) == 1:
			tmp = y_score[:]
		elif len(y_score.shape) == 2:
			tmp = y_score[:, 1]

		self.roc_curve = metrics.roc_curve(y_true, tmp)
		self.auc = metrics.roc_auc_score(y_true, tmp)

		self.accuracy = metrics.accuracy_score(y_true, y_pred)
		self.error = 1 - self.accuracy
		self.precision = metrics.precision_score(y_true, y_pred)
		self.specificity = self.true_negative / (self.true_negative + self.false_positive)
		self.sensitivity = self.true_positive / (self.true_positive + self.false_negative)
		self.f1 = metrics.f1_score(y_true, y_pred)

		self.my_score = 0.1*self.accuracy+0.1*self.precision+0.1*self.specificity+0.1*self.sensitivity+0.5*self.f1+0.1*(1/self.error)

		self.__draw_roc(self.roc_curve)
		self.__draw_confusion(self.confusion_matrix)

	def calculate_neural_network(self, y_true, y_pred):
		"""
		Metoda obliczająca statystyki.
		"""
		self.confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
		[[self.true_negative, self.false_positive], [self.false_negative, self.true_positive]] = self.confusion_matrix

		self.accuracy = metrics.accuracy_score(y_true, y_pred)
		self.error = 1 - self.accuracy
		self.precision = metrics.precision_score(y_true, y_pred)
		self.specificity = self.true_negative / (self.true_negative + self.false_positive)
		self.sensitivity = self.true_positive / (self.true_positive + self.false_negative)
		self.f1 = metrics.f1_score(y_true, y_pred)

		self.my_score = 0.1*self.accuracy+0.1*self.precision+0.1*self.specificity+0.1*self.sensitivity+0.5*self.f1+0.1*(1/self.error)

		self.__draw_confusion(self.confusion_matrix)

		self.roc_curve = metrics.roc_curve(y_true, y_pred)
		self.__draw_roc(self.roc_curve)


	def __draw_confusion(self, confusion):
		plt.figure()
		df = pd.DataFrame(confusion, columns=["Normal", "COVID-19"],index=["Normal", "COVID-19"])
		sn.heatmap(df, annot=True, cmap="YlGnBu")
		plt.xlabel("Predicted")
		plt.ylabel("Actual")
		filename = str(uuid.uuid1())+".png"
		plt.savefig("resources/results/images/"+filename)
		self.confusion_matrix = "<a href=\"./images/" + filename + "\"><img src=\"./images/" + filename + "\" width=200 height=200 /></a>"
		plt.close()
		plt.figure()
		df = pd.DataFrame(confusion/np.sum(confusion), columns=["Normal", "COVID-19"],index=["Normal", "COVID-19"])
		sn.heatmap(df, annot=True, cmap="YlGnBu")
		plt.xlabel("Predicted")
		plt.ylabel("Actual")
		filename = str(uuid.uuid1())+".png"
		plt.savefig("resources/results/images/"+filename)
		self.confusion_matrix_percentage = "<a href=\"./images/" + filename + "\"><img src=\"./images/" + filename + "\" width=200 height=200 /></a>"
		plt.close()

	def __draw_roc(self, roc):
		plt.figure()
		plt.plot(roc[0], roc[1], color='darkorange', label='ROC curve')
		plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc="lower right")
		filename = str(uuid.uuid1())+".png"
		plt.savefig("resources/results/images/"+filename)
		self.roc_curve = "<a href=\"./images/" + filename + "\"><img src=\"./images/" + filename + "\" width=200 height=200 /></a>"
		plt.close()
