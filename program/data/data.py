"""
Moduł zawierający obsługę danych źródłowych.
"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

class Data(object):
	"""
	Dane źródłowe.
	"""
	def __init__(self):
		"""
		Konstruktor.
		"""
		self.class_names = ['normal', 'pneumonia', 'covid']
		self.images = []
		self.labels = []

	def import_data(self):
		for entry in os.scandir(r'resources\raw_data\normal'):
			if entry.path.endswith(".png") and entry.is_file():
				image = plt.imread(entry.path)
				self.images.append(image)
				self.labels.append(0)
		for entry in os.scandir(r'resources\raw_data\pneumonia'):
			if entry.path.endswith(".png") and entry.is_file():
				image = plt.imread(entry.path)
				self.images.append(image)
				self.labels.append(1)
		for entry in os.scandir(r'resources\raw_data\covid'):
			if entry.path.endswith(".png") and entry.is_file():
				image = plt.imread(entry.path)
				self.images.append(image)
				self.labels.append(2)


	def dump_data(self):
		file_images = open('resources\pickled_data\images_pickled', 'wb')
		file_labels = open('resources\pickled_data\labels_pickled', 'wb')
		pkl.dump(self.images, file_images, protocol=2)
		pkl.dump(self.labels, file_labels, protocol=2)
		file_images.close()
		file_labels.close()

