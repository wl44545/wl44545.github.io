import numpy as np

from measuring_quality import MeasuringQuality
from program.data import Data
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import inspect
from tqdm import tqdm
import os
from tensorflow.keras.metrics import *


class CNN:

	def __init__(self, model, data, batch_size, epochs, log_subdir):

		epochs = 1
		self.data = data
		self.train_steps = int((data.data_size[0][0]+data.data_size[0][1]) / batch_size)
		self.epochs = epochs

		self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir="resources/logs/"+log_subdir, histogram_freq=1)

		self.pre_trained_model = model(include_top=False, pooling='avg', input_shape=(224, 224, 3))
		self.pre_trained_model.trainable = False
		self.trainable_model = tf.keras.models.Sequential()
		self.trainable_model.add(self.pre_trained_model)
		self.trainable_model.add(tf.keras.layers.Dense(len(data.class_names), activation='softmax'))
		self.trainable_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

	def fit(self):
		return self.trainable_model.fit(self.data.dataset_train, epochs=self.epochs, validation_data=self.data.dataset_test,
		                                steps_per_epoch=self.train_steps, callbacks=[self.tensorboard])

	def predict(self):
		y_pred = np.array([])
		y_true = np.array([])
		for x, y in self.data.dataset_test:
			y_pred = np.concatenate([y_pred, np.argmax(self.trainable_model.predict(x), axis=-1)])
			y_true = np.concatenate([y_true, np.argmax(y.numpy(), axis=-1)])
		return y_true, y_pred
