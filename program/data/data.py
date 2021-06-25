"""
Moduł zawierający obsługę danych źródłowych.
"""
import os
import cv2
import pickle as pkl

from tensorflow.keras import layers
from tensorflow.python.ops.image_ops_impl import flip_up_down
from tqdm import tqdm
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow import keras
from tensorflow.keras.models import Sequential
from random import randint,uniform
import numpy as np


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
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def import_data(self):
        """
        Odczyt prosto z plików png.
        """
        for file in tqdm(os.listdir(r'resources\raw_data\normal')):
            img_path = os.path.join(r'resources\raw_data\normal', file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.labels.append(0)
        for file in tqdm(os.listdir(r'resources\raw_data\pneumonia')):
            img_path = os.path.join(r'resources\raw_data\pneumonia', file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.labels.append(1)
        for file in tqdm(os.listdir(r'resources\raw_data\covid')):
            img_path = os.path.join(r'resources\raw_data\covid', file)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.labels.append(2)

    def dump_data(self):
        """
        Zapis skompresowanych danych.
        """
        if self.images != [] and self.labels != []:
            file_images = open('resources\pickled_data\images.pickled', 'wb')
            file_labels = open('resources\pickled_data\labels.pickled', 'wb')
            pkl.dump(self.images, file_images, protocol=pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.labels, file_labels, protocol=pkl.HIGHEST_PROTOCOL)
            file_images.close()
            file_labels.close()

    def load_data(self):
        """
        Odczyt skompresowanych danych.
        """
        file_images = open('resources\pickled_data\images.pickled', 'rb')
        file_labels = open('resources\pickled_data\labels.pickled', 'rb')
        self.images = pkl.load(file_images)
        self.labels = pkl.load(file_labels)
        file_images.close()
        file_labels.close()

    def resize_data(self, height, width):
        """
        Przeskalowanie danych.
        """
        for i in range(len(self.images)):
            self.images[i] = cv2.resize(self.images[i], (height, width))

    def augment_data(self, augmentation_factor=0.25):
        """
        Rozszerzenie danych.
        """
        N = len(self.images)
        n = int(augmentation_factor * N)
        for _ in range(n):
            i = randint(0, N)
            label = self.labels[i]
            image = self.images[i]
            for _ in range(4):
                self.images.append(tf.image.central_crop(image, central_fraction=uniform(0.25, 0.75)))
                self.labels.append(label)
                self.images.append(tf.image.adjust_brightness(image, uniform(0.25, 0.75)))
                self.labels.append(label)
                self.images.append(tf.image.adjust_saturation(image, uniform(0.25, 0.75)))
                self.labels.append(label)
                image = tf.image.rot90(image)
                self.images.append(image)
                self.labels.append(label)

    def train_test_split(self, train_fraction=0.75, random_state=0):
        """
        Podział zbioru danych na podzbiory: uczący i testowy.
        """
        N = len(self.images)
        np.random.seed(random_state)
        indexes = np.random.permutation(N).astype(int)
        split = round(train_fraction * N)
        X = self.images[indexes]
        y = self.labels[indexes]
        self.X_train = X[:split]
        self.y_train = y[:split]
        self.X_test = X[split:]
        self.y_test = y[split:]

    def discretize(X, B, X_train_ref=None):
        if X_train_ref is None:
            X_train_ref = X
        mins = np.min(X_train_ref, axis=0)
        maxes = np.max(X_train_ref, axis=0)
        X = np.floor(((X - mins) / (maxes - mins)) * B).astype("int32")
        X = np.clip(X, 0, B - 1)
        return X
