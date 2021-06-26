"""
Moduł zawierający obsługę danych źródłowych.
"""
import os
import cv2
import pickle as pkl
from sklearn.model_selection import train_test_split
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
        self.class_names = ['normal','covid']
        self.images = []
        self.labels = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []


    def import_data(self, size=0):
        """
        Odczyt prosto z plików png.
        """
        normal_counter = 0
        covid_counter = 0
        for file in tqdm(os.listdir(r'resources\raw_data\normal')):
            if normal_counter < size or size == 0:
                img_path = os.path.join(r'resources\raw_data\normal', file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.images.append(image)
                self.labels.append(0)
                normal_counter += 1
        for file in tqdm(os.listdir(r'resources\raw_data\covid')):
            if covid_counter < size or size == 0:
                img_path = os.path.join(r'resources\raw_data\covid', file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.images.append(image)
                self.labels.append(1)
                covid_counter += 1

    def dump_data(self, size=0):
        """
        Zapis skompresowanych danych.
        """
        if self.images != [] and self.labels != []:
            file_images = open('resources\pickled_data\images'+str(size)+'.pickled', 'wb')
            file_labels = open('resources\pickled_data\labels'+str(size)+'.pickled', 'wb')
            pkl.dump(self.images, file_images, protocol=pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.labels, file_labels, protocol=pkl.HIGHEST_PROTOCOL)
            file_images.close()
            file_labels.close()

    def load_data(self, size=0):
        """
        Odczyt skompresowanych danych.
        """
        file_images = open('resources\pickled_data\images'+str(size)+'.pickled', 'rb')
        file_labels = open('resources\pickled_data\labels'+str(size)+'.pickled', 'rb')
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

    def split_data(self, test_size=0.25, random_state=0):
        """
        Podział zbioru danych na podzbiory: uczący i testowy.
        """
        tmp = []
        for image in self.images:
            tmp.append(np.asarray(image).flatten())
        self.images = np.array(tmp)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.images, self.labels, test_size=test_size, random_state=random_state)
