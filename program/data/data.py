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
            file_images = open('resources\pickled_data\images_pickled', 'wb')
            file_labels = open('resources\pickled_data\labels_pickled', 'wb')
            pkl.dump(self.images, file_images, protocol=pkl.HIGHEST_PROTOCOL)
            pkl.dump(self.labels, file_labels, protocol=pkl.HIGHEST_PROTOCOL)
            file_images.close()
            file_labels.close()

    def load_data(self):
        """
        Odczyt skompresowanych danych.
        """
        file_images = open('resources\pickled_data\images_pickled', 'rb')
        file_labels = open('resources\pickled_data\labels_pickled', 'rb')
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

    def augment_data(self, augmentation_factor):
        """
        Rozszerzenie danych.
        """
        N = len(self.images)
        n = augmentation_factor * N
        for _ in range(n):
            i = randint(0, N)
            label = self.labels[i]
            image = self.images[i]
            opt = randint(1, 6)

            if opt == 1:
                new_image = tf.image.flip_left_right(image)
                self.images.append(new_image)
                self.labels.append(label)

            elif opt == 2:
                new_image = tf.image.flip_up_down(image)
                self.images.append(new_image)
                self.labels.append(label)

            elif opt == 3:
                new_image = tf.image.adjust_saturation(image, uniform(0, 10))
                self.images.append(new_image)
                self.labels.append(label)

            elif opt == 4:
                new_image = tf.image.adjust_brightness(image, uniform(0, 1))
                self.images.append(new_image)
                self.labels.append(label)

            elif opt == 5:
                new_image = tf.image.central_crop(image, central_fraction=uniform(0, 1))
                self.images.append(new_image)
                self.labels.append(label)

            elif opt == 6:
                new_image = tf.image.rot90(image)
                self.images.append(new_image)
                self.labels.append(label)