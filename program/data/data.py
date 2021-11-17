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
from shutil import copy,rmtree


class Data(object):
    """
    Dane źródłowe.
    """
    def __init__(self):
        """
        Konstruktor.
        """
        self.class_names = ['normal', 'covid']
        self.data_size = 0
        self.augmented_size = 0
        self.images = []
        self.labels = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.dataset_train = None
        self.dataset_test = None

    def make_data(self, normal_size=0, covid_size=0, split_factor=0, augmentation_factor=0):
        normal_counter = 0
        covid_counter = 0
        rmtree(r'resources\prepared_data')
        os.mkdir(r'resources\prepared_data')
        os.mkdir(r'resources\prepared_data\test')
        os.mkdir(r'resources\prepared_data\train')
        os.mkdir(r'resources\prepared_data\test\normal')
        os.mkdir(r'resources\prepared_data\train\normal')
        os.mkdir(r'resources\prepared_data\test\covid')
        os.mkdir(r'resources\prepared_data\train\covid')
        for file in tqdm(os.listdir(r'resources\raw_data\normal')):
            if normal_counter < normal_size or normal_size==0:
                src_path = os.path.join(r'resources\raw_data\normal', file)
                if normal_counter < normal_size*split_factor:
                    dest_path = os.path.join(r'resources\prepared_data\test\normal', file)
                    copy(src_path,dest_path)
                    image = cv2.imread(src_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.X_test.append(np.asarray(image).flatten())
                    self.y_test.append(0)
                else:
                    dest_path = os.path.join(r'resources\prepared_data\train\normal', file)
                    copy(src_path,dest_path)
                    image = cv2.imread(src_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.X_train.append(np.asarray(image).flatten())
                    self.y_train.append(0)
            normal_counter += 1
        for file in tqdm(os.listdir(r'resources\raw_data\covid')):
            if covid_counter < covid_size or covid_size==0:
                src_path = os.path.join(r'resources\raw_data\covid', file)
                if covid_counter < covid_size*split_factor:
                    dest_path = os.path.join(r'resources\prepared_data\test\covid', file)
                    copy(src_path,dest_path)
                    image = cv2.imread(src_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.X_test.append(np.asarray(image).flatten())
                    self.y_test.append(1)
                else:
                    dest_path = os.path.join(r'resources\prepared_data\train\covid', file)
                    copy(src_path,dest_path)
                    image = cv2.imread(src_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.X_train.append(np.asarray(image).flatten())
                    self.y_train.append(1)
            covid_counter += 1
        self.data_size = (len(self.X_train), len(self.X_test))
        self.augmented_size = self.data_size

    def preprocess_data(self, batch_size):

        self.dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
            r'resources\prepared_data\train',
            labels="inferred",
            label_mode="categorical",
            class_names=self.class_names,
            batch_size=batch_size,
            image_size=(224, 224),
            shuffle=True,
            seed=123
        )
        self.dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
            r'resources\prepared_data\test',
            labels="inferred",
            label_mode="categorical",
            class_names=self.class_names,
            batch_size=batch_size,
            image_size=(224, 224),
            shuffle=True,
            seed=123
        )

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
        self.data_size = len(self.images)
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
                # self.images.append(cv2.GaussianBlur(image,(5,5),0))
                # self.labels.append(label)
                self.images.append(cv2.flip(image, 0))
                self.labels.append(label)
                self.images.append(cv2.flip(image, 1))
                self.labels.append(label)
                self.images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
                self.labels.append(label)
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        self.augmented_size = len(self.images)

    def split_data(self, test_size=0.25, random_state=0):
        """
        Podział zbioru danych na podzbiory: uczący i testowy.
        """
        tmp = []
        for image in self.images:
            tmp.append(np.asarray(image).flatten())
        self.images = np.array(tmp)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.images, self.labels, test_size=test_size, random_state=random_state)
