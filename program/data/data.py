"""
Moduł zawierający obsługę danych źródłowych.
"""
import os
import random

import cv2
from numpy.matlib import rand
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from shutil import copy, rmtree
import logging


class Data(object):
    """
    Dane źródłowe.
    """

    def __init__(self):
        """
        Konstruktor.
        """
        self.class_names = ['normal', 'covid']

        # [[train-normal, train-covid],[test-normal, test-covid]]
        self.data_size = [[0, 0], [0, 0]]
        self.original_size = [[0, 0], [0, 0]]

        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.dataset_train = None
        self.dataset_test = None

    def __prepare(self, normal_size, covid_size, split_factor):
        normal_counter = 0
        covid_counter = 0
        os.mkdir(r'resources\tmp')
        os.mkdir(r'resources\tmp\test')
        os.mkdir(r'resources\tmp\train')
        os.mkdir(r'resources\tmp\test\normal')
        os.mkdir(r'resources\tmp\train\normal')
        os.mkdir(r'resources\tmp\test\covid')
        os.mkdir(r'resources\tmp\train\covid')

        for file in tqdm(os.listdir(r'resources\raw_data\normal')):
            if normal_counter < normal_size:
                src_path = os.path.join(r'resources\raw_data\normal', file)
                if normal_counter < int(normal_size * split_factor):
                    dest_path = os.path.join(r'resources\tmp\test\normal', file)
                    copy(src_path, dest_path)
                    self.original_size[1][0] += 1
                else:
                    dest_path = os.path.join(r'resources\tmp\train\normal', file)
                    copy(src_path, dest_path)
                    self.original_size[0][0] += 1
            normal_counter += 1
        for file in tqdm(os.listdir(r'resources\raw_data\covid')):
            if covid_counter < covid_size:
                src_path = os.path.join(r'resources\raw_data\covid', file)
                if covid_counter < int(covid_size * split_factor):
                    dest_path = os.path.join(r'resources\tmp\test\covid', file)
                    copy(src_path, dest_path)
                    self.original_size[1][1] += 1
                else:
                    dest_path = os.path.join(r'resources\tmp\train\covid', file)
                    copy(src_path, dest_path)
                    self.original_size[0][1] += 1
            covid_counter += 1
        logging.info("Data prepared")

    def __augment(self, augmentation_factor, augmentation_count_factor):
        rmtree(r'resources\data')
        os.mkdir(r'resources\data')
        os.mkdir(r'resources\data\test')
        os.mkdir(r'resources\data\train')
        os.mkdir(r'resources\data\test\normal')
        os.mkdir(r'resources\data\train\normal')
        os.mkdir(r'resources\data\test\covid')
        os.mkdir(r'resources\data\train\covid')

        normal_original_size = len(os.listdir(r'resources\tmp\train\normal'))
        normal_counter = 0
        for file in tqdm(os.listdir(r'resources\tmp\train\normal')):
            if normal_counter < augmentation_factor*normal_original_size:
                source = cv2.imread(os.path.join(r'resources\tmp\train\normal', file))
                for i in range(augmentation_count_factor):
                    opt = random.randrange(0, 8)
                    if opt == 0:
                        image = cv2.GaussianBlur(source, (5, 5), 0)
                    elif opt == 1:
                        image = cv2.flip(source, 1)
                    elif opt == 2:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(0.25, 0.5))
                    elif opt == 3:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(0.5, 0.75))
                    elif opt == 4:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(0.75, 1.0))
                    elif opt == 5:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(1.0, 1.25))
                    elif opt == 6:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(1.25, 1.5))
                    elif opt == 7:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(1.5, 1.75))
                    cv2.imwrite(os.path.join(r'resources\data\train\normal', file) + "_augmented_"+str(i+1)+".png", image)
            normal_counter += 1

        covid_original_size = len(os.listdir(r'resources\tmp\train\covid'))
        covid_counter = 0
        for file in tqdm(os.listdir(r'resources\tmp\train\covid')):
            if covid_counter < augmentation_factor*covid_original_size:
                source = cv2.imread(os.path.join(r'resources\tmp\train\covid', file))
                for i in range(augmentation_count_factor):
                    opt = random.randrange(0, 8)
                    if opt == 0:
                        image = cv2.GaussianBlur(source, (5, 5), 0)
                    elif opt == 1:
                        image = cv2.flip(source, 1)
                    elif opt == 2:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(0.25, 0.5))
                    elif opt == 3:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(0.5, 0.75))
                    elif opt == 4:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(0.75, 1.0))
                    elif opt == 5:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(1.0, 1.25))
                    elif opt == 6:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(1.25, 1.5))
                    elif opt == 7:
                        image = cv2.convertScaleAbs(source, alpha=random.uniform(1.5, 1.75))
                    cv2.imwrite(os.path.join(r'resources\data\train\covid', file) + "_augmented_"+str(i+1)+".png", image)
            covid_counter += 1
        logging.info("Data augmented")

    def __copy(self):
        for file in tqdm(os.listdir(r'resources\tmp\train\normal')):
            src_path = os.path.join(r'resources\tmp\train\normal', file)
            dest_path = os.path.join(r'resources\data\train\normal', file)
            copy(src_path, dest_path)

        for file in tqdm(os.listdir(r'resources\tmp\train\covid')):
            src_path = os.path.join(r'resources\tmp\train\covid', file)
            dest_path = os.path.join(r'resources\data\train\covid', file)
            copy(src_path, dest_path)

        for file in tqdm(os.listdir(r'resources\tmp\test\normal')):
            src_path = os.path.join(r'resources\tmp\test\normal', file)
            dest_path = os.path.join(r'resources\data\test\normal', file)
            copy(src_path, dest_path)

        for file in tqdm(os.listdir(r'resources\tmp\test\covid')):
            src_path = os.path.join(r'resources\tmp\test\covid', file)
            dest_path = os.path.join(r'resources\data\test\covid', file)
            copy(src_path, dest_path)

        rmtree(r'resources\tmp')
        logging.info("Data copied")

    def __read(self):
        for file in tqdm(os.listdir(r'resources\data\train\normal')):
            path = os.path.join(r'resources\data\train\normal', file)
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.X_train.append(np.asarray(image).flatten())
            self.y_train.append(0)
            self.data_size[0][0] += 1

        for file in tqdm(os.listdir(r'resources\data\train\covid')):
            path = os.path.join(r'resources\data\train\covid', file)
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.X_train.append(np.asarray(image).flatten())
            self.y_train.append(1)
            self.data_size[0][1] += 1

        for file in tqdm(os.listdir(r'resources\data\test\normal')):
            path = os.path.join(r'resources\data\test\normal', file)
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.X_test.append(np.asarray(image).flatten())
            self.y_test.append(0)
            self.data_size[1][0] += 1

        for file in tqdm(os.listdir(r'resources\data\test\covid')):
            path = os.path.join(r'resources\data\test\covid', file)
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.X_test.append(np.asarray(image).flatten())
            self.y_test.append(1)
            self.data_size[1][1] += 1

        logging.info("Data prepared for algorithms")

    def __preprocess(self, batch_size):
        self.dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
            r'resources\data\train',
            labels="inferred",
            label_mode="categorical",
            class_names=self.class_names,
            batch_size=batch_size,
            image_size=(224, 224)
        )
        self.dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
            r'resources\data\test',
            labels="inferred",
            label_mode="categorical",
            class_names=self.class_names,
            batch_size=batch_size,
            image_size=(224, 224)
        )
        logging.info("Data prepared for neural networks")

    def __pca_one(self, X, num_components):
        X_meaned = X - np.mean(X, axis=0)
        cov_mat = np.cov(X_meaned, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
        X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
        return X_reduced

    def __pca(self):
        self.X_train = self.__pca_one(np.array(self.X_train), len(self.X_train[0]))
        self.X_test = self.__pca_one(np.array(self.X_test), len(self.X_test[0]))
        logging.info("Data processed by PCA")

    def load(self, normal_size, covid_size, batch_size, split_factor, augmentation_factor, augmentation_count_factor):
        self.__prepare(normal_size,covid_size,split_factor)
        self.__augment(augmentation_factor, augmentation_count_factor)
        self.__copy()
        self.__read()
        self.__preprocess(batch_size)
        # self.__pca()
