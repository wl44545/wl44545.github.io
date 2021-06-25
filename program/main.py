import cv2
from program.data import Data
from program.measuring_quality.measuring_quality import MeasuringQuality
from program.measuring_quality.statictics import Statistics
from program.algorithms.linear_classifiers.naive_bayes_classifier import NaiveBayesClassifier
import time
import numpy as np


data = Data()
data.load_data()
data.train_test_split()

m, n = data.X_train.shape

dnbc = NaiveBayesClassifier(domain_sizes=np.ones(len(data.X_train)).astype("int32"), laplace=True, logarithm=True)
dnbc.fit(data.X_train, data.y_train)
y_test_pred = dnbc.predict(data.X_test)
y_train_pred = dnbc.predict(data.X_train)

mq = MeasuringQuality("dnbc","desc", 0.0,0.0,data.y_test,y_test_pred)

s = Statistics()
s.insert(mq)
s.create_statistics()
s.show()
s.export_csv()
s.export_html()
