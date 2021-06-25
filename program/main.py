import cv2
from program.data import Data
from program.measuring_quality.measuring_quality import MeasuringQuality
from program.measuring_quality.statictics import Statistics
from program.algorithms.linear_classifiers.naive_bayes_classifier import NaiveBayesClassifier
import time
import numpy as np


data = Data()
data.load_data()
data.augment_data(0.1)
data.tts_manual()

# m, n = data.X_train.shape

dnbc = NaiveBayesClassifier(domain_sizes=np.ones(len(data.X_train)).astype("int32"), laplace=True, logarithm=True)
dnbc.fit(data.X_train, data.y_train)
y_test_pred = dnbc.predict(data.X_test)
y_train_pred = dnbc.predict(data.X_train)

mq = MeasuringQuality()
mq.update("dnbc",data.y_test,y_test_pred)
mq.calculate()

s = Statistics()
s.insert(mq)
s.create_statistics()
s.show()
s.export_csv()
s.export_html()
