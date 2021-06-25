import cv2
from program.data import Data
from program.measuring_quality.measuring_quality import MeasuringQuality
from program.measuring_quality.statictics import Statistics
from program.algorithms.linear_classifiers.naive_bayes_classifier import NaiveBayesClassifier
import time
import numpy as np

#
# data = Data()
# data.load_data()
# data.augment_data(0.1)
# data.train_test_split()
#
# m, n = data.X_train.shape
#
# dnbc = NaiveBayesClassifier(domain_sizes=np.ones(n).astype("int32"), laplace=True, logarithm=True)
# dnbc.fit(data.X_train, data.y_train)
# y_test_pred = dnbc.predict(data.X_test)
# y_train_pred = dnbc.predict(data.X_train)
#
#
# print(y_test_pred)


mq = MeasuringQuality()
mq.update("testowe",[0,0,0,1,1,1],[0,0,1,1,1,0])
mq.calculate()

s = Statistics()
s.insert(mq)
s.insert(mq)
s.insert(mq)
s.create_statistics()
s.show()
s.export_csv()
s.export_html()
