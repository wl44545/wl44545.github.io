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

def ai(data):


    num_train,num_validation = data.data_size
    num_classes = len(data.class_names)
    num_iterations = int(num_train/32)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="resources/logs", histogram_freq=1)

    input_shape=(224,224,3)
    train_processed = data.dataset_train
    validation_processed = data.dataset_test

    model = tf.keras.applications.VGG19
    # load the pre-trained model with global average pooling as the last layer and freeze the model weights
    pre_trained_model = model(include_top=False, pooling='avg', input_shape=input_shape)
    pre_trained_model.trainable = False

    # custom modifications on top of pre-trained model
    clf_model = tf.keras.models.Sequential()
    clf_model.add(pre_trained_model)
    clf_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    clf_model.compile(loss='categorical_crossentropy', metrics=[
        TruePositives(),
        TrueNegatives(),
        FalsePositives(),
        FalseNegatives(),
        Accuracy(),
        MeanSquaredError(),
        Precision(),
        AUC()
    ])
    history = clf_model.fit(train_processed, epochs=1, validation_data=validation_processed,
                            steps_per_epoch=num_iterations,callbacks=[tensorboard_callback])

    # Calculate all relevant metrics
    print(history.history['val_accuracy'][-1])

    mq = MeasuringQuality("0", "0", 0, 0)
    mq.calculate_neural_network(history)
    return mq
