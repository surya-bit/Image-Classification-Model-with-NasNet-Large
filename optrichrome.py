# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 22:15:29 2023

@author: HP}
"""

import sys
import os
from os import path
import argparse

# Time
import time
import datetime

# Numerical Data
import random
from random import shuffle
import numpy as np
import pandas as pd
from collections import Counter

# Tools
import shutil
from glob import glob
from tqdm import tqdm
import itertools
import gc
import json

# NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup

# Preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight as cw
from sklearn.utils import shuffle

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Machine Learning Models
from sklearn import svm
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

# Machine Learning Evaluation
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# Deep Learning - Keras -  Preprocessing
from keras.preprocessing.image import ImageDataGenerator

# Deep Learning - Keras - Model
import keras
from keras import models
from keras.models import Model
from keras.models import Sequential

# Deep Learning - Keras - Layers
from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D

# Deep Learning - Keras - Pretrained Model
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge

from keras.applications.nasnet import preprocess_input

# Deep Learning - Keras - Evauation
from keras import optimizers
from keras.optimizers import Adam, SGD , RMSprop
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy

# Deep Learning - Keras - Visualisation
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
# from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

# Deep Learning - TensorFlow
import tensorflow as tf

# Graph/ Visualization
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix

# Image
import cv2
from PIL import Image
from IPython.display import display
from wordcloud import WordCloud, STOPWORDS

# np.random.seed(42)

%matplotlib inline

input_shape=(144,144,3)

# Function built to augument and create training , testing and validation data.

def get_data(batch_size=32, target_size=(144, 144), class_mode="categorical", training_dir=training_dir, testing_dir=testing_dir):
    print("Generating data following preprocessing...\n")

    rescale = 1.0/255

    train_batch_size = batch_size
    test_batch_size = batch_size

    train_shuffle = True
    val_shuffle = True
    test_shuffle = False


    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=45,
        shear_range=16,
        rescale=rescale)

    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=target_size,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True,
        seed=42)

    test_datagen = ImageDataGenerator(rescale=rescale)

    test_generator = test_datagen.flow_from_directory(
        testing_dir,
        target_size=target_size,
        class_mode=class_mode,
        batch_size=1024,
        shuffle=False,
        seed=42)

    steps_per_epoch = len(train_generator)

    print("\nData batches generated.\n")


    return train_generator, test_generator, steps_per_epoch



# Custom function which can be called to instantiate any CNN models using transfer learning

def get_model(model_name, input_shape=(144, 144, 3), num_class=4):
    inputs = Input(input_shape)

    if model_name == "Xception":
        base_model = Xception(include_top=False, input_shape=input_shape)
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    elif model_name == "DenseNet201":
        base_model = DenseNet201(include_top=False, input_shape=input_shape)
    elif model_name == "NASNetMobile":
        base_model = NASNetMobile(include_top=False, input_shape=input_shape)
    elif model_name == "NASNetLarge":
        base_model = NASNetLarge(include_top=False, input_shape=input_shape)

    x = base_model(inputs)

    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)

    out = Concatenate(axis=-1)([out1, out2, out3])

    out = Dropout(0.5)(out)
    out = BatchNormalization()(out)

    if num_class>1:
        out = Dense(num_class, activation="softmax", name="3_")(out)
    else:
        out = Dense(1, activation="sigmoid", name="3_")(out)

    model = Model(inputs, out)


    model.summary()

    return model

# Using NasNetLarge Model

print(" NASNetLarge Model")

# input_shape = (96, 96, 3)
input_shape = (144, 144, 3)

num_class = 4

model = get_model(model_name="NASNetLarge", input_shape=input_shape, num_class=num_class)

print("Starting...\n")

start_time = time.time()
print(date_time(1))



print("\n\nCompliling Model ...\n")
learning_rate = 0.0001
optimizer = Adam(learning_rate)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# steps_per_epoch = 180
# validation_steps = 40

verbose = 1
epochs = 10

print("Trainning Model ...\n")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=verbose)

elapsed_time = time.time() - start_time
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
model.save('nasnet_dnnv2.h5')

# Achieved an training and testing accuracy of 85% 

