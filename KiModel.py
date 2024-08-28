import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import psutil
import time
import csv


print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print(device_lib.list_local_devices())


print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU available:", tf.config.list_physical_devices('GPU'))

pfadData = '../Testdaten/final_df2_cleaned.csv'
data = pd.read_csv(pfadData)# , encoding='ISO-8859-1', delimiter=';', quoting=csv.QUOTE_ALL)
print(data.head())

X = data.drop('Gesamt_MEASURE_FAIL_CODE', axis=1)
Y = data['Gesamt_MEASURE_FAIL_CODE']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=101)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=x_train, y=y_train, epochs=20, validation_data=(x_test, y_test), verbose=1)

pd.DataFrame(model.history.history).plot()

print("Hallo")