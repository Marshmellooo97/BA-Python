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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import psutil
import time
import csv

print("Hallo")
pfadData = '../Testdaten/KiData.csv'
data = pd.read_csv(pfadData, encoding='ISO-8859-1', delimiter=';', quoting=csv.QUOTE_ALL)

data = data.drop(['UPPER_LIMIT', 'LOWER_LIMIT', 'NOMINAL', 'TOLERANCE', 'PART_NUMBER', 'STATION_NUMBER','WORKORDER_NUMBER','STATION_DESC','WORKORDER_DESC'], axis=1)
print(data.nunique())

id_counts = data['BOOKING_ID'].value_counts()

print(data.head())
print("Hallo")
