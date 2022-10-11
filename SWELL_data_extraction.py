import os
import pickle
import numpy as np
import neurokit as nk
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

os.getcwd()

__file__ = '/home/sayandeep/SayanD/Dataset/Swell_dataset/archive/hrv dataset/data'

def root_directory():
    current_path = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(current_path, os.pardir))
def data_directory():
    return os.path.join(root_directory(), "data")

def load_train_set():
    in_file = os.path.join(data_directory(), "final",  "train.csv")
    return pd.read_csv(in_file)
def load_test_set():
    in_file = os.path.join(data_directory(), "final",  "test.csv")
    return pd.read_csv(in_file)
  
train = load_train_set()
test = load_test_set()
target = 'condition'
# display(train)
hrv_features = list(train)
hrv_features = [x for x in hrv_features if x not in [target]]

l = {'no stress' : 0, 'interruption' : 1, 'time pressure':2}

X_train= train[hrv_features]
X_train = X_train.drop(columns='datasetId')
y_train= train[target]

# X_train = X_train.tail(20000)
# y_train = y_train.tail(20000)

X_test = test[hrv_features]
X_test = X_test.drop(columns='datasetId')
y_test = test[target]

for i in range(len(y_train)):
    y_train[i] = l[y_train[i]]
    
for i in range(len(y_test)):
    y_test[i] = l[y_test[i]]

y_train = y_train.astype(int)
y_test = y_test.astype(int)
display(X_train)
display(y_train)


