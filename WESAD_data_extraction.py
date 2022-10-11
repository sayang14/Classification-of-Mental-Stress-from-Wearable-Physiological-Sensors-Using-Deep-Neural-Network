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

class read_data:
    def __init__(self,path,subject):
        self.keys = ['label','subject','signal']
        self.signal_keys = ['wrist','chest']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensor_keys = ['ACC','BVP','EDA','Temp']
        os.chdir(path)
        os.chdir(subject)
        with open(subject + '.pkl','rb') as file:
            data = pickle.load(file,encoding='latin1')
        self.data = data
        
    def get_labels(self):
        return self.data[self.keys[0]]
    
    def get_chest_data(self):
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data
      

data_path = '/home/sayandeep/SayanD/Dataset/WESAD/WESAD'
subject = 'S17'

data_target = {}
data_target[subject] = read_data(data_path,subject)

# print(data_target[subject].data)

# print(data_target[subject].data['label'])

chest_data_dict = data_target[subject].get_chest_data()
chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
print(chest_dict_length)

# data for "baseline" measurements

count = len(data_target[subject].data['label'])
baseline=1
baseline_indices = np.nonzero(data_target[subject].data['label']==baseline)[0]
print(baseline_indices, 
      '\nlen=', len(baseline_indices), 
      '\n % of total data=', len(baseline_indices)/count)



range_to_plot=range(1000)
baseline_to_plot=baseline_indices[range_to_plot]
# print('baseline indices=', baseline_to_plot)

# print('labels array', (data_target[subject]['label']))

print(chest_data_dict)

acc_x = chest_data_dict['ACC'][0:,0]#[baseline_to_plot]
acc_y = chest_data_dict['ACC'][0:,1]#[baseline_to_plot]
acc_z = chest_data_dict['ACC'][0:,2]#[baseline_to_plot]
emg_x = chest_data_dict['EMG'][:,0]#[baseline_to_plot]
eda_x = chest_data_dict['EDA'][:,0]#[baseline_to_plot]
ecg_x = chest_data_dict['ECG'][:,0]#[baseline_to_plot]
temp_x = chest_data_dict['Temp'][:,0]#[baseline_to_plot]
resp_x = chest_data_dict['Resp'][:,0]#[baseline_to_plot]
w_labels = data_target[subject].data['label']

labels_chest = [key for key,value in chest_data_dict.items()]
print(labels_chest)
print(w_labels)



