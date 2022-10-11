import matplotlib.image as mpimg
from keras import Model
from keras.layers import Layer
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN,Flatten,GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.metrics import mean_squared_error
from keras.layers import LSTM,Activation,BatchNormalization,MaxPool2D,Multiply,ReLU
from keras.optimizers import SGD
from keras.layers import Embedding,Conv2D,Dropout
from keras import layers

# img = mpimg.imread('/home/sayandeep/SayanD/Dataset/WESAD/WESAD_GAF/TRAIN/GAF_images 9.png')
# print(img)
# precision, f1_score calculation taken from https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model_CNN():
    model = Sequential() 
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(8,8,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Flatten())
    model.add(Dense(6))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
    return model

model_CNN = model_CNN()
model_CNN.summary()
