gasf = GramianAngularField(method = 'summation')
gadf = GramianAngularField(method = 'difference')

# df_chest_labels = df_3_l.append(df_1_l.append(df_2_l,ignore_index = True),ignore_index = True)

# df_chest_labels = df_chest_labels.sample(frac = 1, ignore_index = True)

# display(df_chest_labels)

# df_chest = df_chest_labels.drop(columns = 'w_labels',axis = 8)

# display(df_chest_train)
# gaf_matrix = gaf.fit_transform(df_chest.to_numpy())

# labels_wesad=df_chest_labels['w_labels'].to_numpy()
print(labels_wesad)
print(len(labels_wesad))

print(len(labels_wesad))
# print(len(gaf_matrix))

# wesad_df = pd.DataFrame({'Image' : pngs, 'Labels' : labels_wesad})

# display(wesad_df)

# wesad_df = wesad_df.sample(frac = 1,ignore_index = True)

# display(wesad_df)



from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import np_utils

X_train  = df_chest.to_numpy()
X_train = gasf.fit_transform(X_train)
Y_train = labels_wesad

Y_train = Y_train.reshape(Y_train.shape[0],1)
print(Y_train.shape)

X_train = X_train.reshape(X_train.shape[0],8,8,1)
X_train = X_train.astype('float32')
X_train /= np.amax(X_train)
Y_train = tf.keras.utils.to_categorical(Y_train,num_classes=4)
print(Y_train)

x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size = 0.4)

# x_train_gasf = gasf.transform(x_train)
# x_test_gasf = gasf.transform(x_test)
# x_train_gadf = gadf.transform(x_train)

# x_train_gaf=np.concatenate((x_train_gasf,x_train_gadf,np.zeros(x_train_gadf.shape)),axis=-1)

print(x_train.shape)

print(np.argmax(y_train[0], axis=None, out=None))
plt.imshow(x_train[0])

# plt.imshow(x_test[0])
    
# print(x_train)
print(y_train.shape)

print(np.shape(x_train))
print(np.shape(x_test))
print(np.shape(y_train))
print(np.shape(y_test))


# model_CNN can be found on model_CNN.py
history = model_CNN.fit(x_train,y_train,batch_size= 32,epochs = 100,validation_data = (x_test,y_test))
# train_wesad = model_CNN.evaluate(x_train, y_train)
# test_wesad = model_CNN.evaluate(x_test, y_test)

# print("Training accuracy : ", train_wesad[1])
# print("Testing accuracy : ", test_wesad[1])

loss, accuracy, f1_score, precision, recall = model_CNN.evaluate(x_test, y_test, verbose=0)

print("validation loss : ",loss)
print("accuracy : ",accuracy)
print("f1_score : ",f1_score)
print("precision : ",precision)
print("recall : ",recall)

# y_pred = model_CNN.predict(x_test)
# y_pred =(y_pred>0.5)
# print(y_pred)

# plt.plot(history.history['loss'])
plt.plot(history.history['val_accuracy'])
# plt.plot(history.history['val_f1_m'])
plt.plot(history.history['loss'])

import seaborn as sns
from tensorflow import math
from tensorflow.math import confusion_matrix
from sklearn.metrics import confusion_matrix

y_pred = model_CNN.predict(x_test)
# categories = ['Meditation','Baseline','Stress','Amusement']
# matrix = math.confusion_matrix(y_pred.argmax(axis=1), y_test.argmax(axis=1))

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
matrix.diagonal()/matrix.sum(axis=1)

sns.heatmap(matrix/np.sum(matrix),cmap='Blues')

from sklearn.metrics import classification_report


y_test_wesad = y_test.argmax(axis=1)
# y_pred = model_CNN.predict(y_test)

target_names = ["Class {}".format(i) for i in range(0,4)]

print(classification_report(y_test_wesad, y_pred.argmax(axis=1),target_names=target_names))
