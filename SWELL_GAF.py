gaf = GramianAngularField()

x_train_swell = gaf.fit_transform(x_train_swell.to_numpy())
x_test_swell = gaf.fit_transform(x_test_swell.to_numpy())

print(x_train_swell.shape)
print(x_test_swell.shape)

plt.imshow(x_train_swell[24])

x_train_swell = x_train_swell.reshape(x_train_swell.shape[0],34,34,1)
x_test_swell = x_test_swell.reshape(x_test_swell.shape[0],34,34,1)

x_train_swell = x_train_swell.astype('float32')
x_train_swell /= np.amax(x_train_swell)

x_test_swell = x_test_swell.astype('float32')
x_test_swell /= np.amax(x_test_swell)

y_train = tf.keras.utils.to_categorical(y_train,num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=3)


print(np.argmax(y_train[12], axis=None, out=None))
plt.imshow(x_train_swell[12])

print(y_train.shape)
print(x_train_swell.shape)

print(x_train_swell.shape)
print(y_train)

history = model_CNN.fit(x_train_swell,y_train,batch_size= 100,epochs = 1,validation_data = (x_test_swell,y_test))

from tensorflow import math
from tensorflow.math import confusion_matrix
from sklearn.metrics import confusion_matrix

train_swell = model_CNN.evaluate(x_train_swell, y_train)
test_swell = model_CNN.evaluate(x_test_swell, y_test)

print("Training accuracy : ", train_swell[1])
print("Testing accuracy : ", test_swell[1])

loss, accuracy, f1_score, precision, recall = model_CNN.evaluate(x_test_swell, y_test, verbose=0)

print("validation loss : ",loss)
print("accuracy : ",accuracy)
print("f1_score : ",f1_score)
print("precision : ",precision)
print("recall : ",recall)

# plt.plot(history.history['val_accuracy'])

y_pred = model_CNN.predict(x_test_swell)
# cm = confusion_matrix(y_test,y_pred)

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
matrix.diagonal()/matrix.sum(axis=1)

print(matrix)

import seaborn as sns
# sns.heatmap(matrix/np.sum(matrix), annot=True, 
#             fmt='.2%', cmap='Blues')

sns.heatmap(matrix/np.sum(matrix),cmap='Blues')

from sklearn.metrics import classification_report


y_test_swell = y_test.argmax(axis=1)
# y_pred = model_CNN.predict(y_test)

print(classification_report(y_test_swell, y_pred.argmax(axis=1)))
