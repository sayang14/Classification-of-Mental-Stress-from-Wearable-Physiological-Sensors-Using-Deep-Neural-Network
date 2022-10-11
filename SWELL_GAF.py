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
