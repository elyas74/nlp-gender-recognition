
from __future__ import print_function

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, RNN
from keras.optimizers import RMSprop, Adam

batch_size = 128
epochs = 500


X_train = np.load('saved_features/X_train.npy')
Y_train = np.load('saved_features/Y_train.npy')

X_test = np.load('saved_features/X_test.npy')
Y_test = np.load('saved_features/Y_test.npy')


X_train = keras.utils.to_categorical(X_train, 950)
Y_train = keras.utils.to_categorical(Y_train, 2)

X_test = keras.utils.to_categorical(X_test, 950)
Y_test = keras.utils.to_categorical(Y_test, 2)


print('X_train.shape => {}'.format(X_train.shape))
print('Y_train.shape => {}'.format(Y_train.shape))

print('X_test.shape => {}'.format(X_test.shape))
print('Y_test.shape => {}'.format(Y_test.shape))


model = Sequential()

# model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.50))

model.add(keras.layers.LSTM(20, input_shape=(
    X_train.shape[1], X_train.shape[2],), activation='tanh', use_bias=True, dropout=0.28, recurrent_dropout=0.15))

# model.add(Dropout(0.25))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))

model.add(Dense(2, activation='softmax'))


# model.load_weights('train_weights.HDF', by_name=True)
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks=[tbCallBack])


model.save_weights('train_weights.HDF')
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
