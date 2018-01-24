
from __future__ import print_function

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
epochs = 10


X_train = np.loadtxt('saved_features/X_train.txt')
Y_train = np.loadtxt('saved_features/Y_train.txt')

X_test = np.loadtxt('saved_features/X_test.txt')
Y_test = np.loadtxt('saved_features/Y_test.txt')


print('X_train.shape => {}'.format(X_train.shape))
print('Y_train.shape => {}'.format(Y_train.shape))

print('X_test.shape => {}'.format(X_test.shape))
print('Y_test.shape => {}'.format(Y_test.shape))


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))

# model.load_weights('save_mlp_w', by_name=True)
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))


# model.save_weights('save_mlp_w')
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
