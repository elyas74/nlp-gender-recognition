
import numpy as np
from glob import glob
import json

import keras

from feature_extraction import feature_extraction

male_dirs = glob('./train_data/male/*.wav')
female_dirs = glob('./train_data/female/*.wav')

print('male samples => {}'.format(len(male_dirs)))
print('female samples => {}'.format(len(female_dirs)))

X_train = []
Y_train = []

sr = 22050
frame_len = 1000
hop_len = 50

for _dir in male_dirs:
    print('extractin features from => {}'.format(_dir))

    features = feature_extraction(_dir, sr=sr, mono=True,
                                  frame_len=frame_len, hop_len=hop_len)

    X_train.append(features)
    Y_train.append(1)

for _dir in female_dirs:
    print('extractin features from => {}'.format(_dir))

    features = feature_extraction(_dir, sr=sr, mono=True,
                                  frame_len=frame_len, hop_len=hop_len)

    X_train.append(features)
    Y_train.append(0)


Y_train = keras.utils.to_categorical(Y_train, 2)

X_train = X_test = np.array(X_train)
Y_train = Y_test = np.array(Y_train)


print('X_train.shape => {}'.format(X_train.shape))
print('Y_train.shape => {}'.format(Y_train.shape))

print('X_test.shape => {}'.format(X_test.shape))
print('Y_test.shape => {}'.format(Y_test.shape))


data = {
    'X_train': X_train,
    'Y_train': Y_train,
    'X_test': X_test,
    'Y_test': Y_test
}


np.savetxt('saved_features/X_train.txt', X_train)
np.savetxt('saved_features/Y_train.txt', Y_train)
np.savetxt('saved_features/X_test.txt', X_test)
np.savetxt('saved_features/Y_test.txt', Y_test)

print('all saved')

# with open('features.json', 'w') as outfile:
# json.dump(X_train, outfile)
