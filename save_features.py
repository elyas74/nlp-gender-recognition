
import numpy as np
from glob import glob
import json

import keras

from feature_extraction import feature_extraction

train_male_dirs = glob('./data/train/male/*.wav')
train_female_dirs = glob('./data/train/female/*.wav')

test_male_dirs = glob('./data/test/male/*.wav')
test_female_dirs = glob('./data/test/female/*.wav')


print('train_male samples => {}'.format(len(train_male_dirs)))
print('train_female samples => {}'.format(len(train_female_dirs)))
print()
print('test_male samples => {}'.format(len(test_male_dirs)))
print('test_female samples => {}'.format(len(test_female_dirs)))


X_train = []
Y_train = []

X_test = []
Y_test = []

sr = 22050
frame_len = 1000
hop_len = 80

for _dir in train_male_dirs:
    print('extracting features from => {}'.format(_dir))

    features = feature_extraction(_dir, sr=sr, mono=True,
                                  frame_len=frame_len, hop_len=hop_len)

    # print("features shape => ", features.shape)
    # print("features => ", features)

    # exit()

    X_train.append(features)
    Y_train.append(1)

    # break

for _dir in train_female_dirs:
    print('extracting features from => {}'.format(_dir))

    features = feature_extraction(_dir, sr=sr, mono=True,
                                  frame_len=frame_len, hop_len=hop_len)

    X_train.append(features)
    Y_train.append(0)

    # break


for _dir in test_male_dirs:
    print('extracting features from => {}'.format(_dir))

    features = feature_extraction(_dir, sr=sr, mono=True,
                                  frame_len=frame_len, hop_len=hop_len)

    X_test.append(features)
    Y_test.append(1)

    # break

for _dir in test_female_dirs:
    print('extracting features from => {}'.format(_dir))

    features = feature_extraction(_dir, sr=sr, mono=True,
                                  frame_len=frame_len, hop_len=hop_len)

    X_test.append(features)
    Y_test.append(0)

    # break


X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)


print('X_train.shape => {}'.format(X_train.shape))
print('Y_train.shape => {}'.format(Y_train.shape))

print('X_test.shape => {}'.format(X_test.shape))
print('Y_test.shape => {}'.format(Y_test.shape))


# data = {
#     'X_train': X_train,
#     'Y_train': Y_train,
#     'X_test': X_test,
#     'Y_test': Y_test
# }


np.save('saved_features/X_train.npy', X_train)
np.save('saved_features/Y_train.npy', Y_train)
np.save('saved_features/X_test.npy', X_test)
np.save('saved_features/Y_test.npy', Y_test)

print('all saved')

# with open('features.json', 'w') as outfile:
# json.dump(X_train, outfile)
