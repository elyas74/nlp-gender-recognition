
# python modules
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy

# my own modules
from framing import framing

# consts
# frame_len = 1000
# hop_len = 20


def feature_extraction(path, sr, mono, frame_len, hop_len):

    # load wav file
    y, sr = librosa.load(path, sr=sr, mono=mono)

    # print('-' * 60)
    # print('y shape =>', y.shape)
    # print('y duration => ', librosa.core.get_duration(y))
    # print('-' * 60)

    # trim it (remove silence from beggining and end)
    yt, index = librosa.effects.trim(y)
    y = np.array(yt)

    # print('trim_y shape =>', y.shape)
    # print('terim_y duration => ', librosa.core.get_duration(y))
    # print('-' * 60)

    frames = framing(y, frame_len, hop_len)

    # print('frames shape =>', frames.shape)
    # print('frames[0] shape =>', frames[0].shape)
    # print('-' * 60)

    features = []

    energys = []
    zcrs = []
    mfccs = []

    for i in range(20):
        mfccs.append([])

    for frame in frames:

        energy = np.sum(np.power(frame, 2))
        energys.append(energy)

        zcr = np.count_nonzero(librosa.core.zero_crossings(frame))
        zcrs.append(zcr)

        mfcc = librosa.feature.mfcc(y=frame, sr=sr)

        for i in range(20):

            # TODO use numpy.average instead of just first one
            mfccs[i].append(mfcc[i][0])

    zcrs = np.array(zcrs)
    energys = np.array(energys)
    mfccs = np.array(mfccs)

    # print('zcrs shape =>', zcrs.shape)
    # print('energys shape =>', energys.shape)
    # print('mfccs shape =>', mfccs.shape)

    def temp(base_features):
        features.append(base_features.min())
        features.append(base_features.max())

        features.append(np.mean(base_features))
        features.append(np.std(base_features))

        features.append(scipy.stats.kurtosis(base_features))
        features.append(scipy.stats.skew(base_features))

    temp(zcrs)
    temp(energys)

    for mfcc in mfccs:
        temp(mfcc)

    # print('features shape =>', np.array(features).shape)
    return np.array(features)


# path = 'train_data/male/Untitled 016.wav'
# a = feature_extraction(path, sr=22050, mono=True, frame_len=1000, hop_len=50)
#
# print('a', a)
