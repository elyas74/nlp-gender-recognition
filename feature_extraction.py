
# python modules
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy

# my own modules
from framing import framing

n_mfcc = 40


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
    estimate_tunings = []
    pitch_tunings = []
    mfccs = []
    # chroma_stfts = []

    # stfts = []

    for i in range(n_mfcc):
        mfccs.append([])

    # for i in range(12):
    #     chroma_stfts.append([])

    for frame in frames:

        energy = np.sum(np.power(frame, 2))
        energys.append(energy)

        zcr = np.count_nonzero(librosa.core.zero_crossings(frame))
        zcrs.append(zcr)

        estimate_tuning = librosa.estimate_tuning(y=frame, sr=sr)
        estimate_tunings.append(estimate_tuning)

        pitch_tuning = librosa.core.pitch_tuning(frame)
        pitch_tunings.append(pitch_tuning)

        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)

        for i in range(n_mfcc):
            mfccs[i].append(np.average(mfcc[i]))

        # chroma_stft = librosa.feature.chroma_stft(y=frame, sr=sr)

        # for i in range(12):
            # chroma_stfts[i].append(np.average(chroma_stft[i]))

        # print(chroma_stfts)
        # exit()

    zcrs = np.array(zcrs)
    energys = np.array(energys)
    mfccs = np.array(mfccs)

    # print('zcrs shape =>', zcrs.shape)
    # print('energys shape =>', energys.shape)
    # print('mfccs shape =>', mfccs.shape)

    def temp(base_features):

        base_features = np.array(base_features)

        features.append(base_features.min())
        features.append(base_features.max())

        # print(base_features.min())
        # print(base_features.max())
        # print(np.mean(base_features))
        # print(np.std(base_features))
        # print(scipy.stats.kurtosis(base_features))
        # print(scipy.stats.skew(base_features))

        features.append(np.mean(base_features))
        features.append(np.std(base_features))

        features.append(scipy.stats.kurtosis(base_features))
        features.append(scipy.stats.skew(base_features))

    temp(zcrs)
    temp(energys)
    temp(estimate_tunings)
    temp(pitch_tunings)

    for mfcc in mfccs:
        temp(mfcc)

    # for chroma_stft in chroma_stfts:
    #     temp(chroma_stft)

    # print('features shape =>', np.array(features).shape)
    return np.array(features)


# path = 'train_data/male/Untitled 016.wav'
# a = feature_extraction(path, sr=22050, mono=True, frame_len=1000, hop_len=50)
#
# print('a', a)
