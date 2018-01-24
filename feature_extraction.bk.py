
# python modules
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy

# my own modules
from framing import framing

# consts
frame_len = 100
hop_len = 20

# load wav file
y, sr = librosa.load('train_data/male/Untitled 016.wav', sr=22050, mono=True)

# plt.figure(1)
# plt.plot(y)
# plt.show()


print('-' * 60)
print('y shape =>', y.shape)
print('y duration => ', librosa.core.get_duration(y))
print('-' * 60)

# trim it (remove silence from beggining and end)
yt, index = librosa.effects.trim(y)
y = np.array(yt)

print('trim_y shape =>', y.shape)
print('terim_y duration => ', librosa.core.get_duration(y))
print('-' * 60)

# plt.figure(1)
# plt.plot(y)
# plt.show()


frames = framing(y, frame_len, hop_len)

# it's not a good function for framming in Dr bakhtiari project i think
# frames = librosa.util.frame(y, frame_length=int(y.size / frame_len), hop_length=1)

print('frames shape =>', frames.shape)
print('frames[0] shape =>', frames[0].shape)
print('-' * 60)


energys = []
zcrs = []
mfccs = []

for frame in frames:
    # print(i)

    # frame_features = []

    # calculate energy of this frame
    energy = np.sum(np.power(frame, 2))
    energys.append(energy)

    # print('energy =>', energy)
    # print('-' * 60)

    # print(np.power(frame, 2))
    # print(frame)
    # plt.figure(1)
    # plt.plot(frame)
    # plt.plot(np.power(frame, 2))
    # plt.show()

    zcr = np.count_nonzero(librosa.core.zero_crossings(frame))
    zcrs.append(zcr)

    # print('zcr count =>', zcr)
    # print('-' * 60)

    # plt.figure(1)
    # plt.plot(frame)
    # plt.plot(librosa.core.zero_crossings(frame))
    # plt.show()

    mfcc = librosa.feature.mfcc(y=frame, sr=sr)

    # TODO should we use other dimentions of this mfcc

    this_mfcc = []

    for i in range(20):
        this_mfcc.append(mfcc[i][0])

    mfccs.append(this_mfcc)

    # print('mfcc shape =>', mfcc.shape)
    # print('mfcc =>', mfcc)
    # print('-' * 60)

    # plt.figure(1)
    # plt.plot(frame)
    # plt.plot(mfcc)
    # plt.show()

    # TODO add this to array
    # caculate diff 1
    # diff1 = np.diff(frame, 1)

    # print('diff1 shape =>', diff1.shape)
    # print('diff1 =>', diff1)

    # TODO add this to array
    # caculate diff 2
    # diff2 = np.diff(frame, 2)

    # print('diff2 shape =>', diff2.shape)
    # print('diff2 =>', diff2)

    # plt.figure(1)
    # plt.plot(frame)
    # plt.plot(diff1)
    # plt.plot(diff2)
    # plt.show()

    # autocorrelate = librosa.autocorrelate(frame)
    #
    # print('autocorrelate shape =>', autocorrelate.shape)
    # print(autocorrelate)
    # plt.figure(1)
    # plt.plot(frame)
    # plt.plot(autocorrelate)
    # plt.show()

    # print('frame_features shape =>', np.array(frame_features).shape)
    # print('frame_features', frame_features)

    # features.append(frame_features)

zcrs = np.array(zcrs)
energys = np.array(energys)
mfccs = np.array(mfccs)

print('min =>', zcrs.min())
print('max =>', zcrs.max())
print('mean =>', np.mean(zcrs))
print('std =>', np.std(zcrs))
print('kurtosis =>', scipy.stats.kurtosis(zcrs))
print('skew =>', scipy.stats.skew(zcrs))

print('zcrs shape =>', zcrs.shape)
print('energys shape =>', energys.shape)
print('mfccs shape =>', mfccs.shape)

features = []


# print('features shape =>', np.array(features).shape)
# print('features', np.array(features))


# librosa.output.write_wav('write_test.wav', frame, sr)
