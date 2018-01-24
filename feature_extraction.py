
# python modules
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# my own modules
from framing import framing as framing

# consts
frame_len = 100
hop_len = 0

# load wav file
y, sr = librosa.load('train_data/male/Untitled 016.wav', sr=22050, mono=True)

# plt.figure(1)
# plt.plot(y)
# plt.show()


print('y shape =>', y.shape)
print('y duration => ', librosa.core.get_duration(y), 's')


# trim it (remove silence from beggining and end)
yt, index = librosa.effects.trim(y)
y = np.array(yt)


print('trim_y shape =>', y.shape)
print('terim_y duration => ', librosa.core.get_duration(y), 's')

# plt.figure(2)
# plt.plot(y)
# plt.show()


frames = framing(y, frame_len, hop_len)

# it's not a good function for framming in Dr bakhtiari project i think
# frames = librosa.util.frame(
# y, frame_length=int(y.size / frame_len), hop_length=1)

print('frames =>', frames.shape)
print('frames[0] =>', frames[0].shape)

print(frames[0])

exit()
# yy = librosa.feature.mfcc(y=y, sr=sr)
# print('yy =>', yy.shape)

# librosa.output.write_wav('write_test.wav', y, sr)


print('y_len :', len(y))

# delete beggining and ending silence
yt, index = librosa.effects.trim(y)


# frames
frames = librosa.util.frame(yt, frame_length=200, hop_length=100)

print('yt_len :', len(yt))

# woking on one frame for test
frame = frames[0]

# calculate energy of this frame
energy = np.sum(np.power(frame, 2))
# print(energy)


# calculate zcr array and count it
zcr = np.count_nonzero(librosa.core.zero_crossings(frame))
# print(zcr)


# calculate mfccs
# mfcc[0 .. 11]
mfcc = librosa.feature.mfcc(y=y, sr=sr)


# caculate diff 1 and 2
diff1 = np.diff(frame, 1)
# print(diff1)
diff2 = np.diff(frame, 2)
# print(diff2)


# plt.figure(1)
# plt.plot(mfcc)
# plt.show()


# frame = y[:800]
# print('frame_len :',len(frame))

# print(y[0],frame[0])

# print(31916/1024)
# print(len(dd))
# print(len(dd[0]))
# print(len(dd[1]))

# plt.figure(1)
# # plt.plot(frame)
# plt.plot(y)
# plt.show()

# BINS_PER_OCTAVE = 12 * 3
# N_OCTAVES = 7
# C = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr,
#                                         bins_per_octave=BINS_PER_OCTAVE,
#                                         n_bins=N_OCTAVES * BINS_PER_OCTAVE),ref=np.max)
# plt.figure(figsize=(12, 4))
# librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
#                          bins_per_octave=BINS_PER_OCTAVE,x_axis='time')
# plt.tight_layout()
# plt.show()
