
import librosa
import librosa.display
import matplotlib.pyplot as plt

import numpy as np

# load wav file
y, sr = librosa.load('train_data/male/Untitled 017.wav', sr=22050, mono=True)

# convet to numpy array
y = np.array(y)

# plt.figure(1)
# plt.plot(y)
# plt.show()



exit()

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
