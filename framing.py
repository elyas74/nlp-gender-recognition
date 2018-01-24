
import numpy as np


def framing(y, frame_len, hop_len):

    flag = True
    start_index = 0
    frames = []

    while flag:

        if start_index + frame_len >= y.size:
            frame = y[y.size - frame_len:]
            frames.append(frame)
            break

        frame = y[start_index:start_index + frame_len]
        frames.append(frame)
        start_index += (frame_len - hop_len)

    return np.array(frames)
