
import numpy as np
from framing import framing as framing


y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 23, 24, 25, 26, 27, 28, 20, 30, 31, 32, 33, 34])

frame_len = 10
hop_len = 2

yy = framing(y, frame_len, hop_len)
print(yy)

# flag = True
#
# start_index = 0
# frames = []
# while flag:
#
#     if start_index + frame_len >= y.size:
#         frame = y[y.size - frame_len:]
#         frames.append(frame)
#         break
#
#     frame = y[start_index:start_index + frame_len]
#     frames.append(frame)
#     start_index += (frame_len - hope_len)
#
# print(np.array(frames))
# for i in range()
# print(x[0:5])


# def framing(y, frame_len, hop_length):
#
#     flag = True
#     start_index = 0
#     frames = []
#
#     while flag:
#
#         if start_index + frame_len >= y.size:
#             frame = y[y.size - frame_len:]
#             frames.append(frame)
#             break
#
#         frame = y[start_index:start_index + frame_len]
#         frames.append(frame)
#         start_index += (frame_len - hope_len)
#
#     return np.array(frames)
