import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


1
# # 读取音频文件
# input_file1 = './Real Noise Examples/QQmusic_1.1.wav'
# input_file2 = './Real Noise Examples/AirConditioner_1.1.wav'
#
# y1, sr1 = librosa.load(input_file1)
# y2, sr2 = librosa.load(input_file2)
#
# # 计算短时傅里叶变换
# D1 = librosa.stft(y1)
# D2 = librosa.stft(y2)
#
# # 取幅度谱
# S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
# S_db2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)
#
# # 绘制频谱图
# plt.figure(figsize=(10, 4))
# plt.subplot(2, 1, 1)
# librosa.display.specshow(S_db1, sr=sr1, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.title('Spectrogram of QQmusic_1.1.wav')
#
# plt.subplot(2, 1, 2)
# librosa.display.specshow(S_db2, sr=sr2, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.title('Spectrogram of AirConditioner_1.1.wav')
#
# plt.tight_layout()
# plt.show()

2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 读取音频文件
fs, data = wavfile.read('./Real Noise Examples/AirConditioner_1.1.wav')
# fs, data = wavfile.read('./Real Noise Examples/merge.wav')

# 计算音频数据的傅里叶变换
fft_result = np.fft.fft(data)
magnitude = np.abs(fft_result)
frequencies = np.fft.fftfreq(len(magnitude), 1/fs)


# 绘制频谱图
plt.figure(figsize=(12, 6))
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
plt.title('Frequency Spectrum of AirConditioner_1.1.wav')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.show()