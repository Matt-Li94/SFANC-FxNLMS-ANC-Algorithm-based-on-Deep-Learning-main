from pesq import pesq
from pystoi.stoi import stoi
import soundfile as sf
import numpy as np

ref_file = './Real Noise Examples/D110_1.1.wav'  # pour音频 参考文件
deg_file = './Real Noise Examples/fxlms1.wav'  # 滤波音频 评估文件PESQ score is 1.0282589197158813
noi_file = './Real Noise Examples/AirConditioner_1.1.wav'  # 滤波音频 评估文件

resample_rate = 16000  # 采样率

# 读取音频文件数据
ref_data, _ = sf.read(ref_file)
deg_data, _ = sf.read(deg_file)
noi_data, _ = sf.read(noi_file)

# # 如果数据是多通道的，选择一个通道进行评估
ref_data = ref_data[:, 1]  # 选择第一个通道
# deg_data = deg_data[:, 0]  # 选择第一个通道

# 计算 PESQ 分数
score = pesq(resample_rate, ref_data, deg_data)
print('PESQ score is', score)  # 打印评分

# # 调整参考文件和评估文件的长度为相同的长度
# min_length = min(len(ref_data), len(deg_data))
# ref_data = ref_data[:min_length]
# deg_data = deg_data[:min_length]
# # # 计算 STOI 分数
# score = stoi(ref_data, deg_data, resample_rate, extended=False)
# print('STOI score is', score)  # 打印 STOI 分数


# 计算信号能量
signal_energy = np.sum(ref_data ** 2)
# 计算噪声文件的能量
noise_energy = np.sum(noi_data ** 2)
# 计算信噪比（SNR）
snr = 10 * np.log10(signal_energy / noise_energy)
print('SNR is', snr)  # 打印 SNR


