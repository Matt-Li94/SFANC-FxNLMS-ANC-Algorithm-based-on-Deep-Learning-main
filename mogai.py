# 1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import savemat
import math
from estmate_Sz import run_LMS_filter
from loading_real_wave_noise import loading_real_wave_noise
from Reading_path_test import loading_paths_from_MAT
from Control_filter_selection import Control_filter_selection
from FxNLMS_algorithm import FxNLMS, train_fxnlms_algorithm ,train_fxnlms_algorithm_mu ,FxNLMS_mu
from VSS_FxNLMS_algorithm import VSS_FxNLMS_mu,train_vss_fxnlms_algorithm_mu
from Disturbance_generation import Disturbance_generation_from_real_noise
from Combine_SFANC_with_FxNLMS import SFANC_FxNLMS,SFANC_FxNLMS_mu ,FxNLMS_mu1
from Read_the_data import write
import numpy as np
import soundfile as sf


print(torch.cuda.is_available())

# 2
# real noises
fs = 16000
StepSize = 0.002 #原来是0.0001 学习率0.002暂时不要改变
sound_name = 'AirConditioner_1.1' #Connect_Aircraft_Traffic
music_name = 'D110_1.1'
waveform, resample_rate = loading_real_wave_noise(folde_name='Real Noise Examples/', sound_name=sound_name+'.wav')
waveform1, resample_rate1 = loading_real_wave_noise(folde_name='Real Noise Examples/', sound_name=music_name+'.wav')
print('yeap')



3
# Pri_path, Secon_path = loading_paths_from_MAT(folder='Pz and Sz', subfolder='Dongyuan', Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
Pri_path, Secon_path = loading_paths_from_MAT(folder='C:\PycharmProjects\SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning-main', subfolder='Primary and Secondary Path',Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
Dis, Fx, Re = Disturbance_generation_from_real_noise(fs=fs, Repet=0, wave_form=waveform, Pri_path=Pri_path, Sec_path=Secon_path)

# Dis: disturbance (cotrolled noise)， Fx: fixed-x signal, Re: repeated waveform (primary_noise) Repetition=Repet+1
# Dis：干扰（cotroll噪声）就是噪声经过初级通道得到的x（n）
# Fx：固定-x信号，就是噪声通过次级通道得到的x1（n）
# Re：重复波形（primary_noise）重复=Repet+1

# -----------------魔改部分
Dis1, Fx1, Re1 = Disturbance_generation_from_real_noise(fs=fs, Repet=0, wave_form=waveform1, Pri_path=Pri_path, Sec_path=Secon_path)
# -------------------
# print('yeap')

4
# print(waveform.shape)
# print(Re.shape)
# print(Dis.shape)
# print(Fx.shape)
# print(waveform1.shape)
# print(waveform1.shape)

waveform1 = waveform1.reshape([1,320000])
waveform1 = waveform1.narrow(1,1,160000)
waveform11 = waveform1.squeeze()
nc = waveform11+waveform
nc = nc.squeeze()
# print(waveform11.shape)

b = 0.8
#修改信号比部分
Fx1 = Fx1 * b


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

# ---------------------------------
# 在线估计二次路径
Order = Secon_path.shape
order = Order[0]
step_size = 0.000001
estimated_signals, errors, Sz = run_LMS_filter(order,step_size)
# Sz = read('weights_list')
# print(Sz)
Sz = Sz.squeeze()
# print(Sz.shape)
# print(Secon_path.shape)
# 需要获取as0和as1的值，需要卷积
as0 = np.convolve(waveform11, Secon_path, mode='same')  #卷积
as1 = np.convolve(waveform11, Sz, mode='same')  #卷积

# ---------------------------------
# 5
from sqlite3 import Time
# FxNLMS or FxNLMS_mu
controller = FxNLMS_mu(Len=1024) # 1024 is the same size of coeffient vector of fixed-filter
ErrorFxNLMS = train_fxnlms_algorithm_mu(Model=controller, Ref=Fx, Disturbance=Dis, as0=as0, as1=as1,Stepsize=StepSize)
ErrorFxNLMS_tensor = torch.tensor(ErrorFxNLMS)
# sum_error_fxnlms = ErrorFxNLMS_tensor + Fx1*1
sum_error_fxnlms = ErrorFxNLMS_tensor + Fx1*1
# 将 sum_error_fxnlms 转换为 NumPy 数组
sum_error_fxnlms_np = sum_error_fxnlms.numpy()

sf.write('./Real Noise Examples/fxlms1.wav', ErrorFxNLMS, fs, format='wav')
# # ErrorFxNLMS 是一个e（n）的列表或数组
# # 如果输入是噪声的话，那么e（n）趋于0就说明噪声完全被消除
# # 如果输入是含有噪声的音频的话，那么e（n）是p（t）*v（t）
# ErrorFxNLMS = np.array(ErrorFxNLMS)  # 将 ErrorFxNLMS 转换为 numpy 数组


# # 创建与 ErrorFxNLMS 长度相同的时间序列
start_time = 0  # 开始时间
end_time = 10  # 结束时间 要修改
lenth = 160000
time_sequence = torch.linspace(start_time, end_time, steps=lenth)
#
# # -------------------------------------------------------
# plt.title('The FxNLMS Algorithm about AirConditioner_1.1.wav')
# plt.plot(time_sequence, Dis, color='blue', label='ANC off')
# plt.plot(time_sequence, ErrorFxNLMS, color='green', label='ANC on')
# plt.ylabel('Magnitude')
# plt.xlabel('Time (seconds)')
# plt.legend()
# plt.grid()
# # plt.savefig('./pdf/FxNLMS_CopyMachine_1.1.wav.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
# plt.show()

# -------------------------------------------------------

# 6
# # Combine SFANC with FxNLMS
# # prediction index
# id_vector = Control_filter_selection(fs=16000, Primary_noise=Re.unsqueeze(0)) # Primary_noise: torch.Size([1, XX])
# print(id_vector)
# # Using prediction index in SFANC_FxNLMS
# FILE_NAME_PATH = 'Trained models/Pretrained_Control_filters.mat'
# SFANC_FxNLMS_Cancellation = SFANC_FxNLMS_mu(MAT_FILE=FILE_NAME_PATH, fs=16000)
# Error_SFANC_FxNLMS = SFANC_FxNLMS_Cancellation.noise_cancellation_mu(Dis=Dis, Fx=Fx, filter_index=id_vector, Stepsize=StepSize,as0=as0,as1=as1)
# Error_SFANC_FxNLMS= torch.tensor(Error_SFANC_FxNLMS)
# sum_error_sfanc_fxnlms = Error_SFANC_FxNLMS + Fx1*1
# # 将 sum_error_fxnlms 转换为 NumPy 数组
# sum_error_sfanc_fxnlms_np = sum_error_sfanc_fxnlms.numpy()
# # 保存经过SFANC-FxNLMS混合算法降噪之后的音频文件
# sf.write('./Real Noise Examples/sfanc_fxlms2.wav', sum_error_sfanc_fxnlms_np, fs)
#
# # -------------------------------------------------------
# # 将 Error_SFANC_FxNLMS 中每个数值转换为其绝对值
# Error_SFANC_FxNLMS_abs = np.abs(Error_SFANC_FxNLMS)
# plt.title('The Hybrid SFANC-FxNLMS Algorithm about AirConditioner_1.1.wav')
# plt.plot(time_sequence, Dis, color='blue', label='ANC off')
# plt.plot(time_sequence, Error_SFANC_FxNLMS, color='red', label='ANC on')
# plt.ylabel('Magnitude')
# plt.xlabel('Time (seconds)')
# plt.legend()
# plt.grid()
# # plt.savefig('./pdf/SFANC_FxNLMS_CopyMachine_1.1.wav.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
# plt.show()
# # -------------------------------------------------------

7
start_time = 0  # 开始时间
end_time = 10  # 结束时间 要修改
lenth = 160000
time_sequence11 = torch.linspace(start_time, end_time, steps=lenth)

nc = waveform11+waveform
nc = nc.squeeze()
waveform2 = waveform.squeeze()
# 绘制音乐和噪声的波形图
plt.figure()
plt.plot(time_sequence11, waveform2, color='orange')
plt.title('noise')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

plt.figure()
plt.plot(time_sequence11, waveform11, color='orange')
plt.title('music')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# 绘制Fx1的波形图
plt.figure()
plt.plot(time_sequence11, nc, color='green')
plt.title('noise-music')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()


8
# -----------GA-SFANC-FXLMS-------------------
# from GA import genetic_algorithm
# Combine SFANC with FxNLMS and GA
# prediction index
id_vector = Control_filter_selection(fs=16000, Primary_noise=Re.unsqueeze(0)) # Primary_noise: torch.Size([1, XX])
# Using prediction index in SFANC_FxNLMS
FILE_NAME_PATH = 'Trained models/Pretrained_Control_filters.mat'
SFANC_FxNLMS_Cancellation1 = SFANC_FxNLMS_mu(MAT_FILE=FILE_NAME_PATH, fs=16000)
Error_SFANC_FxNLMS1 = SFANC_FxNLMS_Cancellation1.noise_cancellation_mu(Dis=Dis, Fx=Fx, filter_index=id_vector, Stepsize=StepSize,as0=as0,as1=as1)
Error_SFANC_FxNLMS1= torch.tensor(Error_SFANC_FxNLMS1)

sum_error_sfanc_fxnlms1 = Error_SFANC_FxNLMS1
# 将 sum_error_fxnlms 转换为 NumPy 数组
sum_error_sfanc_fxnlms_np1 = sum_error_sfanc_fxnlms1.numpy()
# 保存经过SFANC-FxNLMS混合算法降噪之后的音频文件
sf.write('./Real Noise Examples/ga_sfanc_fxlms5.wav', Error_SFANC_FxNLMS1, fs)

plt.title('The Hybrid GA-SFANC-FxNLMS Algorithm about AirConditioner_1.1.wav')
plt.plot(time_sequence, Dis, color='teal', label='ANC off')
plt.plot(time_sequence, Error_SFANC_FxNLMS1, color='violet', label='ANC on')
plt.ylabel('Magnitude')
plt.xlabel('Time (seconds)')
plt.legend()
plt.grid()
# plt.savefig('./pdf/SFANC_FxNLMS_CopyMachine_1.1.wav.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

# 7
from scipy.signal import TransferFunction
from scipy.signal import lfilter
from scipy.constants import g
from scipy.linalg import lstsq
from scipy.signal.windows import kaiser

def adaptive_control_gain_algorithm(error_signal, reference_signal, max_gain, min_gain, step_size):
    """
    Adaptive control gain algorithm to adjust the gain dynamically based on error and reference signals.

    Parameters:
    error_signal (array): Error signal to calculate the gain.
    reference_signal (array): Reference signal.
    max_gain (float): Maximum gain value.
    min_gain (float): Minimum gain value.
    step_size (float): Step size to adjust the gain.

    Returns:
    adjusted_gain (array): Array of adjusted gain values based on the error and reference signals.
    """
    # Initialize array to store adjusted gain values
    adjusted_gain = []

    # Initialize gain with a default value
    gain = 0.5

    # Iterate through the error signal
    for error_value, reference_value in zip(error_signal, reference_signal):
        # Update gain based on error and reference signals
        gain += step_size * (error_value * reference_value)

        # Ensure gain stays within min and max limits
        gain = max(min_gain, min(max_gain, gain))

        # Append the adjusted gain to the list
        adjusted_gain.append(gain)

    return adjusted_gain


# Example usage of the adaptive control gain algorithm
# Assume you have error_signal and reference_signal arrays
adjusted_gain_values = adaptive_control_gain_algorithm(Error_SFANC_FxNLMS1, Fx1*0.1, max_gain=1.7, min_gain=0.5,
                                                       step_size=0.01)
sf.write('./Real Noise Examples/agc5.wav', adjusted_gain_values, fs)



# max_gain=1.3, min_gain=0.7的时候是 1.2621173858642578
# max_gain=1.4, min_gain=0.7的时候是 1.2619296312332153
# max_gain=1.3, min_gain=0.8的时候是 1.1193900108337402
# max_gain=1.5, min_gain=0.9的时候是 1.058366060256958
# max_gain=1.5 /7, min_gain=0.6的时候是 1.2658566236495972
# max_gain=1.8, min_gain=0.6的时候是 1.1699843406677246
# max_gain=1.8, min_gain=0.5的时候是 1.1210683584213257
# max_gain=1.8, min_gain=0.5，step_size=0.1的时候是 1.5788812637329102

from pesq import pesq
from pystoi.stoi import stoi
ref_file = './Real Noise Examples/D110_1.1.wav'  # pour音频 参考文件
deg_file = './Real Noise Examples/agc5.wav'  # 滤波音频 评估文件
noi_file = './Real Noise Examples/AirConditioner_1.11.wav'  # 滤波音频 评估文件

resample_rate = 16000  # 采样率

# 读取音频文件数据
ref_data, _ = sf.read(ref_file)
deg_data, _ = sf.read(deg_file)
noi_data, _ = sf.read(noi_file)

ref_data = ref_data * b
# # 如果数据是多通道的，选择一个通道进行评估
ref_data = ref_data[:, 0]  # 选择第一个通道
# deg_data = deg_data[:, 0]  # 选择第一个通道
ref_data = ref_data
# 计算 PESQ 分数
score = pesq(resample_rate, ref_data, deg_data)
print('PESQ score is', score)  # 打印评分

# 计算信号能量
signal_energy = np.sum(ref_data ** 2)
# 计算噪声文件的能量
noise_energy = np.sum(noi_data ** 2)
# 计算信噪比（SNR）
snr = 10 * np.log10(signal_energy / noise_energy)
print('SNR is', snr)  # 打印 SNR
