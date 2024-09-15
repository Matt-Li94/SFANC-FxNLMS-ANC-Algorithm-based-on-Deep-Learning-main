# 1
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat
import math
from loading_real_wave_noise import loading_real_wave_noise
from Reading_path_test import loading_paths_from_MAT
from Control_filter_selection import Control_filter_selection
from FxNLMS_algorithm import FxNLMS, train_fxnlms_algorithm
from Disturbance_generation import Disturbance_generation_from_real_noise
from Combine_SFANC_with_FxNLMS import SFANC_FxNLMS

import numpy as np
import soundfile as sf

#
# print(torch.cuda.is_available())
#
# # 2
# # real noises
# fs = 16000
# StepSize = 0.002 #原来是0.0001 学习率0.002暂时不要改变
# sound_name = 'merge' #Connect_Aircraft_Traffic
# waveform, resample_rate = loading_real_wave_noise(folde_name='Real Noise Examples/', sound_name=sound_name+'.wav')
# print('yeap')
#
# 3
# # Pri_path, Secon_path = loading_paths_from_MAT(folder='Pz and Sz', subfolder='Dongyuan', Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
# Pri_path, Secon_path = loading_paths_from_MAT(folder='C:\PycharmProjects\SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning-main', subfolder='Primary and Secondary Path',Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
# Dis, Fx, Re = Disturbance_generation_from_real_noise(fs=fs, Repet=0, wave_form=waveform, Pri_path=Pri_path, Sec_path=Secon_path)
# # Dis: disturbance (cotrolled noise)， Fx: fixed-x signal, Re: repeated waveform (primary_noise) Repetition=Repet+1
# # Dis：干扰（cotroll噪声）就是噪声经过初级通道得到的x（n）
# # Fx：固定-x信号，就是噪声通过次级通道得到的x1（n）
# # Re：重复波形（primary_noise）重复=Repet+1
#
# print('yeap')
#
# 4
# print(waveform.shape)
# print(Re.shape)
# print(Dis.shape)
#
# import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000
#
# 5
# from sqlite3 import Time
#
# # FxNLMS
#
# controller = FxNLMS(Len=1024) # 1024 is the same size of coeffient vector of fixed-filter
# ErrorFxNLMS = train_fxnlms_algorithm(Model=controller, Ref=Fx, Disturbance=Dis, Stepsize=StepSize)
#
# # sf.write('./Real Noise Examples/output_noise_reduced1.wav', ErrorFxNLMS, fs)
#
# # ErrorFxNLMS 是一个e（n）的列表或数组
# # 如果输入是噪声的话，那么e（n）趋于0就说明噪声完全被消除
# # 如果输入是含有噪声的音频的话，那么e（n）是p（t）*v（t）
# ErrorFxNLMS = np.array(ErrorFxNLMS)  # 将 ErrorFxNLMS 转换为 numpy 数组
# print(ErrorFxNLMS)
# print(Dis)
# # 创建与 ErrorFxNLMS 长度相同的时间序列
# start_time = 0  # 开始时间
# end_time = 10  # 结束时间 要修改
# lenth = 160000
# time_sequence = torch.linspace(start_time, end_time, steps=lenth)
#
# plt.title('The FxNLMS Algorithm about merge.wav')
# plt.plot(time_sequence, Dis, color='blue', label='ANC off')
# plt.plot(time_sequence, ErrorFxNLMS, color='green', label='ANC on')
# plt.ylabel('Magnitude')
# plt.xlabel('Time (seconds)')
# plt.legend()
# plt.grid()
# # plt.savefig('./pdf/FxNLMS_CopyMachine_1.1.wav.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
# plt.show()

# 6
# # Combine SFANC with FxNLMS
#
# # prediction index
# id_vector = Control_filter_selection(fs=16000, Primary_noise=Re.unsqueeze(0)) # Primary_noise: torch.Size([1, XX])
# print(id_vector)
#
# # Using prediction index in SFANC_FxNLMS
# FILE_NAME_PATH = 'Trained models/Pretrained_Control_filters.mat'
# SFANC_FxNLMS_Cancellation = SFANC_FxNLMS(MAT_FILE=FILE_NAME_PATH, fs=16000)
# Error_SFANC_FxNLMS = SFANC_FxNLMS_Cancellation.noise_cancellation(Dis=Dis, Fx=Fx, filter_index=id_vector, Stepsize=StepSize)
#
# # # 保存经过SFANC-FxNLMS混合算法降噪之后的音频文件
# # sf.write('./Real Noise Examples/output_noise_reduced.wav', Error_SFANC_FxNLMS, fs)
#
# # 将 Error_SFANC_FxNLMS 中每个数值转换为其绝对值
# Error_SFANC_FxNLMS_abs = np.abs(Error_SFANC_FxNLMS)
#
# plt.title('The Hybrid SFANC-FxNLMS Algorithm about merge.wav')
# plt.plot(time_sequence, Dis, color='blue', label='ANC off')
# plt.plot(time_sequence, Error_SFANC_FxNLMS, color='red', label='ANC on')
# plt.ylabel('Magnitude')
# plt.xlabel('Time (seconds)')
# plt.legend()
# plt.grid()
# # plt.savefig('./pdf/SFANC_FxNLMS_CopyMachine_1.1.wav.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
# plt.show()

# # 绘制 Error_SFANC_FxNLMS 前 10 个数据点的图像
# plt.figure()
# plt.title('First 10 Data Points of Error_SFANC_FxNLMS')
# plt.plot(time_sequence[:10], Error_SFANC_FxNLMS_abs[:10], color='purple', marker='o', linestyle='-', label='Error_SFANC_FxNLMS')
# plt.ylabel('Magnitude')
# plt.xlabel('Time (seconds)')
# plt.legend()
# plt.grid()
# plt.show()



# # 3.添加个GA-SFANC_FxNLMS绘图代码
# plt.title('The Hybrid GA-SFANC_FxNLMS Algorithm about Bus_1.2.wav')
# plt.plot(time_sequence, Dis, color='blue', label='ANC off')
# plt.plot(time_sequence, Error_SFANC_FxNLMS, color='darkorange', label='ANC on')
# plt.ylabel('Magnitude')
# plt.xlabel('Time (seconds)')
# plt.legend()
# plt.grid()
# # plt.savefig('./pdf/SFANC_FxNLMS_CopyMachine_1.1.wav.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
# plt.show()


7
def run_anc_algorithm(sound_name,Wc1,count_i,StepSize=0.002, end_time=10 ):
    fs = 16000
    waveform, resample_rate = loading_real_wave_noise(folde_name='Real Noise Examples/', sound_name=sound_name+'.wav')
    Pri_path, Secon_path = loading_paths_from_MAT(folder='C:\PycharmProjects\SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning-main', subfolder='Primary and Secondary Path',Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
    Dis, Fx, Re = Disturbance_generation_from_real_noise(fs=fs, Repet=0, wave_form=waveform, Pri_path=Pri_path, Sec_path=Secon_path)

    controller = FxNLMS(Len=1024)
    ErrorFxNLMS = train_fxnlms_algorithm(Model=controller, Ref=Fx, Disturbance=Dis, Stepsize=StepSize)
    ErrorFxNLMS = np.array(ErrorFxNLMS)

    time_sequence = torch.linspace(0, end_time, steps=len(ErrorFxNLMS))

    # plt.title(f'The FxNLMS Algorithm about {sound_name}.wav')
    # plt.plot(time_sequence, Dis, color='blue', label='ANC off')
    # plt.plot(time_sequence, ErrorFxNLMS, color='green', label='ANC on')
    # plt.ylabel('Magnitude')
    # plt.xlabel('Time (seconds)')
    # plt.legend()
    # plt.grid()
    # plt.show()

    id_vector = Control_filter_selection(fs=16000, Primary_noise=Re.unsqueeze(0))
    SFANC_FxNLMS_Cancellation = SFANC_FxNLMS(MAT_FILE='Trained models/Pretrained_Control_filters.mat', fs=16000, Wc0=Wc1)
    Error_SFANC_FxNLMS = SFANC_FxNLMS_Cancellation.noise_cancellation(Dis=Dis, Fx=Fx, filter_index=id_vector, Stepsize=StepSize)


    # 计算FxNLMS的误差
    ErrorFxNLMS = torch.tensor(ErrorFxNLMS)
    Error_off_FxNLMS = np.abs(ErrorFxNLMS - Dis)

    #
    # plt.title(f'The Hybrid GA-SFANC-FxNLMS Algorithm about {sound_name}.wav')
    # plt.plot(time_sequence, Dis, color='blue', label='ANC off')
    # plt.plot(time_sequence, Error_SFANC_FxNLMS, color='darkorange', label='ANC on')
    # plt.ylabel('Magnitude')
    # plt.xlabel('Time (seconds)')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Error_SFANC_FxNLMS_np = Error_SFANC_FxNLMS.detach().numpy()
    Error_SFANC_FxNLMS = torch.tensor(Error_SFANC_FxNLMS)
    # 计算SFANC_FxNLMS的误差

    Error_on_SFANC_FxNLMS = np.abs(Error_SFANC_FxNLMS - Dis)
    # 此处的计算两个的差值感觉有误，因为出现了Error_SFANC_FxNLMS大于Dis的情况，所以此处尝试修改为直接对Error_SFANC_FxNLMS的求和进行寻优
    # 将 Error_SFANC_FxNLMS 中每个数值转换为其绝对值，然后再求和，求其最小值。
    Error_SFANC_FxNLMS_abs = np.abs(Error_SFANC_FxNLMS)
    Error_SFANC_FxNLMS_abs_sum = Error_SFANC_FxNLMS_abs.sum().item()

    print(f'Error_SFANC_FxNLMS_abs_sum is {1/Error_SFANC_FxNLMS_abs_sum}')


    # # 绘制对比图像
    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(time_sequence, Error_off_FxNLMS, color='blue', label='Error - FxNLMS')
    # plt.ylabel('Magnitude')
    # plt.xlabel('Time (seconds)')
    # plt.legend()
    # plt.grid()
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(time_sequence, Error_on_SFANC_FxNLMS, color='red', label='Error - SFANC_FxNLMS')
    # plt.ylabel('Magnitude')
    # plt.xlabel('Time (seconds)')
    # plt.legend()
    # plt.grid()
    #
    # plt.suptitle(f'Error Comparison - {sound_name}.wav and it is the {count_i}th drawing')
    # plt.tight_layout()
    # plt.show()

    return 1 / Error_SFANC_FxNLMS_abs_sum
    # 返回倒数以获得函数的最大值，实质上是得到了Error_SFANC_FxNLMS_sum的最小值 1 / data
# # 调用函数并传入参数
# # run_anc_algorithm(sound_name='CopyMachine_1.1', StepSize=0.002, end_time=10)


















