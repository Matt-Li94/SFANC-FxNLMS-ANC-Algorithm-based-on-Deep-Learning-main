1
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat
import math
from loading_real_wave_noise import loading_real_wave_noise
from Reading_path_test import loading_paths_from_MAT
from Control_filter_selection import Control_filter_selection
from FxNLMS_algorithm import FxNLMS, train_fxnlms_algorithm
from VSS_FxNLMS_algorithm import VSS_FxNLMS, train_vss_fxnlms_algorithm
from Disturbance_generation import Disturbance_generation_from_real_noise
from Combine_SFANC_with_FxNLMS import SFANC_FxNLMS
from Combine_SFANC_with_VSS_FxNLMS import SFANC_VSS_FxNLMS
from Read_the_data import read

print(torch.cuda.is_available())

2
# real noises
fs = 16000
#StepSize = 0.0001
sound_name = 'merge'
waveform, resample_rate = loading_real_wave_noise(folde_name='Real Noise Examples/', sound_name=sound_name+'.wav')
print('yeap')

3
# Pri_path, Secon_path = loading_paths_from_MAT(folder='Pz and Sz', subfolder='Dongyuan', Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
Pri_path, Secon_path = loading_paths_from_MAT(folder='C:\PycharmProjects\SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning-main', subfolder='Primary and Secondary Path',Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
Dis, Fx, Re = Disturbance_generation_from_real_noise(fs=fs, Repet=0, wave_form=waveform, Pri_path=Pri_path, Sec_path=Secon_path)
# Dis: disturbance (cotrolled noise)， Fx: fixed-x signal, Re: repeated waveform (primary_noise) Repetition=Repet+1
print('yeap')

4
print(waveform.shape)
print(Re.shape)
print(Dis.shape)

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

5
from sqlite3 import Time

# FxNLMS

min = 0.002
max = 0.1 #没有用
beta = 0.1


controller = VSS_FxNLMS(Len=1024,mu_min=min, mu_max=max, beta=beta) # 1024 is the same size of coeffient vector of fixed-filter
ErrorVSSFxNLMS = train_vss_fxnlms_algorithm(Model=controller, Ref=Fx, Disturbance=Dis, mu_min=min, mu_max=max, beta=beta) #训练VSS_FxNLMS模型

# print(ErrorFxNLMS)
# print(Dis)

# 假设 ErrorFxNLMS 是一个包含FxNLMS误差的列表或数组
ErrorFxNLMS = np.array(ErrorVSSFxNLMS)  # 将 ErrorFxNLMS 转换为 numpy 数组

# 创建与 ErrorFxNLMS 长度相同的时间序列
start_time = 0  # 开始时间
end_time = 10 # 结束时间 要修改
lenth = 160000
time_sequence = torch.linspace(start_time, end_time, steps=lenth)


6
# Combine SFANC with VSS_FxNLMS

# prediction index
id_vector = Control_filter_selection(fs=16000, Primary_noise=Re.unsqueeze(0)) # Primary_noise: torch.Size([1, XX])
print(id_vector) #这个本身含有预测

# Using prediction index in SFANC_VSS_FxNLMS
FILE_NAME_PATH = 'Trained models/Pretrained_Control_filters.mat'
SFANC_VSS_FxNLMS_Cancellation = SFANC_VSS_FxNLMS(MAT_FILE=FILE_NAME_PATH, fs=16000) #又来一个预测
Error_SFANC_VSS_FxNLMS = SFANC_VSS_FxNLMS_Cancellation.noise_cancellation(Dis=Dis, Fx=Fx, filter_index=id_vector, mu_min=0.01, mu_max=0.1, beta=0.01)

plt.title('The Hybrid VSS_FxNLMS Algorithm about merge.wav.wav')
plt.plot(time_sequence, Dis, color='gold', label='ANC off')
plt.plot(time_sequence, ErrorVSSFxNLMS, color='violet', label='ANC on')
plt.ylabel('Magnitude')
plt.xlabel('Time (seconds)')
plt.legend()
plt.grid()
# plt.savefig('./pdf/SFANC_VSS_FxNLMS_NeighborSpeaking_1.1.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

plt.title('The Hybrid SFANC-FxNLMS Algorithm about merge.wav')
plt.plot(time_sequence, Dis, color='blue', label='ANC off')
plt.plot(time_sequence, Error_SFANC_VSS_FxNLMS, color='red', label='ANC on')
plt.ylabel('Magnitude')
plt.xlabel('Time (seconds)')
plt.legend()
plt.grid()
# plt.savefig('./pdf/SFANC_FxNLMS_CopyMachine_1.1.wav.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()



def run_vss_anc_algorithm(sound_name,Wc1,count_i,end_time=10):
    # real noises
    fs = 16000
    waveform, resample_rate = loading_real_wave_noise(folde_name='Real Noise Examples/', sound_name=sound_name+'.wav')
    Pri_path, Secon_path = loading_paths_from_MAT(folder='C:\PycharmProjects\SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning-main', subfolder='Primary and Secondary Path', Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
    Dis, Fx, Re = Disturbance_generation_from_real_noise(fs=fs, Repet=0, wave_form=waveform, Pri_path=Pri_path, Sec_path=Secon_path)

    controller = VSS_FxNLMS(Len=1024, mu_min=0.01, mu_max=0.1, beta=0.01)

    # ErrorFxNLMS = train_vss_fxnlms_algorithm(Model=controller, Ref=Fx, Disturbance=Dis, mu_min=0.01, mu_max=0.1, beta=0.01)
    ErrorFxNLMS = train_fxnlms_algorithm(Model=controller, Ref=Fx, Disturbance=Dis, Stepsize=0.002) #要iu画出原始的和变异之后的
    ErrorFxNLMS = torch.tensor(ErrorFxNLMS)

    start_time = 0
    lenth = 160000
    time_sequence = torch.linspace(start_time, end_time, steps=lenth)

    id_vector = Control_filter_selection(fs=16000, Primary_noise=Re.unsqueeze(0))

    FILE_NAME_PATH = 'Trained models/Pretrained_Control_filters.mat'
    SFANC_VSS_FxNLMS_Cancellation = SFANC_VSS_FxNLMS(MAT_FILE=FILE_NAME_PATH, fs=16000,Wc0=Wc1)
    Error_SFANC_VSS_FxNLMS = SFANC_VSS_FxNLMS_Cancellation.noise_cancellation(Dis=Dis, Fx=Fx, filter_index=id_vector, mu_min=0.01, mu_max=0.1, beta=0.01)

    plt.title(f'The Hybrid GA-SFANC_VSS_FxNLMS Algorithm about {sound_name}')
    plt.plot(time_sequence, Dis, color='green', label='ANC off')
    plt.plot(time_sequence, Error_SFANC_VSS_FxNLMS, color='magenta', label='ANC on')
    plt.ylabel('Magnitude')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.grid()
    plt.show()

    Error_SFANC_VSS_FxNLMS = torch.tensor(Error_SFANC_VSS_FxNLMS)
    # 计算SFANC_VSS_FxNLMS的误差

    Error_on_SFANC_VSS_FxNLMS = np.abs(Error_SFANC_VSS_FxNLMS - Dis)
    # 此处的计算两个的差值感觉有误，因为出现了Error_SFANC_FxNLMS大于Dis的情况，所以此处尝试修改为直接对Error_SFANC_FxNLMS的求和进行寻优
    # 将 Error_SFANC_VSS_FxNLMS 中每个数值转换为其绝对值，然后再求和，求其最小值。
    Error_SFANC_VSS_FxNLMS_abs = np.abs(Error_SFANC_VSS_FxNLMS)
    Error_SFANC_VSS_FxNLMS_abs_sum = Error_SFANC_VSS_FxNLMS_abs.sum().item()

    print(f'Error_SFANC_VSS_FxNLMS_abs_sum is {Error_SFANC_VSS_FxNLMS_abs_sum}')

    Error_off_FxNLMS = np.abs(ErrorFxNLMS - Dis)
    Error_on_SFANC_FxNLMS = np.abs(Error_SFANC_VSS_FxNLMS - Dis)

    # 绘制对比图像
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_sequence, Error_off_FxNLMS, color='blue', label='Error - FxNLMS')
    plt.ylabel('Magnitude')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time_sequence, Error_on_SFANC_FxNLMS, color='red', label='Error - SFANC_VSS_FxNLMS')
    plt.ylabel('Magnitude')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.grid()

    plt.suptitle(f'Error Comparison - {sound_name}.wav and it is the {count_i}th drawing')
    plt.tight_layout()
    plt.show()

    return 1 / Error_SFANC_VSS_FxNLMS_abs_sum
    # 返回倒数以获得函数的最大值，实质上是得到了Error_SFANC_VSS_FxNLMS_abs_sum的最小值 1 / data


# # 调用函数并传入参数
# Wc0 = read()
# run_vss_anc_algorithm(sound_name='NeighborSpeaking_1.1',Wc1=Wc0,count_i=1, end_time=10)












