from Reading_path_test import loading_paths_from_MAT
import numpy as np
import pandas as pd

class LMSFilter:
    def __init__(self, order, step_size):
        self.order = order
        self.step_size = step_size
        self.weights = np.zeros(order)
        self.input_buffer = np.zeros(order)  # 初始化 input_buffer

    def update(self, input_signal, desired_signal):
        input_signal = np.concatenate(([input_signal], self.input_buffer), axis=None)[:self.order]  # 只取前 order 个元素
        estimated_signal = np.dot(self.weights, input_signal)  #y
        error = desired_signal - estimated_signal
        self.weights = self.weights + 2 * self.step_size * error * input_signal
        self.input_buffer = np.roll(self.input_buffer, 1)
        self.input_buffer[0] = input_signal[0]
        return estimated_signal, error, self.weights


# # Example usage
# order = 256 #长度
# step_size = 0.000001
# lms_filter = LMSFilter(order, step_size)
#
# Pri_path, Secon_path = loading_paths_from_MAT(folder='C:\PycharmProjects\SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning-main', subfolder='Primary and Secondary Path',Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
# print(Secon_path)
# input_signal = np.random.normal(1, 0.5, order)   #创建N长度的白噪声
# desired_signal = np.convolve(input_signal, Secon_path, mode='same')  #卷积产生期望信号
#
# estimated_signals = []
# errors = []
# weights_list = []
#
# for i in range(len(input_signal)):
#     estimated_signal, error, weights = lms_filter.update(input_signal[i], desired_signal[i])
#     estimated_signals.append(estimated_signal)
#     errors.append(error)
#     weights_list.append(weights)

# print("Estimated Signals:", estimated_signals)
# print("Errors:", errors)
# print("Final Weights:", weights_list[-1])
#
#
# # 写入ecxel表格中
# weights_df = pd.DataFrame(weights_list[-1])
# weights_df.to_excel('./Trained models/weights_list.xlsx', index=False)
#
#
# # 计算绝对误差
# absolute_error = np.abs(weights_list - Secon_path)
#
# # 计算平均绝对误差
# mean_absolute_error = np.mean(absolute_error)
#
# print("绝对误差数组：", absolute_error)
# print("平均绝对误差：", mean_absolute_error)


def run_LMS_filter(order, step_size):  #在线估计Sz
    lms_filter = LMSFilter(order, step_size)

    Pri_path, Secon_path = loading_paths_from_MAT(folder='C:\PycharmProjects\SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning-main', subfolder='Primary and Secondary Path',Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
    input_signal = np.random.normal(1, 0.5, order)
    desired_signal = np.convolve(input_signal, Secon_path, mode='same')

    estimated_signals = []
    errors = []
    weights_list = []

    for i in range(len(input_signal)):
        estimated_signal, error, weights = lms_filter.update(input_signal[i], desired_signal[i])
        estimated_signals.append(estimated_signal)
        errors.append(error)
        weights_list.append(weights)

    return estimated_signals, errors, weights_list[-1]

# # Example usage
# order = 256
# step_size = 0.000001
# estimated_signals, errors, weights_list = run_LMS_filter(order, step_size)

