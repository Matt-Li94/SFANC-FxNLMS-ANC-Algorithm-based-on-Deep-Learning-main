import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import scipy.io as sio
import pandas as pd
from Read_the_data import read
#------------------------------------------------------------------------------
# Class: FxNLMS algorithm with initial coefficients determined by SFANC
#------------------------------------------------------------------------------
class FxNLMS():

    def __init__(self, Len, Ws):
        self.Wc = torch.tensor(Ws, requires_grad=True) # Ws: initial coefficients determined by SFANC
        self.Xd = torch.zeros(1, Len, dtype=torch.float)

    def feedforward(self,Xf):
        self.Xd = torch.roll(self.Xd,1,1)
        self.Xd[0,0] = Xf
        yt = self.Wc.double() @ self.Xd.t().double()
        power = self.Xd.double() @ self.Xd.t().double()  # 明确设置数据类型为 Double # different from FxLMS
        return yt, power

    def LossFunction(self, y, d, power):
        e = d-y
        return e**2/(2*power), e

    def _get_coeff_(self):
        return self.Wc.detach().numpy()

# ------------------------------------------------------------------------------
# Function : train_fxlms_algorithm() 魔改篇
# ------------------------------------------------------------------------------
class FxNLMS_mu1():

    def __init__(self, Len, Ws):
        self.Wc = torch.tensor(Ws, requires_grad=True) # Ws: initial coefficients determined by SFANC
        self.Xd = torch.zeros(1, Len, dtype=torch.float)

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf  # 这个是输入数据，Xd是参考信号，就是噪声
        yt = self.Wc.double() @ self.Xd.t().double()  # yt相当于是消除信号
        power = self.Xd.double() @ self.Xd.t().double()  # FxNLMS different from FxLMS
        return yt, power

    def LossFunction_mu(self, y, d, as0, as1, power):  #as0定义为通过真实二次路径的音频，as1定义为通过估计二次路径的音频
        e = d - y - as0 + as1  # disturbance-control signal
        return e ** 2 / (2 * power), e

    def _get_coeff_(self):  # 返回参数
        return self.Wc.detach().numpy()



#----------------------------------------------------------------
# Function: SFANC_FxNLMS
# Description: Using FxNLMS to optimize the control filter, the initial weights come from SFANC
#----------------------------------------------------------------
class SFANC_FxNLMS():
    # # 原始版本用这个 GA最优子代可以直接赋值给self.Current_Filter
    # def __init__(self, MAT_FILE, fs):  # MAT_FILE是.mat文件的路径，fs是采样频率。
    #     self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE)  # torch.Size([15, 1024])
    #     Len = self.Wc.shape[1]
    #     self.fs = fs
    #     self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)  #原始版本 第一代
    #     # self.Current_Filter = read() # read()望周知 GA寻优之后的赋值版本 第三代 读取数据并且赋值

    # 对init函数传入一个初始化的Wc GA寻优版本用这个 第二代
    def __init__(self, MAT_FILE, fs, Wc0):  # MAT_FILE是.mat文件的路径，fs是采样频率。
        self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE)  # torch.Size([15, 1024])
        Len = self.Wc.shape[1]
        self.fs = fs
        self.Current_Filter = Wc0


    def noise_cancellation(self, Dis, Fx, filter_index, Stepsize):
        Error = []
        j = 0
        model = FxNLMS(Len=1024, Ws=self.Current_Filter)
        optimizer = optim.SGD([model.Wc], lr=Stepsize)  # Stepsize is learning_rate

        for ii, dis in enumerate(Dis):
            y, power = model.feedforward(Fx[ii])
            loss, e = model.LossFunction(y, dis, power)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Error.append(e.item())

            if (ii + 1) % self.fs == 0:
                # print(j)
                # if self.Current_Filter[0].equal(self.Wc[filter_index[j]]) == False:
                if not np.array_equal(self.Current_Filter[0], self.Wc[filter_index[j]]):
                    # if prediction index is changed, change initial weights of FxNLMS
                    # print('change the initial weights of FxNLMS')
                    self.Current_Filter = self.Wc[filter_index[j]].unsqueeze(0)  # torch.Size([1, 1024])
                    model = FxNLMS(Len=1024, Ws=self.Current_Filter)
                    optimizer = optim.SGD([model.Wc], lr=Stepsize)  # Stepsize is learning_rate
                j += 1
        return Error

    def Load_Pretrained_filters_to_tensor(self, MAT_FILE):  # Loading the pre-trained control filter from the mat file
        mat_contents = sio.loadmat(MAT_FILE)
        Wc_vectors = mat_contents['Wc_v']
        return torch.from_numpy(Wc_vectors).type(torch.float)


class SFANC_FxNLMS_mu():
    # 原始版本用这个 GA最优子代可以直接赋值给self.Current_Filter
    def __init__(self, MAT_FILE, fs):  # MAT_FILE是.mat文件的路径，fs是采样频率。
        self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE)  # torch.Size([15, 1024])
        Len = self.Wc.shape[1]
        self.fs = fs
        # self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)  #原始版本 第一代
        self.Current_Filter = read('best_solution') # read()望周知 GA寻优之后的赋值版本 第三代 读取数据并且赋值

    # # 对init函数传入一个初始化的Wc GA寻优版本用这个 第二代
    # def __init__(self, MAT_FILE, fs, Wc0):  # MAT_FILE是.mat文件的路径，fs是采样频率。
    #     self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE)  # torch.Size([15, 1024])
    #     Len = self.Wc.shape[1]
    #     self.fs = fs
    #     self.Current_Filter = Wc0


    def noise_cancellation_mu(self, Dis, Fx, filter_index, Stepsize, as0, as1):
        Error = []
        j = 0
        model = FxNLMS_mu1(Len=1024, Ws=self.Current_Filter)
        optimizer = optim.SGD([model.Wc], lr=Stepsize)  # Stepsize is learning_rate

        for ii, dis in enumerate(Dis):
            y, power = model.feedforward(Fx[ii])
            loss, e = model.LossFunction_mu(y, dis, as0[ii], as1[ii], power)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Error.append(e.item())

            if (ii + 1) % self.fs == 0:
                print(j)
                # if self.Current_Filter[0].equal(self.Wc[filter_index[j]]) == False:
                if not np.array_equal(self.Current_Filter[0], self.Wc[filter_index[j]]):
                    # if prediction index is changed, change initial weights of FxNLMS
                    # print('change the initial weights of FxNLMS')
                    self.Current_Filter = self.Wc[filter_index[j]].unsqueeze(0)  # torch.Size([1, 1024])
                    model = FxNLMS_mu1(Len=1024, Ws=self.Current_Filter)
                    optimizer = optim.SGD([model.Wc], lr=Stepsize)  # Stepsize is learning_rate
                j += 1
        return Error

    def Load_Pretrained_filters_to_tensor(self, MAT_FILE):  # Loading the pre-trained control filter from the mat file
        mat_contents = sio.loadmat(MAT_FILE)
        Wc_vectors = mat_contents['Wc_v']
        return torch.from_numpy(Wc_vectors).type(torch.float)