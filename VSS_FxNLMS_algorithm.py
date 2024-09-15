import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import progressbar

#------------------------------------------------------------------------------
# Class: FxNLMS algorithm
#------------------------------------------------------------------------------
class VSS_FxNLMS():
    def __init__(self, Len, mu_min, mu_max, beta):
        self.Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float)  # 初始化的Wc是全0 可以优化
        self.Xd = torch.zeros(1, Len, dtype=torch.float)  # 用于接受输入数据，初始化也是0
        self.mu_min = mu_min  # 最小步长
        self.mu_max = mu_max  # 最大步长
        self.beta = beta  # 步长调整参数
        self.power_prev = 1.0
        self.mu = mu_min

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf  # 这个是输入数据
        yt = self.Wc @ self.Xd.t()
        power = self.Xd @ self.Xd.t()  # FxNLMS different from FxLMS
        return yt, power

    def update_step_size(self, power):
        mu = self.mu_min + (self.mu_max - self.mu_min) * self.beta / (self.beta + power / self.power_prev)
        self.power_prev = power
        return mu

    def get_mu(self):
        return self.mu

    def update_weights(self, e, mu):
        self.Wc = self.Wc + mu * e * self.Xd  # 深拷贝替代原地操作

    def LossFunction(self, y, d, power):
        e = d - y  # disturbance-control signal
        return e ** 2 / (2 * power), e

    def _get_coeff_(self):  # 返回参数
        return self.Wc.detach().numpy()

#------------------------------------------------------------------------------
# Function : train_fxlms_algorithm()
#------------------------------------------------------------------------------
def train_vss_fxnlms_algorithm(Model, Ref, Disturbance, mu_min, mu_max, beta):
    bar = progressbar.ProgressBar(maxval=2 * Disturbance.shape[0],
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    optimizer = optim.SGD([Model.Wc], lr=Model.get_mu())

    bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]

    for itera in range(len_data):
        # Feedforward
        xin = Ref[itera]
        dis = Disturbance[itera]
        y, power = Model.feedforward(xin)
        loss, e = Model.LossFunction(y, dis, power)

        # Progress shown
        bar.update(2 * itera + 1)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.item())

        # Progress shown
        bar.update(2 * itera + 2)

        # # Update step size
        # mu = Model.update_step_size(power)
        #
        # # Update weights
        # Model.update_weights(e, mu)
        # Erro_signal.append(e.item())
        #
        # # Progress shown
        # bar.update(2 * itera + 2)

    bar.finish()
    return Erro_signal


#--------------魔改篇--------------------
class VSS_FxNLMS_mu():
    def __init__(self, Len, mu_min, mu_max, beta):
        self.Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float)  # 初始化的Wc是全0 可以优化
        self.Xd = torch.zeros(1, Len, dtype=torch.float)  # 用于接受输入数据，初始化也是0
        self.mu_min = mu_min  # 最小步长
        self.mu_max = mu_max  # 最大步长
        self.beta = beta  # 步长调整参数
        self.power_prev = 1.0
        self.mu = mu_min

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf  # 这个是输入数据
        yt = self.Wc @ self.Xd.t()
        power = self.Xd @ self.Xd.t()  # FxNLMS different from FxLMS
        return yt, power

    def update_step_size(self, power):
        mu = self.mu_min + (self.mu_max - self.mu_min) * self.beta / (self.beta + power / self.power_prev)
        if mu <= self.mu_min:
            mu = self.mu_min
        if mu >= self.mu_max:
            mu = self.mu_max
        self.power_prev = power
        return mu

    def get_mu(self):
        return self.mu

    def update_weights(self, e, mu):
        self.Wc = self.Wc + mu * e * self.Xd  # 深拷贝替代原地操作

    def LossFunction(self, y, d,as0,as1, power):  #as0定义为通过真实二次路径的音频，as1定义为通过估计二次路径的音频
        e = d - y - as0 + as1  # disturbance-control signal
        return e ** 2 / (2 * power), e

    def _get_coeff_(self):  # 返回参数
        return self.Wc.detach().numpy()

#------------------------------------------------------------------------------
# Function : train_fxlms_algorithm()
#------------------------------------------------------------------------------
def train_vss_fxnlms_algorithm_mu(Model, Ref, Disturbance, mu_min, mu_max, beta,as0,as1):
    bar = progressbar.ProgressBar(maxval=2 * Disturbance.shape[0],
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    optimizer1 = optim.SGD([Model.Wc], lr=Model.get_mu())
    bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]

    for itera in range(len_data):
        # Feedforward
        xin = Ref[itera]
        dis = Disturbance[itera]
        AS0 = as0[itera] #as0定义为通过真实二次路径的音频
        AS1 = as1[itera] #as1定义为通过估计二次路径的音频
        y, power = Model.feedforward(xin)
        loss, e = Model.LossFunction(y, dis, AS0,AS1,power)

        # Progress shown
        bar.update(2 * itera + 1)

        # Backward
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        Erro_signal.append(e.item())

        # Progress shown
        bar.update(2 * itera + 2)

        # Update step size
        # mu = Model.update_step_size(power)

        # optimizer1.mu = mu


        # # Progress shown
        # bar.update(2 * itera + 2)

    bar.finish()
    return Erro_signal


#------------------------------------------------------------
# Function : Generating the testing bordband noise 
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024 
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y = signal.lfilter(bandpass_filter,1,xin) #使用 lfilter 函数对 xin 应用 bandpass_filter 滤波器，输出结果赋给变量 y。
    yout= y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)
    #将 yout 转换为 torch.Tensor 类型的数据，再转换为 float 类型，并在第0维上增加一个维度，最后作为函数的返回值返回