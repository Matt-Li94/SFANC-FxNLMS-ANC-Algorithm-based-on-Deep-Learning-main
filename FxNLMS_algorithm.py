import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import progressbar

#------------------------------------------------------------------------------
# Class: FxNLMS algorithm
#------------------------------------------------------------------------------
class FxNLMS():
    
    def __init__(self, Len):
        self.Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float) #初始化的Wc是全0 可以优化
        self.Xd = torch.zeros(1, Len, dtype= torch.float) #用于接受输入数据，初始化也是0
    
    def feedforward(self,Xf):
        self.Xd = torch.roll(self.Xd,1,1)
        self.Xd[0,0] = Xf #这个是输入数据，Xd是参考信号，就是噪声
        yt = self.Wc @ self.Xd.t() #yt相当于是消除信号
        power = self.Xd @ self.Xd.t() # FxNLMS different from FxLMS
        return yt, power
    
    def LossFunction(self, y, d, power):
        e = d-y # disturbance-control signal
        return e**2/(2*power), e
    
    def _get_coeff_(self): #返回参数
        return self.Wc.detach().numpy()


#------------------------------------------------------------------------------
# Function : train_fxlms_algorithm()
#------------------------------------------------------------------------------
def train_fxnlms_algorithm(Model, Ref, Disturbance, Stepsize=0.0001): #原来Stepsize是0.0001

    bar = progressbar.ProgressBar(maxval=2*Disturbance.shape[0], \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #这里创建了一个进度条对象 bar，用于显示训练进度
    optimizer= optim.SGD([Model.Wc], lr=Stepsize) #创建一个随机梯度下降优化器
    
    # bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]
    for itera in range(len_data):
        # Feedfoward
        xin = Ref[itera]
        dis = Disturbance[itera]
        y, power = Model.feedforward(xin)
        loss, e = Model.LossFunction(y, dis, power)
        
        # Progress shown
        # bar.update(2*itera+1)
            
        # Backward
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.item())
        
        # Progress shown 
        # bar.update(2*itera+2)
    # bar.finish()
    return Erro_signal


# ------------------------------------------------------------------------------
# Function : train_fxlms_algorithm() 魔改篇
# ------------------------------------------------------------------------------
class FxNLMS_mu():

    def __init__(self, Len):
        self.Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float)  # 初始化的Wc是全0 可以优化
        self.Xd = torch.zeros(1, Len, dtype=torch.float)  # 用于接受输入数据，初始化也是0

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf  # 这个是输入数据，Xd是参考信号，就是噪声
        yt = self.Wc @ self.Xd.t()  # yt相当于是消除信号
        power = self.Xd @ self.Xd.t()  # FxNLMS different from FxLMS
        return yt, power

    def LossFunction(self, y, d,as0,as1, power):  #as0定义为通过真实二次路径的音频，as1定义为通过估计二次路径的音频
        e = d - y - as0 + as1  # disturbance-control signal
        return e ** 2 / (2 * power), e

    def _get_coeff_(self):  # 返回参数
        return self.Wc.detach().numpy()


def train_fxnlms_algorithm_mu(Model, Ref, Disturbance,as0,as1, Stepsize=0.0001):  # 原来Stepsize是0.0001

    bar = progressbar.ProgressBar(maxval=2 * Disturbance.shape[0], \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # 这里创建了一个进度条对象 bar，用于显示训练进度
    optimizer = optim.SGD([Model.Wc], lr=Stepsize)  # 创建一个随机梯度下降优化器

    bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]
    for itera in range(len_data):
        # Feedfoward
        xin = Ref[itera] #Ref is Fx
        dis = Disturbance[itera] #Disturbance is Dis
        AS0 = as0[itera] #as0定义为通过真实二次路径的音频
        AS1 = as1[itera] #as1定义为通过估计二次路径的音频
        y, power = Model.feedforward(xin)
        loss, e = Model.LossFunction(y, dis, AS0,AS1,power)


        # Progress shown
        bar.update(2 * itera + 1)

        # Backward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        Erro_signal.append(e.item())

        # Progress shown
        bar.update(2 * itera + 2)
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