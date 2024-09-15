import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import scipy.io as sio

#------------------------------------------------------------------------------
# Class: VSS_FxNLMS algorithm with initial coefficients determined by SFANC
#------------------------------------------------------------------------------
class VSS_FxNLMS():
    def __init__(self, Len, Ws, mu_min, mu_max, beta):
        self.Wc = torch.tensor(Ws, requires_grad=True)  # 初始化的Wc是全0 可以优化
        self.Xd = torch.zeros(1, Len, dtype=torch.float)  # 用于接受输入数据，初始化也是0
        self.mu_min = mu_min  # 最小步长
        self.mu_max = mu_max  # 最大步长
        self.beta = beta  # 步长调整参数
        self.power_prev = 1.0
        self.mu = mu_min

    def feedforward(self, Xf):
        self.Xd = torch.roll(self.Xd, 1, 1)
        self.Xd[0, 0] = Xf  # 这个是输入数据
        yt = self.Wc.double() @ self.Xd.t().double()
        power = self.Xd.double() @ self.Xd.t().double()  # FxNLMS different from FxLMS
        return yt, power

    def update_step_size(self, power):
        self.mu = self.mu_min + (self.mu_max - self.mu_min) * self.beta / (self.beta + power / self.power_prev)
        self.power_prev = power
        return self.mu

    def get_mu(self):
        return self.mu

    def update_weights(self, e, mu):
        self.Wc = self.Wc + mu * e * self.Xd  # 深拷贝替代原地操作

    def LossFunction(self, y, d, power):
        e = d - y  # disturbance-control signal
        return e ** 2 / (2 * power), e

    def _get_coeff_(self):  # 返回参数
        return self.Wc.detach().numpy()


#----------------------------------------------------------------
# Function: SFANC_VSS_FxNLMS
# Description: Using VSS_FxNLMS to optimize the control filter, the initial weights come from SFANC
#----------------------------------------------------------------
class SFANC_VSS_FxNLMS():
    # # 原始版本用这个 GA最优子代可以直接赋值给self.Current_Filter
    def __init__(self, MAT_FILE, fs):  # MAT_FILE是.mat文件的路径，fs是采样频率。
        self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE)  # torch.Size([15, 1024])
        Len = self.Wc.shape[1]
        self.fs = fs
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float) #原始版本 第一代
        # self.Current_Filter = read() # GA寻优之后的赋值版本 第三代 读取数据并且赋值

    # # 对init函数传入一个初始化的Wc GA寻优版本用这个 第二代
    # def __init__(self, MAT_FILE, fs, Wc0):  # MAT_FILE是.mat文件的路径，fs是采样频率。
    #     self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE)  # torch.Size([15, 1024])
    #     Len = self.Wc.shape[1]
    #     self.fs = fs
    #     self.Current_Filter = Wc0

    def noise_cancellation(self, Dis, Fx, filter_index, mu_min, mu_max, beta):
        Error = []
        j = 0
        model = VSS_FxNLMS(Len=1024,Ws=self.Current_Filter, mu_min=mu_min, mu_max=mu_max, beta=beta)
        mu = model.get_mu()
        optimizer = optim.SGD([model.Wc], lr=mu)  # Stepsize is learning_rate

        for ii, dis in enumerate(Dis):
            y, power = model.feedforward(Fx[ii])
            loss, e = model.LossFunction(y, dis, power)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Error.append(e.item())

            if (ii + 1) % self.fs == 0:
                print(j)
                # if self.Current_Filter[0].equal(self.Wc[filter_index[j]]) == False:
                if not np.array_equal(self.Current_Filter[0], self.Wc[filter_index[j]]):
                    # if prediction index is changed, change initial weights of FxNLMS
                    print('change the initial weights of VSS_FxNLMS')
                    self.Current_Filter = self.Wc[filter_index[j]].unsqueeze(0)
                    model = VSS_FxNLMS(Len=1024,Ws=self.Current_Filter, mu_min=mu_min, mu_max=mu_max, beta=beta)
                    mu = model.get_mu()
                    optimizer = optim.SGD([model.Wc], lr=mu)  # Stepsize is learning_rate
                j += 1
        return Error

    def Load_Pretrained_filters_to_tensor(self, MAT_FILE):  # Loading the pre-trained control filter from the mat file
        mat_contents = sio.loadmat(MAT_FILE)
        Wc_vectors = mat_contents['Wc_v']
        return torch.from_numpy(Wc_vectors).type(torch.float)





