import torch
import torch.optim as optim
import librosa
import os
from Network import m6_res

# 设置训练数据的文件夹路径
data_folder = r"C:\Users\千骑卷平冈\Desktop\毕业设计\Synthesized_Dataset\Testing_data"

# 加载并处理训练数据
training_data = []
training_labels = []

for filename in os.listdir(data_folder):
    if filename.endswith(".wav"):
        # 加载音频文件
        audio_path = os.path.join(data_folder, filename)
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=0.01)  # 以单声道、22050Hz采样率加载5秒音频

        # 进行必要的预处理，例如提取特征、转换成张量等
        # 这里假设您使用MFCC特征作为输入
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = torch.tensor(mfccs)  # 转换成张量

        # 添加到训练数据列表
        training_data.append(mfccs)

        # 添加相应的标签（假设标签已经按顺序准备好）
        # 这里假设标签在文件名中，例如"speech_1.wav"对应标签1
        label = int(filename.split("_")[1].split(".")[0])
        training_labels.append(label)

# 转换训练数据列表为张量
training_data = torch.stack(training_data)
training_labels = torch.tensor(training_labels)

# 创建您提供的神经网络模型
model = m6_res

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    # 将模型设为训练模式
    model.train()

    # 前向传播
    outputs = model(training_data)
    loss = criterion(outputs, training_labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

print('训练完成')

# 保存模型
torch.save(model.state_dict(), 'trained_model.pth')
print('模型已保存')