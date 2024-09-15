import pandas as pd
import torch
import csv

def read(name):
    best_solution = pd.read_excel(f'./Trained models/{name}.xlsx')
    # 提取数值数据
    data = best_solution.values
    # 转换为PyTorch张量
    tensor_data = torch.tensor(data)
    return tensor_data

# 以下是测试
# t=read('weights_list')
# print(t)
def write(data):
    # 打开 CSV 文件并追加数据
    with open('./Trained models/3-1-1.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

