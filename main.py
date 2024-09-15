import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib

# x值为分贝值，需要转换为数值
x_values_db = ['-10db', '-5db', '0db', '5db', '10db']
# 将分贝值转换为数值，这里我们假设分贝值是线性的，即每5db一个单位
x_values = [-10, -5, 0, 5, 10]


# 指定字体路径（请根据你的字体文件实际位置进行修改）
font_path = "path/to/your/font/SimHei.ttf"

# 创建一个字体属性对象，指定中文字体
font_prop = FontProperties(fname=font_path, size=14)

# 添加字体到matplotlib的字体列表
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 两组y值数据
y1 = [1.24, 1.38, 1.77, 1.86, 2.19]
y2 = [1.42, 1.84, 2.15, 2.35, 2.82]

# 绘制线图
plt.figure(figsize=(10, 5))  # 设置图形的大小
plt.plot(x_values, y1, marker='o', label='优化前')  # 第一组数据的线图
plt.plot(x_values, y2, marker='x', label='优化后')  # 第二组数据的线图

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('PESQ得分')


# 显示网格
plt.grid(True)

# 显示图形
plt.show()