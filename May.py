import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SFANC_FxNLMS_for_ANC import run_anc_algorithm
# from SFANC_VSS_FxNLMS_for_ANC import run_vss_anc_algorithm
from Read_the_data import write
from scipy.spatial.distance import cosine
import time
from math import exp


# Sphere函数作为适应度函数
def sphere_function(individual):
    return np.sum(individual**2)

known_data = pd.read_csv('./Trained models/3-1-1.csv', encoding='utf-8')#loading 精英个体
row,col = known_data.shape #获取已经获得的精英个体

# 目标函数（示例函数）??????????????????????????????????????????????????????????/
counter_objective = 0  # 目标函数计数器
best_fitness = 0  # 定义全局变量 best_fitness
def objective_function(data):

    global counter_objective
    global best_fitness  # 声明在函数内使用全局变量 best_fitness
    # sum_data = 0  # 初始化sum_data变量
    if counter_objective < 1000:  # 控制打印次数
        # print('适应度函数')
        counter_objective += 1
    # sum_data = np.sum(data)
    print(counter_objective)
    sum_data = sphere_function(data)
    return 1/sum_data

# 遗传算法相关参数
POP_SIZE = (10 + row) # 种群数量 5+x
# POP_SIZE = 13 # 种群数量,没有动态变化时候的数量
GENE_LENGTH = 1024  # 基因长度
MUTATION_RATE = 0.05  # 初始变异率0.0001-0.1
GENERATIONS = 200  # 迭代次数 50-200
Tmg = 12 #初始保留个数
Fderta = 0
j = 0

# # ----------------------------------------------初代目
# 生成随机个体
def generate_random_individual():
    return np.random.rand(GENE_LENGTH)

# 初始化种群
def init_population_r():
    population2 = []
    for _ in range(POP_SIZE):
        individual = generate_random_individual()
        population2.append(individual)
    return np.array(population2)
# ---------------------------------------------------
# 初始化种群
def init_population():
    formatted_data = format_known_data(known_data)
    return formatted_data

# 将已知数据格式化
def format_known_data(known_data):
    formatted_data = []
    data_array = known_data.to_numpy().flatten().astype(np.float64)
    num_individuals = len(data_array) // GENE_LENGTH

    for i in range(num_individuals):
        individual = data_array[i * GENE_LENGTH : (i + 1) * GENE_LENGTH]
        formatted_data.append(individual)

    formatted_data = np.array(formatted_data, dtype=np.float64)
    # print('格式化操作')
    return formatted_data

# 使用已知数据初始化种群
known_data_formatted = format_known_data(known_data)

# 评估适应度
counter_fitness = 0  # 适应度函数计数器
def fitness(individual):
    global counter_fitness
    global best_fitness  # 声明在函数内使用全局变量 best_fitness
    fitness_value = objective_function(individual)
    return fitness_value


# 选择适应度较高的个体
def select(population):
    return sorted(population,key=fitness,reverse = True)
    # return sorted(population,key=fitness)

# 交叉操作
def crossover(parent1, parent2):
    parent1 = parent1.reshape(1, -1)
    parent2 = parent2.reshape(1, -1)
    split_point = random.randint(0, GENE_LENGTH)
    child = np.hstack((parent1[:, :split_point], parent2[:, split_point:]))
    return child.flatten()

# 变异操作
def mutate(individual,MUTATION_RATE):
    for i in range(GENE_LENGTH):
        if random.random() < MUTATION_RATE:
            individual[i] = random.random()
    return individual



people_fitness_values = []
# 主要遗传算法函数
def genetic_algorithm():
    global best_fitness,Tmg,MUTATION_RATE,Fderta,j# 声明在函数内使用全局变量 best_fitness
    start_time = time.time()  # 记录开始时间
    population1 = init_population().astype(np.float64)  # 精英个体的载入
    population = np.vstack((population1,init_population_r()))
    # population = init_population_r() #没有动态变化时候的种族初始化
    run_count = 0  # 初始化计数器
    best_fitness_values = []  # 存储每次迭代中的最优个体的适应度值
    distances_to_best_individual = []  # 存储每个个体与最优个体之间的余弦距离
    print('---------------------------------')

    for i in range(GENERATIONS):
        fitness_values = [fitness(ind) for ind in population]
        people_fitness_values.append(fitness_values)  # 记录每个个体的适应度值
        population = select(population)
        best_individual = max(population, key=fitness)
        best_fitness_values.append(fitness(best_individual))  #将每次迭代中的最优个体的适应度值存储起来
        next_generation = population[:int(Tmg)]  # 选择适应度较高的前10个体直接保留,动态保留
        # next_generation = population[:8]  # 选择适应度较高的前10个体直接保留。没有动态变化时候的淘汰机制



        while len(next_generation) < POP_SIZE:

            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            parent3 = mutate(parent1, MUTATION_RATE)
            parent4 = mutate(parent2, MUTATION_RATE)
            next_generation.append(parent3)
            next_generation.append(parent4)
            # child = mutate(child,MUTATION_RATE)  #子代变异
            next_generation.append(child)
        population = np.array(next_generation, dtype=np.float64)
        # 计数器
        run_count += 1

        best_individual = max(population, key=fitness)
        best_fitness_values.append(fitness(best_individual))  # 将每次迭代中的最优个体的适应度值存储起来

        # 计算每个个体与最优个体之间的绝对余弦距离并存储
        distances = [abs(cosine(best_individual, ind)) for ind in population]
        distances_to_best_individual.append(distances)
        total_distance = sum(sum(distances_to_best_individual, [] ))
        Similar = total_distance/(POP_SIZE-1)
        #定义动态变异率和动态淘汰机制
        Tmg=Tmg*(1-Similar)
        MUTATION_RATE = MUTATION_RATE*exp(Similar) #动态变异率
        # MUTATION_RATE = 0.05 #没有动态变化时候的变异率

        # 添加停止条件：当上下一代之间的最优个体的适应度值变化率不足10%时停止遗传算法
        if len(best_fitness_values) >1:
            Fderta = (abs(best_fitness_values[-1]-best_fitness_values[-2]))/best_fitness_values[-2]

        if Fderta < 0.05:
            break

    best_individual = max(population, key=fitness)
    best_fitness = fitness(best_individual)
    best_fitness = max(best_fitness_values)
    end_time = time.time()  # 记录结束时间
    total_time = (end_time - start_time)/60
    total_miao = end_time - start_time
    print("上下一代之间的最优个体的适应度值变化率不足5%，停止遗传算法。")
    print(f"整个运行时间：{total_miao} 秒")
    print(f"整个运行时间：{total_time} 分钟")

    return best_individual, best_fitness,best_fitness_values,distances_to_best_individual

# 运行遗传算法
best_solution, best_fitness,best_fitness_values,distances_to_best_individual = genetic_algorithm()

print("最优解：", best_solution)
print("最优解的适应度：", max(best_fitness_values,key=abs))
print("最优解的适应度的倒数：", 1/(max(best_fitness_values,key=abs)))
write(best_solution)
