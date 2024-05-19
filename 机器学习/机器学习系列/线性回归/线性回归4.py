#!/user/bin/env python3
# -*- coding: utf-8 -*-


# Python-Web
# https://www.cnblogs.com/cleven/p/10858016.html
# https://blog.csdn.net/shifengboy/article/details/114274271
# https://blog.csdn.net/wly55690/article/details/131683846

# Scikit-Learn
# TensorFlow
# PyTorch


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression        # 线性回归
# from sklearn.preprocessing import PolynomialFeatures     # 多项式回归
# from sklearn.pipeline import Pipeline                    # 流程管道
# from sklearn.preprocessing import StandardScaler         # 标准化、归一化
# from sklearn.metrics import mean_squared_error as MSE    # 均方误差
# from sklearn.model_selection import train_test_split     # 划分训练测试集

# 梯度图
# import matplotlib.pylab as plt
# import numpy as np
#
# # 定义二次函数
# def f(x):
#     return pow((x-5), 2) + 2
#
# # 曲线：f(x)=x^2 -10x + 27
# # 求导（切线斜率）：f(x)' = 2x - 10
# # (1.5,14.25)  在x=1.5处的斜率：-7  y=-7x+b  b=24.75  y=-7x+24.75
# # (2.5,8.25)   在x=2.5处的斜率：-5  y=-5x+b  b=20.75  y=-5x+20.75
# # (3.5,4.25)
#
# # 定义一次函数（切线）
# def y1(x):
#     return -7*x+24.75
#
# def y2(x):
#     return -5*x+20.75
#
# # 设置画布大小
# plt.figure(figure=(10, 5))
#
# # 设置x轴的范围
# x = np.linspace(0, 10, 100)
#
# # 绘制二次曲线
# plt.plot(x, f(x), color='red', linewidth=1)
#
# # 绘制切线1
# plt.plot(x, y1(x), color='green', ls='--', linewidth=1)
# # 切点1
# plt.scatter(1.5, 14.25, color="green")
# # 标记虚线
# plt.plot([0, 1.5], [14.25, 14.25], c='green', linestyle='--', linewidth=1)
# plt.plot([1.5, 1.5], [0, 14.25], c='green', linestyle='--', linewidth=1)
# # 对应x，y坐标标签
# plt.text(1.4, -1.4, r'$\omega_1$')
# plt.text(-1, 14, r'$L(\omega_1)$')
# # 标注点A
# plt.text(1.8, 14.4, 'A')
#
# # 绘制切线2
# plt.plot(x, y2(x), ls='--', linewidth=1)
# # 切点2
# plt.scatter(2.5, 8.25)
# # 标记虚线
# plt.plot([0, 2.5], [8.25, 8.25], c='#1f77b4', linestyle='--', linewidth=1)
# plt.plot([2.5, 2.5], [0, 8.25], c='#1f77b4', linestyle='--', linewidth=1)
# # 对应x，y坐标标签
# plt.text(2.4, -1.4, r'$\omega_2$')
# plt.text(-1, 8, r'$L(\omega_2)$')
# # 标注点B
# plt.text(2.8, 8.4, 'B')
#
# # 再标注一个点C
# plt.scatter(3.5, 4.25, color='blue')
# plt.text(3.8, 4.4, 'C')
#
# # x、y轴坐标限制（解决两个坐标轴0点不重合问题）
# plt.xlim([0, 10])
# plt.ylim([0, 25])
#
# # 添加坐标轴标签
# plt.xlabel(r'$\omega$', loc='right')
# # rotation=0：设置y轴标签水平显示（默认90）
# plt.ylabel(r'$L$', loc='top', rotation=0)
# plt.show()

# 梯度下降法的实现
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 假设损失函数（L）
# def L(w):
#     return pow((w-5), 2) + 2
#
# # 设置x轴的范围（ω）
# w = np.linspace(0, 10, 100)
#
# # 可视化
# plt.plot(w, L(w))
# plt.show()
#
# # 定义梯度（变化率）
# def dL(w):
#     # 对损失函数（L）求导
#     return 2*w - 10
#
# # 梯度下降
# # 步长（默认）
# alpha = 0.8
# # 初始点（默认）
# theta = 0.0
# # 梯度的最小边界
# diff = 1e-8
#
# theta_history = [theta]
#
# # 求解迭代梯度（变化率）
# while True:
#     # 求解梯度
#     gradient = dL(theta)
#     # 记录上一个theta值
#     last_theta = theta
#     # 寻找下一个theta值，即当前的theta值加上步长乘以损失函数递减的变化率
#     theta = theta - alpha*gradient
#     # 保存历史每次移动的初始点
#     theta_history.append(theta)
#     # 当新找到的theta值与上一个theta值之差小于1e-8时，表明此时变化率已经趋于0了，新的theta值可以使损失函数达到极小值
#     if abs(L(theta) - L(last_theta)) < diff: break
#
# print(len(theta_history))     # 49
#
# plt.plot(w, L(w))
# plt.plot(np.array(theta_history), L(np.array(theta_history)), color='r', marker='+')
# plt.show()

# 步长对解的求解的影响

import numpy as np
import matplotlib.pyplot as plt

# # 步长（默认）
# alpha = 0.1
# # 初始点（默认）
# theta = 0.0
# # 梯度的最小边界
# diff = 1e-8
#
# # 记录每次移动后的历史自变量值
# theta_history = [theta]
#
# # 求解迭代梯度（变化率）
# while True:
#     # 求解梯度
#     gradient = dL(theta)
#     # 记录上一个theta值
#     last_theta = theta
#     # 寻找下一个theta值，即当前的theta值加上步长乘以损失函数递减的变化率
#     theta = theta - alpha*gradient
#     # 保存历史每次移动的初始点
#     theta_history.append(theta)
#     # 当新找到的theta值与上一个theta值之差小于1e-8时，表明此时变化率已经趋于0了，新的theta值可以使损失函数达到极小值
#     if abs(L(theta) - L(last_theta)) < diff: break
#
# print(len(theta_history))     # 49
#
# # 可视化
# plt.plot(w, L(w))
# plt.plot(np.array(theta_history), L(np.array(theta_history)), color='r', marker='+')
# plt.show()

# 线性回归中的梯度下降法

# 定义代价函数（损失函数）
def cost(X, Y, m, theta):
    diff = np.dot(X, theta) - Y
    return (1 / (2 * m)) * np.dot(diff.transpose(), diff)


# 定义代价函数的梯度函数
def grad(X, Y, m, theta):
    diff = np.dot(X, theta) - Y
    return (1 / m) * np.dot(X.transpose(), diff)


# 梯度下降迭代算法
def gradient_descent(X, Y, m, theta, alpha, n_iters=1e4, diff=1e-8):
    i_iter = 0
    while i_iter < n_iters:
        gradient = grad(X, Y, m, theta)
        last_theta = theta
        theta = theta - alpha * gradient
        if abs(cost(X, Y, m, theta) - cost(X, Y, m, last_theta)) < diff: break
        i_iter += 1
    print(f"迭代次数: {i_iter}")
    return theta


# 可视化
def plot(X, Y, theta):
    # 横坐标x的取值
    x = X[:, 1:].flatten()
    y = theta[0] + theta[1]*x
    plt.scatter(x, Y, s=10, c="red", marker="o")
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# 生成数据区间在0~10包含100个元素的均匀分布序列，返回数组
x = np.linspace(0, 10, 100)
# 参数
k = 1.25
b = 3.5
# 目标预测函数（理论拟合结果）
y = np.polyval([k, b], x)
# 给拟合值添加噪音(正态分布随机数)
yi = y + np.random.normal(size=100)
# 数据样本只有一个特征，转换为10行1列的2D数组
X = np.reshape(x, (-1, 1))

# 样本大小
m = 100
# 生成X0列，生成一个m行1列的向量（X0列全是1）
X0 = np.ones((m, 1))
# print(X0)
# 生成样本数据：由于样本数据X只有一个特征，则X即为X1列
# np.hstack()：水平堆叠数组，用于将两个或多个数组沿水平轴（按列）连接起来
X = np.hstack([X0, X])
# print(X)
# 初始化theta向量为元素全为0的向量（数量为X行数）
theta = np.zeros(X.shape[1])
# print(theta)
# 学习率（步长）
alpha = 0.01

# 梯度下降迭代
optimal_theta = gradient_descent(X, yi, m, theta, alpha)
print(optimal_theta)

# 可视化
plot(X, yi, optimal_theta)

# 梯度下降法求解线性回归案例（波士顿房价预测）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# 数据集简介
COLS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# 波士顿房价数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston_df = pd.DataFrame(data)
boston_df.columns = COLS
# print(boston_df.head().to_string())
target = raw_df.values[1::2, 2]
boston_df['MEDV'] = target
# print(boston_df.head().to_string())
'''
      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  MEDV
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98  24.0
1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14  21.6
2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03  34.7
3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94  33.4
4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33  36.2
'''

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.preprocessing import StandardScaler  # 特征工程：标准化
from sklearn.model_selection import train_test_split  # 数据集划分

# X = boston_df.iloc[:, 0: 13]
# y = np.array(boston_df.iloc[:, -1])

# 将数据集划分为训练集和测试集（随机采样20%的数据作为测试样本，其余作为训练样本）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # 数据标准化（只需处理特征）
# # 初始化对特征和目标值的标准化器
# ss = StandardScaler()
# # 分别对训练和测试数据的特征进行标准化处理
# X_train = ss.fit_transform(X_train)
# X_test = ss.fit_transform(X_test)


# # 封装使用梯度下降法求解线性回归的最优解的类（训练数据、曲线拟合）
# class GradLinearRegression(object):
#
#     def __init__(self):
#         self._theta = None
#         self.coef_ = None
#         self.intercept_ = None
#
#     def fit_grad(self, X_train, y_train, alpha=0.01, n_iters=1e4):
#
#         # 定义代价函数（损失函数）
#         def cost(X, Y, m, theta):
#             diff = np.dot(X, theta) - Y
#             return (1 / (2 * m)) * np.dot(diff.transpose(), diff)
#
#         # 定义代价函数的梯度函数
#         def grad(X, Y, m, theta):
#             diff = np.dot(X, theta) - Y
#             return (1 / m) * np.dot(X.transpose(), diff)
#
#         # 梯度下降迭代算法
#         def gradient_descent(X, Y, m, theta, alpha, n_iters, diff=1e-8):
#             i_iter = 0
#             while i_iter < n_iters:
#                 gradient = grad(X, Y, m, theta)
#                 last_theta = theta
#                 theta = theta - alpha * gradient
#                 if abs(cost(X, Y, m, theta) - cost(X, Y, m, last_theta)) < diff: break
#             return theta
#
#         # 构建X
#         X = np.hstack([np.ones((len(X_train), 1)), X_train])
#         # 初始化theta向量为元素全为0的向量
#         theta = np.zeros(X.shape[1])
#
#         self._theta = gradient_descent(X, y_train, len(X), theta, alpha, n_iters)
#         self.intercept_ = self._theta[0]
#         self.coef_ = self._theta[1:]
#
#         return self

from sklearn.linear_model import LinearRegression  # 线性回归模型

# # A、使用线性回归模型拟合房价
# lr = LinearRegression()
# # 在训练集上拟合模型
# lr.fit(X_train, y_train)
# # 最优解
# print(lr.coef_)
# print(lr.intercept_)
# # 拟合程度R^2
# print(lr.score(X_test, y_test))

# # B、使用梯度下降法线性回归模型拟合房价
# # 计时
# start = time.perf_counter()
# glr = GradLinearRegression()
# # 在训练集上拟合模型
# glr.fit_grad(X_train, y_train)
# # 最优解
# print(glr.coef_)
# print(glr.intercept_)
# print(time.perf_counter()-start)    # 0.1291685999603942

# Scikit-Learn梯度下降法

from sklearn.linear_model import SGDRegressor  # 随机梯度下降法中的线性回归模型
from sklearn.metrics import mean_squared_error as MSE  # 均方误差

# # C、使用随机梯度下降法训练波士顿房价数据集并预测
# # 计时
# start = time.perf_counter()
# sgd_reg = SGDRegressor()
# # 在训练集上拟合模型
# sgd_reg.fit(X_train, y_train)
# # 最优解
# print(sgd_reg.coef_)
# print(sgd_reg.intercept_)
# print(time.perf_counter()-start)    # 0.0019124000100418925
#
# # 在测试集上进行预测
# y_pred = sgd_reg.predict(X_test)
#
# # 决定系数R^2
# print(sgd_reg.score(X_test, y_test))           # 0.7558329227008684
# # 均方误差MSE
# print(MSE(y_test, y_pred))                     # 19.370077456157663
# # 均方根误差RMSE
# print(MSE(y_test, y_pred, squared=False))      # 4.401145016488058


# 管道的使用

# from sklearn.pipeline import Pipeline
# from sklearn.pipeline import make_pipeline
#
# # D、使用随机梯度下降法训练波士顿房价数据集并预测（管道优化）
# # 管道是由包含（键，值）对的列表构建的，其中键是包含此步骤名称的字符串，而值是估计器对象
# pipe = Pipeline([("std_sca", StandardScaler()), ('sgd_reg', SGDRegressor())])
# # 功能函数make_pipeline是构造管道的简写。它使用可变数量的估计器并返回管道，而且自动填充名称(类名小写)
# # pipe = make_pipeline(StandardScaler(), SGDRegressor())
# # 在训练集上拟合模型
# pipe.fit(X_train, y_train)
# # 最优解
# # 获取模型的系数、截距
# print(pipe.named_steps['sgd_reg'].coef_)
# print(pipe.named_steps['sgd_reg'].intercept_)
# # 预测值
# y_pred = pipe.predict(X_test)
#
# # 评估
# # 决定系数R^2
# print(pipe.score(X_test, y_test))
# # 均方误差MSE
# print(MSE(y_test, y_pred))



