#!/user/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso  # 套索回归

# from sklearn.datasets import make_regression
#
# # 制作一个包含100个样本和100个特征的高维数据集，其中噪声样本数占10
# X, y = make_regression(
#     n_samples=100, n_features=100, noise=10
# )
#
# # 生成区间从10的-2到10的3次方平均包含200个值的数组
# alphas = np.logspace(-2, 3, 200)
# # 构造空列表，用于存储模型的偏回归系数
# coefs = []
# for alpha in alphas:
#     lasso = Lasso(alpha=alpha, max_iter=10000)
#     lasso.fit(X, y)
#     coefs.append(lasso.coef_)
#
# # 可解决负号无法显示的问题
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#
# # 绘制Alpha与回归系数的关系
# plt.plot(alphas, coefs)
# # 对X轴作对数变换
# plt.xscale('log')
# # 设置折线图x轴和y轴标签
# plt.xlabel('Alpha(λ)')
# plt.ylabel('权重系数')
# # 限制X、Y轴数值范围
# plt.xlim(pow(10, -1), pow(10, 2.5))
# plt.ylim(0, 100)
# # 显示图形
# plt.show()

import pandas as pd
from sklearn.linear_model import Lasso                  # 套索回归
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.metrics import mean_squared_error as MSE   # 均方误差
from sklearn.datasets import fetch_california_housing as fch

# 加州房价数据集
# 加州位于美国西南部，是美国经济较为发达、人口较为密集的行政区之一。加州的房价受到房龄、人口规模、地理位置等多种因素影响。本案例将使用加州房价数据集，首先对数据进行探索性分析和数据预处理，然后再依次进行特征提取、标准化、特征选择，最后建立Lasso回归模型对房价进行预测
# 加州房价数据集包含了1990年加州的所有普查区域，共计20640个实例。每个实例包含了8个特征和1个目标标签变量。由于该数据集提供了每个家庭的平均房间数和卧室数，因此对于一些家庭较少、空房屋较多的区域，如度假村，一些特征可能会取得意外大的值。这些特征被用来预测该地区房屋价格的中位数

fch_df = pd.DataFrame(fch().data)
fch_df.columns = fch().feature_names
fch_df['MEDV'] = fch().target
# print(fch_df.head().to_string())
'''
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude   MEDV
0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23  4.526
1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22  3.585
2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24  3.521
3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25  3.413
4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25  3.422
'''

# 是否含有空值（该数据集已经经过官方处理了，包含空值的字段已被移除）
# print(fch_df.isnull().sum())      # 无空值
# print(fch_df.shape)               # (20640, 9)

# Lasso回归基本使用
X = pd.DataFrame(fch().data)
y = fch().target

# 将数据集划分为训练集和测试集（随机采样20%的数据作为测试样本，其余作为训练样本）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # 使用Lasso模型拟合房价
# lasso = Lasso(alpha=0.001)
# # 训练模型
# lasso.fit(X_train, y_train)
# # 最优解
# print(lasso.coef_)
# print(lasso.intercept_)
# # 预测
# y_pred = lasso.predict(X_test)
#
# # 拟合程度（是否拟合了足够的信息）
# print(lasso.score(X_test, y_test))   # 0.5989352234266341
# # 均方误差（是否预测了正确的数值）
# print(MSE(y_test, y_pred))           # 0.564140207307865

# 交叉验证Lasso回归
from sklearn.linear_model import LassoCV      # 交叉验证lasso回归

# # 建立Lasso进行alpha选择的范围
# # 范围从10的-7到2次方，200个alpha值
# alpha_range = np.logspace(-7, 2, 200)
# # 使用10折交叉验证Lasso回归模型
# lasso_cv = LassoCV(alphas=alpha_range, cv=10, max_iter=1000)
# # 使用训练数据拟合模型
# lasso_cv.fit(X_train, y_train)
# # 获取最优alpha值
# alpha_best = lasso_cv.alpha_
# print(alpha_best)          # 0.002437444150122222
# # 最优alpha值时的迭代次数
# print(lasso_cv.n_iter_)    # 112
# # 最优alpha值时的模型均方误差MSE
# print(lasso_cv.mse_path_.flatten()[lasso_cv.n_iter_])    # 1.3473315980350162
# # 最优alpha值时的模型相关系数评分R^2
# print(lasso_cv.score(X_test, y_test))    # 0.6109308964346865

# # LassoCV默认参数配置获取最优alpha
# lasso_cv = LassoCV(eps=0.0001, n_alphas=200, cv=10, max_iter=1000)
# # 使用训练数据拟合模型
# lasso_cv.fit(X_train, y_train)
# # 获取最优alpha值
# print(lasso_cv.alpha_)    # 0.0034993404718341313
# # 最优alpha值时的模型相关系数评分R^2
# print(lasso_cv.score(X_test, y_test))    # 0.6104623584124887

# 使用交叉验证的LassoCV参数与RidgeCV略有不同，这是因为Lasso对于alpha（λ）的取值更加敏感，因此LassoCV对λ取值范围的处理更加细腻，可以通过参数eps规定正则化路径以及路径中的个数（参数n_alphas），Sklearn会自动计算并生成λ的取值以供交叉验证类使用。当然也可以通过指定alpha范围实现

# # 基于最佳的alpha（λ）建模
# # 使用Lasso模型拟合房价
# lasso = Lasso(alpha=10**4)
# # 训练模型
# lasso.fit(X_train, y_train)
# # 最优解
# print(lasso.coef_)
# # print(lasso.intercept_)
# # 预测
# y_pred = lasso.predict(X_test)
#
# # 模型评估
# # 拟合程度（是否拟合了足够的信息）
# print(lasso.score(X_test, y_test))   # 0.6102870995620837
# # 均方误差（是否预测了正确的数值）
# print(MSE(y_test, y_pred))           # 0.5172694619650446

# # 根据模型预测
# # 可视化
# # 设置画板尺寸
# plt.figure(figsize=(15, 6))
# # 设置字体
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.title('加州房价预测曲线与实际曲线对比图', fontsize=15)
# # x轴数据（由于数据较多，这里只取测试集的前100个数据）
# x = range(len(y_test))[0:100]
# # 实际价格曲线
# plt.plot(x, y_test[0:100], color='r', label='实际房价')
# # 预测价格曲线
# plt.plot(x, y_pred[0:100], color='g', ls='--', label='预测房价')
# # 显示图例
# plt.legend(fontsize=12, loc=1)
# plt.show()

# 线性回归、岭回归与Lasso回归的比较

# from sklearn.linear_model import LinearRegression, Ridge, Lasso
#
# # A、线性回归模型
# lr = LinearRegression()
# # 使用训练集训练
# lr.fit(X_train, y_train)
# # 系数
# print(lr.coef_)
#
# # B、岭回归模型
# # 使用较大的L2正则项系数，方便观察模型的系数的变化
# ridge = Ridge(alpha=10**4)
# # 在训练集上训练模型
# ridge.fit(X_train, y_train)
# # 系数
# print(ridge.coef_)
#
# # C、Lasso回归模型
# # 使用较大的L1正则项系数，方便观察模型的系数的变化
# lasso = Lasso(alpha=10**4)
# # 训练模型
# lasso.fit(X_train, y_train)
# # 最优解
# print(lasso.coef_)
#
# # 绘制每个特征与特征对应的回归系数之间的关系
# plt.figure(figsize=(10, 6))
# plt.plot(range(1,9), (lr.coef_*100).tolist(), color="red", label="LR")
# plt.plot(range(1,9), (ridge.coef_*100).tolist(), color="green", label="Ridge")
# plt.plot(range(1,9), (lasso.coef_*100).tolist(), color="blue", label="Lasso")
# plt.plot(range(1,9), [0]*8, color="grey", linestyle="--")
# plt.ylabel("Weight ω")
# plt.xlabel("Feature x")
# plt.legend()
# plt.show()


# Scikit-Learn Lasso回归与Pipeline

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler  # 归一化
#
# # 封装Pipeline，制作一个Lasso回归的管道
# def LassoRegression(alpha):
#     return Pipeline([
#         ("std_sca", StandardScaler()),
#         ("lasso", Lasso(alpha=alpha))
#     ])
#
# # 基于最优的alpha建模
# lasso = LassoRegression(alpha_best)
# # 在训练集上训练模型
# lasso.fit(X_train, y_train)
# # 获取模型的系数、截距
# print(lasso.named_steps['lasso'].coef_)
# print(lasso.named_steps['lasso'].intercept_)
# # 在测试集上预测
# y_pred = lasso.predict(X_test)
#
# # 模型评估
# # 拟合程度、相关系数R^2
# print(lasso.score(X_test, y_test))
# # 均方误差MSE
# print(MSE(y_test, y_pred))

# 最小角回归
from sklearn.linear_model import LassoLars      # 最小角回归模型
from sklearn.linear_model import LassoLarsCV    # 交叉验证最小角回归模型

# # 使用LassoLarsCV获取最优alpha
# ll_cv = LassoLarsCV(eps=0.0001, cv=10, max_iter=1000)
# # 使用训练数据拟合模型
# ll_cv.fit(X_train, y_train)
# # 获取最优alpha值
# alpha_best = ll_cv.alpha_
# print(alpha_best)    # 0.0020195287698618536
#
# # 初始化Lasso回归器，使用最小角回归法
# ll = LassoLars(alpha=alpha_best)
# # 拟合线性模型
# ll.fit(X_train, y_train)
# # 权重系数
# print(ll.coef_)
# # 截距
# print(ll.intercept_)
# # 预测
# y_pred = ll.predict(X_test)
#
# # 模型评估
# # 拟合程度（是否拟合了足够的信息）
# print(ll.score(X_test, y_test))      # 0.600563140522733
# # 均方误差（是否预测了正确的数值）
# print(MSE(y_test, y_pred))           # 0.546772691420003

# ElasticNet回归（弹性网络回归）

# 前面我们详细介绍了岭回归与Lasso回归两种正则化的方法，当多个特征存在相关时，Lasso回归可能只会随机选择其中一个，岭回归则会选择所有的特征
# Ridge回归使用L2正则化，Lasso回归则使用L1正则化。正则化的目的是防止过拟合，正则化的本质是约束（限制）要模型参数（系数），以使我们的模型性能达到最优
# 弹性网络（ElasticNet）回归是一种使用了L1正则化和L2正则化的线性回归模型。这种组合允许拟合到一个只有少量参数是非零稀疏的模型，就像Lasso回归一样，但它同时保留了一些Ridge回归的性质
# ElasticNet将Lasso和Ridge组成一个具有两种惩罚因素的单一模型：一个与L1正则化成比例，另外一个与L2正则化成比例。结合这两种方式所得到的模型就像纯粹的Lasso回归一样稀疏，但同时具有与岭回归提供的一样的L2正则化能力

# ElasticNet回归的损失函数定义为：
# $$L(\omega)={\sum_{i=1}^m}(y_i-y)^2+\lambda {\sum_{i=1}^m}|ω_i|=||Y-X\omega||^2+\lambda\rho {||\omega||_1} + $$

from sklearn.linear_model import ElasticNet      # 弹性网络回归模型
from sklearn.linear_model import ElasticNetCV    # 交叉验证弹性网络回归模型

# 使用ElasticNetCV获取最优alpha，L1正则化与L2正则化各占一半
en_cv = ElasticNetCV(eps=0.0001, l1_ratio=0.5, cv=10, max_iter=1000)
# 使用训练数据拟合模型
en_cv.fit(X_train, y_train)
# 获取最优alpha值
alpha_best = en_cv.alpha_
print(alpha_best)    # 0.006062898814827176

# 初始化ElasticNet回归器（坐标轴下降法）
en = ElasticNet(alpha=alpha_best)
# 拟合线性模型
en.fit(X_train, y_train)
# 权重系数
print(en.coef_)
# 截距
print(en.intercept_)
# 预测
y_pred = en.predict(X_test)

# 模型评估
# 拟合程度（是否拟合了足够的信息）
print(en.score(X_test, y_test))      # 0.6114705180467421
# 均方误差（是否预测了正确的数值）
print(MSE(y_test, y_pred))           # 0.5077069487376945


