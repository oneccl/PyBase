#!/user/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

from sklearn.linear_model import Ridge  # 岭回归模型

# 范数

import numpy as np

# 向量
x = np.array([3, -4])
print(x)          # [ 3 -4]
# 负无穷范数
LNegInf = np.linalg.norm(x, -np.inf)
print(LNegInf)    # 3.0
# L0范数
L0 = np.linalg.norm(x, 0)
print(L0)         # 2.0
# L1范数
L1 = np.linalg.norm(x, 1)
print(L1)         # 7.0
# L2范数
L2 = np.linalg.norm(x, 2)
print(L2)         # 5.0
# 无穷范数
LInf = np.linalg.norm(x, np.inf)
print(LInf)       # 4.0

# 正则化

import numpy as np
from sklearn import preprocessing as pp

data = np.array([[1, 2, 2], [2, 5, 4], [2, 3, 4]])
print(data)
# L1正则化
L1 = pp.normalize(data, norm="l1")
print(L1)
'''
[[0.2        0.4        0.4       ]
 [0.18181818 0.45454545 0.36363636]
 [0.22222222 0.33333333 0.44444444]]
'''
# L2正则化
L2 = pp.normalize(data, norm="l2")
print(L2)
'''
[[0.33333333 0.66666667 0.66666667]
 [0.2981424  0.74535599 0.59628479]
 [0.37139068 0.55708601 0.74278135]]
'''

# 岭回归的原理

from sklearn.datasets import make_regression

# 制作一个包含100个样本和10个特征的数据集
# 在10个特征中，8个特征具有信息并且有助于回归，而其余2个特征对目标变量没有任何影响（它们的真实系数为0）
# 数据是无噪声的，因此我们期望我们的回归模型能够准确地恢复真实系数ω
X, y, w = make_regression(
    n_samples=100, n_features=10, n_informative=8, coef=True, random_state=1
)

# 获取真实系数
print(w)

# 训练岭回归模型
import numpy as np
from sklearn.linear_model import Ridge                   # 岭回归
from sklearn.metrics import mean_squared_error as MSE    # 均方误差

# 使用不同的惩罚项参数λ（alpha）训练模型，λ（alpha）用于控制正则化强度
clf = Ridge()

# 生成200个在-3~4区间均匀分布的alpha值
alphas = np.logspace(-3, 4, 200)

# 用于存储不同的正则化强度训练模型的回归系数
coefs = []
# 用于存储不同的正则化强度训练模型的均方误差
errors_coefs = []
# 使用不同的正则化强度训练模型
# 对于每个经过训练的模型计算真实系数ω与模型找到的系数之间的均方误差
for a in alphas:
    clf.set_params(alpha=a).fit(X, y)
    coefs.append(clf.coef_)
    errors_coefs.append(MSE(clf.coef_, w))

# 绘制训练系数和均方误差
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "SimHei"

alphas = pd.Index(alphas, name="alpha")
coefs = pd.DataFrame(coefs, index=alphas, columns=[f"Feature {i}" for i in range(10)])
errors = pd.Series(errors_coefs, index=alphas, name="mse")
# 画板大小
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
# 可视化
coefs.plot(
    ax=axs[0],
    logx=True,
    title="不同正则化强度与岭回归系数的关系",
)
axs[0].set_ylabel("岭回归系数")
errors.plot(
    ax=axs[1],
    logx=True,
    title="不同正则化强度与岭回归均方误差的关系",
)
_ = axs[1].set_ylabel("岭回归均方误差")
plt.show()


# 使用岭回归模型拟合房价

from sklearn.model_selection import train_test_split   # 数据集划分
from sklearn.linear_model import LinearRegression      # 线性回归模型
from sklearn.linear_model import Ridge                 # 岭回归模型

# 将数据集划分为训练集和测试集（随机采样20%的数据作为测试样本，其余作为训练样本）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 1）使用线性回归模型拟合房价
lr = LinearRegression()
# 在训练集上拟合模型
lr.fit(X_train, y_train)
# 最优解
print(lr.coef_)
print(lr.intercept_)
# 拟合程度R^2
print(lr.score(X_test, y_test))

# 2）使用岭回归模型拟合房价
ridge = Ridge(alpha=1)
# 训练模型
ridge.fit(X_train, y_train)
# 最优解
print(ridge.coef_)
print(ridge.intercept_)
# 拟合程度R^2
print(ridge.score(X_test, y_test))

# 拟合程度/相关系数（偏差）随alpha的变化
from sklearn.model_selection import cross_val_score   # 交叉验证

# 5折交叉验证下，岭回归与线性回归的拟合程度随alpha的变化
alpha_range = np.arange(1, 1001, 10)
clf, linear = [], []
for alpha in alpha_range:
    ridge = Ridge(alpha=alpha)
    lr = LinearRegression()
    clf_score = cross_val_score(ridge, X, y, cv=5, scoring='r2').mean()
    linear_score = cross_val_score(lr, X, y, cv=5, scoring="r2").mean()
    clf.append(clf_score)
    linear.append(linear_score)
plt.plot(alpha_range, clf, label='Ridge')
plt.plot(alpha_range, linear, c='blue', label='LR')
plt.xlabel('alpha')
plt.ylabel(r'$R^2$', rotation=0)
plt.legend()
plt.show()

# 查找R2极大值点对应的最优alpha：
# np.argmax()用于查找clf数组中最大值对应的索引
max_idx = np.argmax(clf)
print(max_idx)         # 17
max_x, max_y = alpha_range[max_idx], clf[max_idx]
print(max_x, max_y)    # 171 0.4982332192004183

# 岭回归案例（波士顿房价预测）

from sklearn.linear_model import RidgeCV                 # 交叉验证岭回归

# 给定alpha的变化范围
alpha_range = np.arange(1, 1001, 10)
# 交叉验证岭回归模型
ridge_cv = RidgeCV(alphas=alpha_range, scoring='r2', cv=5)
# 拟合模型
ridge_cv.fit(X, y)
# 获取最优alpha值
print(ridge_cv.alpha_)         # 171
# 最优alpha值时的模型相关系数评分R^2
print(ridge_cv.best_score_)    # 0.4982332192004183

# 划分训练集和测试集
from sklearn.metrics import mean_squared_error as MSE    # 均方误差

# 将数据集划分为训练集和测试集（随机采样20%的数据作为测试样本，其余作为训练样本）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 使用交叉验证岭回归模型搜索最优alpha
alpha_range = np.arange(1, 1001, 10)
# 交叉验证岭回归模型
ridge_cv = RidgeCV(alphas=alpha_range, scoring='neg_mean_squared_error', cv=5)
# 使用训练数据拟合模型
ridge_cv.fit(X_train, y_train)
# 获取最优alpha值
alpha_best = ridge_cv.alpha_
print(alpha_best)         # 1

# 使用最佳alpha值建模
ridge = Ridge(alpha=alpha_best)
# 使用最佳正则化系数训练模型
ridge.fit(X_train, y_train)
# 回归系数和截距
print(ridge.coef_)
print(ridge.intercept_)
# 在测试集上进行预测
y_pred = ridge.predict(X_test)

# 模型评估
# 拟合程度、相关系数R^2
print(ridge.score(X_test, y_test))
# 均方误差MSE
print(MSE(y_test, y_pred))
# 均方根误差RMSE
print(MSE(y_test, y_pred, squared=False))


# 岭回归与Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # 归一化

# 封装Pipeline，制作一个岭回归的管道
def RidgeRegression(alpha):
    return Pipeline([
        ("std_sca", StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])

# 交叉验证岭回归模型
ridge = RidgeRegression(alpha_best)
# 在训练集上拟合模型
ridge.fit(X_train, y_train)
# 获取模型的系数、截距
print(ridge.named_steps['ridge'].coef_)
print(ridge.named_steps['ridge'].intercept_)
# 在测试集上进行预测
y_pred = ridge.predict(X_test)

# 模型评估
# 拟合程度、相关系数R^2
print(ridge.score(X_test, y_test))
# 均方误差MSE
print(MSE(y_test, y_pred))
# 均方根误差RMSE
print(MSE(y_test, y_pred, squared=False))


# 多项式岭回归

from sklearn.preprocessing import PolynomialFeatures     # 多项式回归

np.random.seed(1)
# 生成100个样本
m = 100
# 从一个均匀分布的区域中随机采样，在-3~3区间生成均匀分布的100个元素，返回数组
x = np.random.uniform(-3, 3, size=m)
# 从标准正态（高斯）分布中抽取随机样本，给拟合值添加噪声
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size=m)
# 散点图可视化
plt.scatter(x, y, c='green', s=10, label='original data')
plt.legend(loc='upper left')

# 转换为矩阵形式
X = x.reshape(-1, 1)

# 封装Pipeline，制作一个多项式岭回归的管道
def PolynomialRidge(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_sca", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

# 不同degree线的颜色
colors = ['#1f77b4', 'teal', 'darkorange', '#FA06F3']
# 给定4种不同的degree取值：1、2、10、100
# 使用多项式岭回归拟合模型并作图
for idx, degree in enumerate([1, 2, 10, 100]):
    # 多项式岭回归模型
    pr = PolynomialRidge(degree)
    # 训练模型
    pr.fit(X, y)
    # 预测
    y_pred = pr.predict(X)
    # 可视化
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color=colors[idx], linewidth=1, label=f'degree {degree}')
    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('y')

plt.show()




