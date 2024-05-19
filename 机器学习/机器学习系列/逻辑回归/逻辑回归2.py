
# 多项式特征逻辑回归

from matplotlib.colors import ListedColormap

# 决策边界分类轮廓填充
def decision_boundary_fill(model, axis):
    # np.meshgrid()：定义X/Y坐标轴上的起始点和结束点以及点的密度，返回这些网格点的X和Y坐标矩阵
    X0, X1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    # ravel()：将高维数组降为一维数组
    # np.c_[]：将两个数组以列的形式拼接起来形成矩阵，这里将上面每个网格点的X和Y坐标组合
    X_grid_matrix = np.c_[X0.ravel(), X1.ravel()]
    # 通过训练好的逻辑回归模型，预测平面上这些网格点的分类
    y_grid_pred = model.predict(X_grid_matrix)
    y_pred_matrix = y_grid_pred.reshape(X0.shape)
    # plt.contourf(X,Y,Z)：绘制等高线，cmap用于设置填充轮廓，默认为viridis，还可设置为热力图色彩plt.cm.hot或自定义；alpha用于设置填充透明度
    # ListedColormap()：自定义填充色彩列表
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(X0, X1, y_pred_matrix, alpha=0.5, cmap=custom_cmap)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 为什么使用多项式

# # 随机数种子，只需设置一次，设置后只要种子不变，每次生成相同的随机数
# np.random.seed(666)
# # 构建均值为0，标准差为1（标准正态分布）的矩阵，200个样本
# X = np.random.normal(0, 1, size=(200, 2))
# # 构建一个生成y的函数，将y以>1.5还是<1.5进行分类
# y = np.array(X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5, dtype='int')
# # 绘制样本数据
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()

# 训练逻辑回归模型
# lr = LogisticRegression()
# lr.fit(X, y)
# 准确度评分
# print(lr.score(X, y))   # 0.605

# # 绘制决策边界
# decision_boundary(lr, axis=[-4, 4, -4, 4])
# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()


# 如果逻辑回归处理的是不规则决策边界的分类问题，那么我们就应该多考虑运用多项式回归
# 以下是一个为逻辑回归添加多项式项管道的示例：
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# # 构建管道
# def PolyLogisticRegression(degree, solver, multi_class):
#     return Pipeline([
#         ('poly', PolynomialFeatures(degree=degree)),
#         ('std_sca', StandardScaler()),
#         ('lr', LogisticRegression(solver=solver, multi_class=multi_class))
#     ])

# # 实例化多项式逻辑回归模型
# plr = PolyLogisticRegression(degree=20)
# # 训练
# plr.fit(X, y)
# # 准确度评分
# print(plr.score(X, y))   # 0.96
#
# # 绘制决策边界
# decision_boundary(plr, axis=[-4, 4, -4, 4])
# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()

# 鸢尾花三分类

# # 多项式逻辑回归分类器
# plr = PolyLogisticRegression(degree=200, solver='liblinear', multi_class='ovr')
# # 训练
# plr.fit(X_train, y_train)
# # 准确度评分
# print(plr.score(X_test, y_test))   # 0.7333333333333333
# # degree=100时：0.7333333333333333
#
# # 前两个特征的全部分类
# # 绘制决策边界（前两个特征的全部分类）
# decision_boundary_fill(plr, axis=[4, 8, 1.9, 4.5])
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
# plt.scatter(X[y == 2, 0], X[y == 2, 1], color='green')
# plt.show()


# 逻辑回归使用正则化

import numpy as np
import matplotlib.pyplot as plt

# 随机数种子，只需设置一次，设置后只要种子不变，每次生成相同的随机数
np.random.seed(1)
# 构建均值为0，标准差为1（标准正态分布）的矩阵，200个样本，每个样本2个特征
X = np.random.normal(0, 1, size=(200, 2))
# 构建一个生成y的函数，将y以>1.5还是<1.5进行分类
y = np.array(X[:, 0] ** 2 + X[:, 1] < 1.5, dtype='int')
# 在样本数据中加一些噪音
for _ in range(20):
    y[np.random.randint(200)] = 1
# 绘样本数据
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

# 逻辑回归（二分类）训练模型
from sklearn.model_selection import train_test_split

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 训练逻辑回归模型
# 默认使用L2正则化项，求解器默认使用lbfgs，惩罚系数（正则化强度的倒数）C默认为1
lr = LogisticRegression()
lr.fit(X_train, y_train)
# 准确度评分
print(lr.score(X_test, y_test))   # 0.875
# 模型参数
print(lr.get_params())
'''
{'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
'''

# 绘制决策边界
decision_boundary_fill(lr, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

# 多项式逻辑回归训练模型
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# 构建多项式逻辑回归管道
def PolyLogisticRegression(degree, penalty='l2', solver='lbfgs', multi_class='auto', C=1.0):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_sca', StandardScaler()),
        ('lr', LogisticRegression(penalty=penalty, solver=solver, multi_class=multi_class, C=C))
    ])

# 多项式逻辑回归分类器
# 默认使用L2正则化
# plr = PolyLogisticRegression(degree=20)
# 使用L1正则化
plr = PolyLogisticRegression(degree=20, penalty='l1', solver='liblinear', C=0.1)
# 训练
plr.fit(X_train, y_train)
# 准确度评分
print(plr.score(X_test, y_test))   # 0.925
# L2正则化| degree=2, C=1.0| 0.925
# L2正则化(增大正则化强度)| degree=2, C=0.2| 0.925
# L2正则化(增大阶数)| degree=20, C=1.0| 0.925
# L2正则化(增大阶数，同时增大正则化强度)| degree=20, C=0.2| 0.925
# L1正则化| degree=2, C=1.0| 0.925
# L1正则化(增大正则化强度)| degree=2, C=0.2| 0.875
# L1正则化(增大阶数)| degree=20, C=1.0| 0.95
# L1正则化(增大阶数，同时增大正则化强度)| degree=20, C=0.2| 0.925


# 绘制决策边界
decision_boundary_fill(plr, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

# 准确度随其他参数调整的变化记录
# 正则化	                            参数调整	          准确度
# L2正则化	                        degree=2, C=1.0	   0.925
# L2正则化(增大正则化强度)	            degree=2, C=0.2	   0.925
# L2正则化(增大阶数)	                degree=20, C=1.0	0.925
# L2正则化(增大阶数，同时增大正则化强度)	degree=20, C=0.2	0.925
# L1正则化	                        degree=2, C=1.0	    0.925
# L1正则化(增大正则化强度)	            degree=2, C=0.2	    0.875
# L1正则化(增大阶数)	                degree=20, C=1.0	0.95
# L1正则化(增大阶数，同时增大正则化强度)	degree=20, C=0.2	0.925


