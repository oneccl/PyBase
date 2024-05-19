
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression        # 线性回归
from sklearn.preprocessing import PolynomialFeatures     # 多项式回归
from sklearn.pipeline import Pipeline                    # 流程管道
from sklearn.preprocessing import StandardScaler         # 标准化、归一化
from sklearn.metrics import mean_squared_error as MSE    # 均方误差
from sklearn.model_selection import train_test_split     # 划分训练测试集

# 生成具有一个特征的三个样本[0, 1, 2]
X = np.arange(0, 3).reshape(-1, 1)
# 使用PolynomialFeatures添加新特征
X_ = PolynomialFeatures(degree=3).fit_transform(X)
print(X.reshape(-1))
'''
[0 1 2]
'''
print(X_)
'''
[[1. 0. 0. 0.]
 [1. 1. 1. 1.]
 [1. 2. 4. 8.]]
'''

X_p = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
print(X_p)
'''
[[0. 0. 0.]
 [1. 1. 1.]
 [2. 4. 8.]]
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression        # 线性回归
from sklearn.metrics import mean_squared_error as MSE    # 均方误差

# 按顺序产生一组固定的数组，若要使每次生成的随机数(组)相同，需要每次调用都np.random.seed()一下，表示种子相同，从而生成的随机数相同；种子可随意给定
# 更多参考：https://zhuanlan.zhihu.com/p/266472620、https://www.cnblogs.com/subic/p/8454025.html
np.random.seed(1)
# 生成100个样本
m = 100
# 从一个均匀分布的区域中随机采样，在-3~3区间生成均匀分布的100个元素，返回数组
x = np.random.uniform(-3, 3, size=m)
# 从标准正态（高斯）分布中抽取随机样本，给拟合值添加噪声
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, size=m)
# 散点图可视化
plt.scatter(x, y, c='green')
plt.show()

# 线性回归
# X的形状需要2D（X的维度必须是(样本数,特征数)）
X = x.reshape(-1, 1)
# print(X)

# 线性回归模型
lr = LinearRegression()
lr.fit(X, y)

# 预测值
y_pred = lr.predict(X)
# 可视化
plt.plot(x, y_pred, color='blue')
# x、y轴标签名称
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 决定系数R^2
print(lr.score(X, y))    # 0.45068280832255214
# 均方误差MSE
mse = MSE(y, y_pred)
print(mse)               # 2.924678017341418

# 多项式
from sklearn.preprocessing import PolynomialFeatures     # 多项式回归

# X的形状需要2D（X的维度必须是(样本数,特征数)）
X = x.reshape(-1, 1)
# print(X)

# 使用PolynomialFeatures添加新特征
poly = PolynomialFeatures(degree=2, include_bias=False)
# 拟合数据，然后对其进行转换
X_ = poly.fit_transform(X)
# 将数据转换为多项式特征（否则报错：X has 1 features, but LinearRegression is expecting 2 features as input.）
X = poly.transform(X)

# 线性回归模型
lr = LinearRegression()
lr.fit(X_, y)

# 预测值
y_pred = lr.predict(X)
# 可视化
# np.sort(x)：对样本数据从小到大排序，也就是从数组中最小的元素开始绘制
# np.argsort(x)：求数组从小到大排序后元素的索引
# y_pred[np.argsort(x)]：求数组从小到大排序后元素对应的y值
plt.plot(np.sort(x), y_pred[np.argsort(x)], color='blue')
# x、y轴标签名称
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 决定系数R^2
print(lr.score(X, y))    # 0.8515721504191915
# 均方误差MSE
mse = MSE(y, y_pred)
print(mse)               # 0.7902604822991763

# 输入新的样本数据并转换为多项式特征，返回一维数组类型
# 0.5*4^2+4+2=14
print(lr.predict(poly.transform([[4]])))    # [14.35009763]


# 多项式回归与Pipeline

# 导入Pipeline和其他需要打包进Pipeline的类
from sklearn.pipeline import Pipeline    # 流程管道
from sklearn.preprocessing import StandardScaler    # 标准化、归一化

# 封装Pipeline，制作一个多项式回归的管道
def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_sca", StandardScaler()),
        ("lr", LinearRegression())
    ])


# X的形状需要2D（X的维度必须是(样本数,特征数)）
X = x.reshape(-1, 1)
# print(X)

# 实例化一个多项式回归
pr = PolynomialRegression(degree=2)
# 训练
pr.fit(X, y)
# 预测值
y_pred = pr.predict(X)

# 可视化原始数据和预测数据
plt.scatter(x, y, c='green')
plt.plot(np.sort(x), y_pred[np.argsort(x)], color='blue')
# x、y轴标签名称
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 决定系数R^2
print(lr.score(X, y))    # 0.8515721504191915
# 均方误差MSE
mse = MSE(y, y_pred)
print(mse)               # 0.7902604822991763

print(pr.predict([[4]]))    # [14.35009763]


# 过拟合与模型泛化

from sklearn.model_selection import train_test_split     # 划分训练测试集

# 拆分样本数据为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 实例化一个多项式回归
pr = PolynomialRegression(degree=100)
# 训练
pr.fit(X_train, y_train)
# 使用训练出的模型对训练特征数据预测目标值
y_pred = pr.predict(X_train)

# 求均方误差
mse = MSE(y_train, y_pred)
print(mse)               # 0.3448035353366178


# 使用训练出的模型对测试特征数据预测目标值
y_pred = pr.predict(X_test)

# 求均方误差
mse = MSE(y_test, y_pred)
print(mse)               # 68902753.20829579


# 拆分样本数据为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 实例化一个多项式回归
pr = PolynomialRegression(degree=2)
# 训练
pr.fit(X_train, y_train)

# 使用训练出的模型对测试特征数据预测目标值
y_pred = pr.predict(X_test)

# 求均方误差
mse = MSE(y_test, y_pred)
print(mse)               # 0.849881699724714




