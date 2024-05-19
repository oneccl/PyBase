
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 线性回归：简单线性回归、多项式回归和多元线性回归
# 机器学习的真正力量来自于训练模型。机器学习模型通过历史数据（先验知识）进行训练以自动捕获（寻找）数据间的依赖关系（规律），并通过此规律预测新的结果

# 案例：南瓜价格：什么时候是购买南瓜的最佳时间？
df = pd.read_csv('new_pumpkins.csv')
# print(df.to_string())

# 概念
# 线性回归包括简单线性回归（一元线性回归）、多项式回归、多元线性回归
# 简单线性回归可以看作是次数和特征数量为1的多元线性回归；多项式回归也可以转化为多元线性回归

# 简单线性回归

# 最小二乘回归就是绘制回归线的常用方法
# 最小二乘法（Least squares）是一种数学优化技术。它通过最小化误差的平方和寻找数据的最佳匹配函数。利用最小二乘法可以方便地求得未知的线性函数，并使得线性函数拟合的数据与实际数据之间误差的平方和为最小。最小二乘法还可用于曲线拟合。其核心就是保证所有数据偏差的平方和最小

# 最小二乘回归线也称为最佳拟合线：Y = wX + b
# X是自变量，Y是因变量，直线的斜率是w，也称权重或系数；b是Y的截距，也称常数项系数
# 因为监督分类是有训练的机器学习，因此我们训练的数据中已经包括了特征变量X与预测标签Y，因此基于最小二乘回归模型需要求解的参数只有w与b
# 因此，训练此模型的流程就可以分为：计算w和b；测试评估模型

# 随着特征变量的增多，我们需要计算的w（系数或斜率）也增多，截距或常量b则始终只有一个

# 多项式线下回归

# 虽然变量之间有时存在线性关系，如南瓜体积越大，价格越高；但并不是所有的关系都是呈现线性，在这种情况下一元线性回归线不能很好的预测价格，因此我们可以采用多项式回归
# 在数学中，多项式（Polynomial）是指由变量、系数以及它们之间的加、减、乘、幂运算（非负整数次方）得到的表达式。如若给定两个输入特征变量X和Y，一次多项式表示输入特征为X,Y；二次多项式表示X^2、XY和Y^2三个特征；以此类推
# 多项式中的每个单项式叫做多项式的项，这些单项式中的最高项次数就是这个多项式的次数。单项式由单个或多个变量和系数相乘组成，也有常数项；多项式回归模型创建的曲线可以更好地拟合非线性数据

# 多项式回归算法并没有新的特点，完全是使用线性回归的思路，关键在于为数据添加新的特征，而这些新的特征是原有的特征的多项式组合，采用这样的方式就能解决非线性问题。这样的思路跟PCA这种降维思想刚好相反，而多项式回归则是升维，添加了新的特征之后，使得更好地拟合高维数据
# make_pipeline()：可以将多个算法模型串联起来，如将特征提取、归一化、分类组织在一起形成一个典型的机器学习问题工作流
# PolynomialFeatures()：可以通过输入的变量来创建新特征，其参数表示多项式的次数

# 多元线性回归
# 多元，即特征向量不唯一，这与多项式回归类似；实际上多项式回归也可以转化为多元线性回归
# 随着特征变量X的增多，我们需要计算的w系数也增加。这与简单线性回归原理区别不大，只是拟合出来的回归线是曲线。多元回归线往往有更好的相关性


# 相关系数
# 相关系数是表示自变量与因变量之间相关程度的系数，其取值区间为（0，1），越靠近1相关系数越高；1为完全相关，0为完全不相关；我们可以使用散点图来可视化此系数。聚集在回归线周围的数据点具有高度相关性，分散在回归线周围的数据点具有低相关性
# 一个优秀的线性回归模型是使用最小二乘回归方法且数据具有高度相关性


# 简单线性回归
# 所谓简单线性回归，就是单变量的线性回归，如销售日期与南瓜价格的线性关系或南瓜种类与南瓜价格的线性关系，简单线性回归的回归线是一条直线，它可以单方面的探索特征间的相关性
# 按月预测每体积单位的南瓜价格：将月份作为自变量X，将南瓜价格作为因变量Y，根据拟合出的回归线，我们便可根据月份预测出当月的南瓜价格，预测结果不仅取决于斜率w，也取决于截距b

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split

# 销售日期-南瓜价格
# 数据格式转换：将输入值（特征变量）和预期的输出值（预测标签）分离到单独的numpy数组中：
# 线性回归需要一个二维数组作为输入，其中数组的每一行都对应于输入要素的向量；由于我们只有一个输入，因此需要一个形状为N×1的数组，其中N是数据集大小
# X = df['DayOfYear'].to_numpy().reshape(-1, 1)
# y = df['Price']

# # 划分数据集：将数据集拆分为训练集和测试集，以便在训练后验证模型
# '''
# train_test_split(train_data,train_target,test_size,random_state)
# - train_data：被划分的样本特征集
# - train_target：被划分的样本标签
# - test_size：若是浮点数，在0-1之间，表示样本占比；若是整数表示样本的数量
# - random_state：随机数种子，在需要重复试验的时候，保证得到一组一样的随机数
# '''
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# # 训练模型
# # 线性回归训练器
# lr = LinearRegression()
# # 训练线性回归模型
# lr.fit(X_train, y_train)
#
# # 经过训练的LinearRegression()对象包含回归的所有系数，我们可以使用对象属性访问这些系数
# # 回归系数
# print(lr.coef_)         # [-0.06874322]
# # 截距
# print(lr.intercept_)    # 47.54998044636355
#
# # 模型评估
# # 为了评估模型的精度，我们在测试数据集上预测价格，然后计算预测价格与实际价格的误差
# y_pred = lr.predict(X_test)
# # 这可以使用均方根误差（RMSE）指标来完成，它是预期值和预测值之间所有平方差的平均值（数学期望）的平方根，它的本质是在MSE上做了一个开根号，这样做是为了将评估值的量纲和原值的量纲保持一致
# # 计算MSE和RMSE指标
# mse = MSE(y_test, y_pred)
# rmse = MSE(y_test, y_pred, squared=False)
# print(mse, rmse)    # 113.62754126951437 10.659622004063483
# # 计算相关系数
# score = lr.score(X_test, y_test)
# print(score)        # 0.033931104080158314
# # 线性回归可视化
# plt.scatter(X_test, y_test)
# plt.plot(X_test, y_pred, color='red')
# plt.show()

# # 南瓜种类-南瓜价格
# # 使用不同的南瓜类型来检验南瓜类型与价格的相关性
#
# # 以南瓜的类别作为特征变量，并转化数值类型
# # 将南瓜数据被分为4类，每类包含415个样本
# X = pd.get_dummies(df['Variety'])
# # print(X.info())
# y = df['Price']
#
# # 以8：2的比例划分训练集与测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# # 训练模型
# # 线性回归训练器
# lr = LinearRegression()
# # 训练线性回归模型
# lr.fit(X_train, y_train)
# # 将测试集上的预测
# y_pred = lr.predict(X_test)
#
# # 模型评估
# # 计算MSE和RMSE指标
# mse = MSE(y_test, y_pred)
# rmse = MSE(y_test, y_pred, squared=False)
# print(mse, rmse)    # 27.621753576504613 5.255640168096044
# # 计算相关系数
# score = lr.score(X_test, y_test)
# print(score)        # 0.765158018180377

# 结论：南瓜价格与南瓜品种的相关性高于销售日期


# 多项式回归
# 销售日期-南瓜价格
# 使用二阶多项式X^2来构建回归模型。如果需要，也可以使用更高阶多项式
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# # 构建自动化流程，其中PolynomialFeatures(2)是对输入数据进行2次多项式拟合，后者为构建线性回归模型
# pipe = make_pipeline(PolynomialFeatures(2), LinearRegression())
# # 训练模型
# pipe.fit(X_train, y_train)
# # 在测试集上预测
# y_pred = pipe.predict(X_test)
#
# # 模型评估
# # 计算MSE和RMSE指标
# mse = MSE(y_test, y_pred)
# rmse = MSE(y_test, y_pred, squared=False)
# print(mse, rmse)    # 113.21565590044068 10.640284577981957
# # 计算相关系数
# score = pipe.score(X_test, y_test)
# print(score)        # 0.03743298081973112
# # 线性回归可视化
# plt.scatter(X_test, y_test)
# plt.plot(np.sort(X_test), y_pred[np.argsort(X_test)], color='red')
# plt.show()

# 南瓜种类-南瓜价格

# # 构建自动化流程，其中PolynomialFeatures(2)是对输入数据进行2次多项式拟合，后者为构建线性回归模型
# pipe = make_pipeline(PolynomialFeatures(2), LinearRegression())
# # 训练模型
# pipe.fit(X_train, y_train)
# # 在测试集上预测
# y_pred = pipe.predict(X_test)
#
# # 模型评估
# # 计算MSE和RMSE指标
# mse = MSE(y_test, y_pred)
# rmse = MSE(y_test, y_pred, squared=False)
# print(mse, rmse)    # 27.297243975903612 5.224676447006418
# # 计算相关系数
# score = pipe.score(X_test, y_test)
# print(score)        # 0.7679170203383522


# 多元线性回归
# 多元，即多个特征，多项式是基于输入特征生成特征，而多元既可以表示1~N个生成的特征，也表示输入的特征
# 基于多种特征变量回归的特征曲线拟合出来的机器学习模型更加适用于实际操作

# 数据准备：考虑销售日期、销售地区、南瓜种类等多个因素
# 为了获得更准确的预测，我们需要考虑更多的分类特征
# X = pd.get_dummies(df['Variety']).join(df['DayOfYear']).join(pd.get_dummies(df['City Name'])).join(pd.get_dummies(df['Package']))
# y = df['Price']
#
# # 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# # 训练模型
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# # 在测试集上预测
# y_pred = lr.predict(X_test)
#
# # 模型评估
# # 计算MSE和RMSE指标
# mse = MSE(y_test, y_pred)
# rmse = MSE(y_test, y_pred, squared=False)
# print(mse, rmse)    # 8.139721848221225 2.8530197770469843
# # 计算相关系数
# score = lr.score(X_test, y_test)
# print(score)        # 0.9307955447143391


# 多元线性回归（多项式）

X = pd.get_dummies(df['Variety']).join(df['Month']).join(pd.get_dummies(df['City Name'])).join(pd.get_dummies(df['Package']))
y = df['Price']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 构建自动化流程，其中PolynomialFeatures(3)是对输入数据进行3次多项式拟合，后者为构建线性回归模型
pipe = make_pipeline(PolynomialFeatures(3), LinearRegression())
# 训练模型
pipe.fit(X_train, y_train)
# 在测试集上预测
y_pred = pipe.predict(X_test)

# 模型评估
# 计算MSE和RMSE指标
mse = MSE(y_test, y_pred)
rmse = MSE(y_test, y_pred, squared=False)
print(mse, rmse)    # 4.668034573531741 2.160563485188931
# 计算相关系数
score = pipe.score(X_test, y_test)
print(score)        # 0.9603120602964471


'''
np.polyfit(x, y, deg)
- x、y：需要拟合的散点的X、Y坐标序列
- deg：需要拟合的多项式的最高项数，当dug=2时，会构建ax^2+bx+c，拟合函数，返回系数[a, b, c]
'''
import pylab
import matplotlib.pyplot as plt

# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [2.83, 9.53, 14.52, 21.57, 38.26, 53.92, 73.15, 101.56, 129.54, 169.75, 207.59]
#
# z = np.polyfit(x, y, 2)    # 曲线拟合，返回多项式的回归系数
# print(list(z))        # [2.1063519813519838, -1.0216107226107443, 6.133006993007059]
# p = np.poly1d(z)      # 返回多项式的表达式
# print(p)              # 2.106 x^2 - 1.022 x + 6.133
# print(p(11))          # 249.7638787878789
# y_pred = p(x)         # 根据函数表达式，求解y
#
# plt.plot(x, y, "*", label='original')
# plt.plot(x, y_pred, "r", label='fit')
# pylab.show()

# 拟合任意函数（非线性回归）
from scipy.optimize import curve_fit

# def gauss(x, a, b, c):
#     return a*np.exp(-(x-b)**2/c**2)
#
# x = np.arange(100)/10
# y = gauss(x, 2, 5, 3) + np.random.rand(100)/10
#
# # 曲线拟合，func为任意函数表达式，popt为多项式的各项系数，pcov为popt参数下的协方差
# popt, pcov = curve_fit(gauss, x, y)
# print(popt)
# y_pred = gauss(x, *popt)  # 根据函数表达式，求解y
# print(y_pred)
#
# plt.scatter(x, y, marker='.', label='original')
# plt.plot(x, y_pred, c='red', label='fit')
# plt.show()

# 多元多项式曲线拟合（结果是多维图形）
# def func(x, a, b, c, d):
#     r = a * np.exp(-((x[0] - b) ** 2 + (x[1] - d) ** 2) / (2 * c ** 2))
#     return r.ravel()
#
# # np.indices((m,n)): 返回一个m*n的序号网格，密集分布
# x = np.indices([10, 10])
# z = func(x, 10, 5, 2, 5) + np.random.normal(size=100)/100
#
# popt, pcov = curve_fit(func, x, z)
# print(popt)     # [9.99433072 4.99879919 2.00075038 4.99971176]
#
# z = z.reshape(10, 10)
# z_pred = func(x, *popt).reshape(10, 10)  # 根据函数表达式，求解z
#
# ax = plt.subplot(projection='3d')
# ax.scatter3D(x[0], x[1], z, color='red')
# ax.plot_surface(x[0], x[1], z_pred, cmap='rainbow')
# plt.show()







