"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/12/23
Time: 14:48
Description:
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1）数据收集及准备
# 假设：y=kx+b，其中k=1.25，b=3.5
# 生成随机数据
# 生成数据区间在0~10包含10个元素的均匀分布序列，返回数组
x = np.linspace(0, 10, 10)
# 参数
k = 1.25
b = 3.5
# polyval()：在给定的多项式系数下计算多项式值(计算多项式拟合的值)
y = np.polyval([k, b], x)

# 画散点图
# 给拟合值添加噪音
yi = y + 1.5 * np.random.randn(10)
plt.scatter(x, yi, c='green', marker='v')
# x、y轴标签名称
plt.xlabel('x')
plt.ylabel('y')
# plt.show()

# 2）选择模型及训练
# 训练集（本次使用全部输入为训练集）的x坐标转换为矩阵形式（参数类型需要）
X = np.asarray([[xi] for xi in x])
# print(X)
# print(y)

# sklearn.linear_model.LinearRegression线性回归模型求解
lr = LinearRegression()
lr.fit(X, y)
# 回归系数
kr = lr.coef_
# 截距
br = lr.intercept_
print(kr)              # [1.25]
print(br)              # 3.499999999999999
# plt.plot(x, y, color='blue', linewidth=2)
# plt.show()

# 3）模型评估
print(lr.score(X, y))  # 1.0

# 4）根据模型预测
# 输入新的样本数据2（二维矩阵类型），返回一维数组类型
# 20*1.25+3.5=28.5
print(lr.predict([[20]]))   # [28.5]

# =================

# 波士顿房价预测

"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/12/26
Time: 22:52
Description:
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# 加载波士顿房价的数据集
# boston = datasets.load_boston()
#
# boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
# boston_df['MEDV'] = boston.target
# print(boston_df.head())


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
print(boston_df.head().to_string())
'''
      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT  MEDV
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98  24.0
1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14  21.6
2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03  34.7
3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94  33.4
4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33  36.2
'''
# boston_df.to_csv('boston_dataset.csv', index=False, encoding='utf-8')

# 查看是否有空缺值
print(boston_df.isnull().sum())
'''
CRIM       0
ZN         0
INDUS      0
CHAS       0
NOX        0
RM         0
AGE        0
DIS        0
RAD        0
TAX        0
PTRATIO    0
B          0
LSTAT      0
MEDV       0
dtype: int64
'''

# 查看数据集大小
print(boston_df.shape)
'''
(506, 14)
'''

# 波士顿房屋数据集于1978年开始统计，涵盖了麻省波士顿不同郊区房屋14种特征的信息
# 该数据集共有506个样本，每个样本有13个特征属性及1个目标标签变量MEDV（房价中位数）
# 数据集的属性信息（13特征+1标签）如下：
'''
CRIM: 城镇人均犯罪率
ZN: 住宅用地所占比例
INDUS: 城镇中非住宅用地所占比例
CHAS: 是否靠近查尔斯河（1表示靠近，0表示不靠近）
NOX: 环保指数：一氧化氮浓度（每1000万份）
RM: 房屋平均房间数
AGE: 自住房屋中建于1940年前的房屋所占比例
DIS: 距离5个波士顿就业中心的加权距离
RAD: 距离高速公路的便利指数
TAX: 每10000美元的全额物业税率
PTRATIO: 城镇中学生与教师比例
B: 城镇中的黑人比例
LSTAT: 地区中房东属于低收入人群比例
MEDV: 自住房屋房价中位数（即均价，单位：千美元）
'''
# 应用：房价预测：训练Boston数据集，预测波士顿地区房价中位数

# 特征之间的相关性分析：
# 计算每一个特征与标签MEDV（均价）的相关系数
corr_data = boston_df.corr().iloc[-1]
# 可视化
# corr_data.sort_values().plot.bar()
# plt.show()

# 结论：各特征与标签之间的相关性不大

# 查看特征的数据分布区间：
# sns.boxplot(data=boston_df.iloc[:, 0:13])
# plt.show()

# 结论：各特征的取值范围差异较大

# 解决：标准化/归一化（转换为符合标准正态分布）
# 原因：
# - 过大或过小的数值范围会导致计算时的浮点上溢或下溢
# - 不同的数值范围会导致不同属性对模型的重要性不同，这会对优化的过程造成困难，使训练时间大大的加长

from sklearn.linear_model import LinearRegression        # 线性回归模型
from sklearn.preprocessing import StandardScaler         # 特征工程：标准化
from sklearn.preprocessing import MinMaxScaler           # 特征工程：归一化
from sklearn.model_selection import train_test_split     # 数据集划分
from sklearn.metrics import mean_squared_error as MSE    # 均方误差

# 特征
X = boston_df.iloc[:, 0: 13]
# print(X.head().to_string())
# 标签
y = np.array(boston_df.iloc[:, -1])
# print(y.head().to_string())

# 将数据集划分为训练集和测试集（随机采样20%的数据作为测试样本，其余作为训练样本）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 数据标准化（只需处理特征）

# 初始化对特征和目标值的标准化器
ss = StandardScaler()
# 分别对训练和测试数据的特征进行标准化处理
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

# 数据归一化（只需处理特征）
# MinMaxScaler对异常值较敏感

# # 初始化转换器（feature_range是归一化的范围，即最小值-最大值）
# transfer = MinMaxScaler(feature_range=(0, 1))
# # 分别对训练和测试数据的特征进行归一化处理
# X_train = transfer.fit_transform(X_train)
# X_test = transfer.fit_transform(X_test)


# 使用线性回归模型预测房价
# 训练模型
lr = LinearRegression()
# 在训练集上拟合模型
lr.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = lr.predict(X_test)

# 可视化
# 设置画板尺寸
plt.figure(figsize=(15, 6))
# 设置字体
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title('波士顿房价预测曲线与实际曲线对比图', fontsize=15)
# x轴数据
x = range(len(y_test))
# 实际价格曲线
plt.plot(x, y_test, color='r', label='实际房价')
# 预测价格曲线
plt.plot(x, y_pred, color='g', ls='--', label='预测房价')
# 显示图例
plt.legend(fontsize=12, loc=1)
plt.show()


# 模型评估
# 决定系数R^2
print(lr.score(X_test, y_test))            # 0.6860012233572248
# 均方误差MSE
print(MSE(y_test, y_pred))                 # 24.472608321135862
# 均方根误差RMSE
print(MSE(y_test, y_pred, squared=False))  # 4.946979717073424




