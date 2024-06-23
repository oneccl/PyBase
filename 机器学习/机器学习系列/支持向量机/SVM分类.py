
# Scikit-Learn支持向量机分类

# https://blog.csdn.net/weixin_42279212/article/details/121504641

# 支持向量机（SVM）概述
# 在机器学习中，支持向量机（Support Vector Machine，SVM）算法既可以用于回归问题（SVR），也可以用于分类问题（SVC）
# 支持向量机是一种经典的监督学习算法，通常用于分类问题。SVM在机器学习知识结构中的位置如下：

# SVM的核心思想是将分类问题转化为寻找分类平面的问题，并通过最大化分类边界点（支持向量）到分类平面的距离（间隔）来实现分类
# 图
# 如图所示，左图展示了三种可能的线性分类器的决策边界，虚线所代表的模型表现非常糟糕，甚至都无法正确实现分类；其余两个模型在训练集上表现堪称完美，但是它们的决策边界与实例过于接近，导致在面对新样本时，表现可能不会太好
# 右图中的实线代表SVM分类器的决策边界，两虚线表示最大间隔超平面，虚线之间的距离（两个异类支持向量到超平面的距离之和）称为超平面最大间隔，简称间隔；SVM的决策边界不仅分离了两个类别，而且尽可能的远离了最近的训练实例，距离决策边界最近的实例称为支持向量

# SVM原理

# SVM的最优化问题就是要找到各类样本点到超平面的距离最远，也就是找到最大间隔超平面。任意超平面的方程为
# $$\omega^Tx+b=0$$
# 其中$\omega$为超平面的法向量，决定了超平面的方向；$b$为位移项，决定了超平面到原点间的距离
# 二维空间点$(x,y)$到直线$Ax+By+C=0$的距离公式为
# $$\frac{|Ax+By+C|}{\sqrt{A^2+B^2}}$$
# 扩展到N维空间中，点$(x_1,x_2,...x_n)$到直线$\omega^Tx+b=0$的距离为
# $$\frac{|\omega^Tx+b|}{||\omega||}$$
# 其中，$||\omega||$=$\sqrt{\omega_1^2+\omega_2^2+...+\omega_n^2}$

# SVM假设样本是线性可分的，则任意样本点到超平面的距离可写为
# $$d=\frac{|\omega^Tx+b|}{||\omega||}$$
# 为方便描述和计算，设$y_i\in{-1,1}$，其中1表示正例，-1表示负例，则有
# $$\begin{cases} \omega^Tx_i+b≥+1 & \text{y_i=+1} \\\\ \omega^Tx_i+b≤-1 & \text{y_i=-1} \end{cases}$$
# 此时，两个异类支持向量到超平面的距离之和为
# $$\gamma_i=y_i\left(\frac{\omega^T}{||\omega||}\cdot x_i + \frac{b}{||\omega||} \right) = \frac{2}{||\omega||}$$
# 其中，$\gamma$称为间隔。最大间隔不仅与$\omega$有关，偏置$b$也会隐性影响超平面的位置，进而对间隔产生影响
# 现在，我们只需要使间隔$\gamma$最大，即
# $$\arg \mathop{\max}\limits_{\omega,b} \frac{2}{||\omega||}$$
# 最大化间隔$\gamma$，显然只需要最小化$||\omega||$，于是，上式可重写为
# $$\arg \mathop{\min}\limits_{\omega,b} \frac{1}{2}||\omega||^2$$
# 这里的平方和之前一样，一是为了方便计算，二是可以将目标函数转化为凸函数的凸优化问题。称该式为SVM的基本型

# SVM的损失函数
# 软间隔与硬间隔
# 如果我们严格让所有实例都不在最大间隔之间，并且位于正确的一边，这就是硬间隔分类。但是硬间隔分类有两个问题：首先，它只在数据是线性可分时才有效；其次，它对异常值较敏感
# 要避免这些问题，可以使用更灵活的模型。目标是尽可能在保持最大间隔的同时允许间隔违例（在最大间隔之间，甚至位于错误的一边），在最大间隔与违例之间找到良好的平衡，这就是软间隔分类
# 软间隔的目标函数为
# $$J=\frac{1}{2}||\omega||^2 + C\sum_{i=1}^{n}\varepsilon_i$$
# 其中，超参数$C$为惩罚系数，$\varepsilon$为松弛因子。$C$越小，惩罚越小（间隔越宽，违例越多）

# 核函数
# 对于非线性数据集，线性支持向量机无法处理。我们希望将非线性问题转化为线性可分问题来求解，这时，需要引入一个新的概念：核函数
# 核函数可以将样本从原始空间映射到一个高维空间，使得样本在新的空间中线性可分
# 核函数将原始空间中的向量作为输入向量，并返回转换后的特征空间中向量的内积。通过核方法可以学习非线性支持向量机，等价于在高维特征空间中学习线性支持向量机
# 所以在非线性SVM中，核函数的选择就是影响SVM最大的变量。常用核函数有：线性核、多项式核、高斯核、拉普拉斯核和Sigmoid核等

# 优缺点
# 优点
# - 可适用于处理高维空间数据，对于数据维度远高于样本数据量的情况也有效
# - 在决策函数中使用少部分训练数据(支持向量)进行决策，内存占用小，效率高
# - 通过支持向量选取最优决策边界，对噪声和异常值的敏感度较低，稳定性较好
# - 更加通用，可处理非线性分类任务，提供了多种通用核函数，也支持自定义核函数
# 缺点
# - 解释性差：不像K-Means、决策树那样直观，不易于理解，可解释性差
# - 对参数和核函数敏感：性能高度依赖于惩罚参数C和核函数的选择。如果参数选择不当，容易导致过拟合或欠拟合
# - 非线性分类训练时间长：核函数涉及到二次规划问题，需要使用复杂的优化算法，当支持向量的数量较大时，计算复杂度较高
#


# Scikit-Learn支持向量机分类API
# Scikit-Learn支持向量机分类的API如下：


# 手写数字识别
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# 手写数字数据集
# 手写数字数据集由1797个8x8像素的数字图像组成。数据集的每个图像存储为8x8灰度值的二维数组；数据集的属性存储每个图像代表的数字，这包含在图像的标题中

# 加载手写数字数据集
data = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, data.images, data.target):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Label: {label}")

# plt.show()

# 手写数字图像存储为一个8x8的二维数组
# print(data.images[0])

# 为了对这些数据应用分类器，我们需要将图像展平，将每个图像的灰度值从8x8的二维数组转换为64x1的一维数组
from sklearn.model_selection import train_test_split

n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
y = data.target

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.svm import SVC

# SVM分类器
clf = SVC()

# 训练模型
clf.fit(X_train, y_train)

# 在测试集上预测
y_pred = clf.predict(X_test)

# 准确度评分
print(clf.score(X_test, y_test))  # 0.9916666666666667


# 鸢尾花分类

from matplotlib.colors import ListedColormap

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


from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
# 使用鸢尾花前两个特征的前两个分类（二分类）
X = iris.data
y = iris.target
X = X[y < 2, :2]
y = y[y < 2]

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# SVM分类器
clf = SVC()

# 训练模型
clf.fit(X_train, y_train)

# 在测试集上预测
y_pred = clf.predict(X_test)

# 准确度评分
print(clf.score(X_test, y_test))  # 1.0

# 二分类决策边界
decision_boundary_fill(clf, axis=[4.0, 7.0, 2, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
plt.show()


# 使用鸢尾花前两个特征的全部分类（三分类）
X = iris.data[:, :2]
y = iris.target

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# SVM分类器
# clf = SVC(C=1.0, decision_function_shape='ovr', break_ties=True)
clf = SVC(kernel='poly', degree=20)

# 训练模型
clf.fit(X_train, y_train)

# 在测试集上预测
y_pred = clf.predict(X_test)

# 准确度评分
print(clf.score(X_test, y_test))  # 0.7333333333333333
# 多项式核函数：degree=3：0.7333333333333333  degree=10：0.6333333333333333

# 三分类决策边界
decision_boundary_fill(clf, axis=[4, 8, 1.9, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='green')
plt.show()



