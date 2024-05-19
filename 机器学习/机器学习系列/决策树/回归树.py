
# 决策树

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
# https://scikit-learn.org.cn/view/785.html

# 条件概率、全概率公式、贝叶斯公式
# 样本空间：一次试验所有可能的结果的集合称为样本空间，用$\Omega$表示
# 样本点：每一种可能的结果称为样本点，样本点也称为单例、基本事件
# 随机事件：一个随机事件是样本空间$\Omega$的子集，由若干个样本点构成，用大写字母$A,B,...$表示
# 由于我们将随机事件定义为了样本空间$\Omega$的子集，故我们可以将集合运算（交、并、补等）移植到随机事件上。记号与集合运算保持一致
# 特别的，事件的并$A\cupB$也可记作$A+B$，事件的交$A\capB$也可记作$AB$，此时也可分别称作和事件和积事件

# 独立事件：定义：事件A与事件B互不影响（不相关），事件A与事件B同时发生的概率等于事件A发生的概率乘以事件B发生的概率，即
# $$P(AB)=P(A)×P(B)$$
# 或
# $$P(A|B)=P(A)$$
# 例如，抛骰子，两次抛出的点数没有影响；再例如，从袋子中随机有放回的取出一个球（独立重复试验），第二次取出某种颜色球的概率不受第一次影响
# 互斥事件：定义：在一次试验中，事件A与事件B不能同时发生，事件A的发生与否决定了事件B的发生与否，即
# $$P(AB)=0$$
# 或
# $$P(A∪B)=P(A)+P(B)$$
# 例如，抛硬币，只要正面朝上，那么一定不会反面朝上

# 条件概率：定义：事件A与事件B互为相关事件，事件B在事件A发生的条件下发生的概率，即
# $$P(B|A)=\frac{P(AB)}{P(A)}$$
# 例如，从袋子中随机不放回的取出一个球，第一次取出某种颜色的球会影响第二次取出某种颜色球的概率
# 当事件A与事件B是独立事件时，条件概率等于事件本身的概率，即
# $$P(A|B)=P(A)$$
# 通常，我们可以通过画决策树的方式来表达条件概率
# 画决策树的步骤分三步：确定根节点（制定目标）；确定子节点（例举目标的所有实现方案）；评估（评估每种方案实现的概率）
# 根据条件概率定义我们可以得到概率乘法公式和全概率公式：
# 概率乘法公式：在概率空间中，若$P(A)$>0，则对任意事件B都有
# $$P(AB)=P(A)P(B|A)$$
# 事件A与事件B同时发生的概率等于事件A发生的概率乘以事件A已发生的条件下事件B发生的概率；若事件A与事件B不相关，则事件A与事件B同时发生的概率等于每个事件单独发生概率的乘积
# 全概率公式：在概率空间中，若一组事件$A_1,A_2,...,A_n$两两互斥且概率和为1，则对任意事件B都有
# $$P(B)=\sum_{i=1}^{n}P(A_i)P(B|A_i)$$
# 推导过程如下：设$\Omega$是一个必然事件，且$\Omega$为事件全集$A_1,A_2,...,A_n$，即
# $$\Omega=A_1+A_2+...+A_n$$
# 因此有
# $$P(B)=P(\OmegaB)=P(\Omega∪B)=P(A_1B)+P(A_2B)+...+P(A_nB)=P(A_1)P(B|A_1)+...+P(A_n)P(B|A_n)$$

# 贝叶斯（Bayes）公式
# 一般的，设可能导致事件B发生的原因为$A_1,A_2,...,A_n$，则在$P(A_i)$和$P(B|A_i)$已知时可以通过全概率公式计算事件B发生的概率
# 但是，在很多情况下，我们需要根据事件B发生这一结果反推其各个原因事件的发生概率：
# $$P(A_k|B)=\frac{P(A_kB)}{P(B)}=\frac{P(A_k)P(B|A_k)}{\sum_{i=1}^{n}P(A_i)P(B|A_i)}$$
# 上式为贝叶斯公式。其中，$P(A_1),P(A_2),...,P(A_n)$称为先验概率（经验），是试验前已知的；$P(A_k|B)$称为后验概率（试验），后验概率是对先验概率的更新和修正。通过不断迭代，从而达到最优，由此得到的决策叫做贝叶斯决策
# 贝叶斯公式将条件概率$P(A|B)$与$P(B|A)$紧密联系起来，其最根本的数学基础就是
# $$P(AB)=P(A)P(B|A)=P(B)P(A|B)$$
# 其本质内涵在于，全概率公式由因得果，贝叶斯公式则由果推因。贝叶斯公式在结果事件B已经发生的情况下，推断结果事件B是由于原因事件$A_i$造成的概率大小


# 信息熵、条件熵、信息增益、信息增益比与基尼指数
# 信息量
# 信息熵
# 条件熵
# 条件熵是指在给定随机变量X的条件下，随机变量Y的不确定性，即
# $$\begin{aligned}
# H(H|X)&=\sum_{i=1}^{n}P(X=x_i)H(Y|X=x_i) \\
# &=-\sum_{i=1}^{n}P(x_i)\sum_{j=1}^{m}P(y_j|x_i)\ln P(y_j|x_i) \\
# &=-\sum_{i=1}^{n}\sum_{j=1}^{m}P(x_iy_j)\ln P(y_j|x_i) \\
# &=-\sum_{x,y}^{}P(xy)\ln P(y|x)
# \end{aligned}$$
# 上式表示Y的条件概率分布的熵对X的期望。其物理意义为：在得知某一确定信息的基础上获取另外一个信息时所获得的信息量
# 当信息熵和条件熵是由训练数据估计而来时，那么对应的熵和条件熵称为经验熵和经验条件熵

# 信息增益（互信息）
# 信息增益表示信息X使信息Y的不确定性减少的程度，即信息X让信息Y的不确定性降低
# 在条件熵中我们发现相关的信息可以消除不确定性，所以需要一个度量相关性的变量：信息增益（互信息）
# $$I(X,Y)=H(X)-H(Y|X)=H(Y)-H(X|Y)$$
# 上式的物理意义是，某一确定信息与在此基础上获取另外一个信息时所需要的增量信息量。当X与Y完全不相关时，$I(X,Y)$=0

# 信息增益比
# 信息增益的大小是相对于训练数据而言的，并没有绝对意义。当某个特征的取值种类非常多时，会导致该特征对训练数据的信息增益偏大，反之，信息增益会偏小。使用信息增益比可以对这一问题进行校正。这是特征选择的另一准则
# 特征A对训练数据集D的信息增益比定义为：其信息增益$G(D,A)$与训练数据集D关于特征A值的熵$H_A(D)$之比，即
# $$G_R(D,A)=\frac{G(D,A)}{H_A(D)}$$
# 其中
# $$H_A(D)=-\sum_{i=1}^{n}\frac{|D_i|}{|D|}\ln \frac{|D_i|}{|D|}$$

# 基尼指数
# 基尼制数（Gini）是衡量数据集纯度的一种方式。在分类问题中，假设有K个类，样本点属于第k类的概率为$p_k$，则概率分布的基尼指数定义为
# $$Gini(D)=\sum_{k=1}^{K}p_k(1-p_k)=1-\sum_{k=1}^{K}p_k^2$$
# 其中，$p_k$=$\frac{|C_k|}{|D|}$表示数据集中第k类样本的比例，因此
# $$Gini(D)=1-\sum_{k=1}^{K}\left(\frac{|C_k|}{|D|}\right)^2$$
# 基尼指数的物理意义是，从数据集D中随机抽取两个样本，它们类别不一样的概率。因此基尼指数越小表明数据集D中同一类样本的数量越多，其纯度越高
# 这样，将数据集按属性a进行划分后的基尼指数为
# $$Gini(D,a)=\sum_{v=1}^{V}\frac{|D^v|}{|D|}Gini(D^v)$$

# https://blog.csdn.net/ShowMeAI/article/details/123401318

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Scikit-Learn回归树
# 决策树（DTs）是一种用于回归和分类的有监督学习方法。通常，决策树用于分类问题；当决策树用于回归问题时，称为回归树。回归树在机器学习知识结构中的位置如下：
#
# 回归树的目标是创建一个模型，通过学习从数据特性中推断出的简单决策规则来预测目标变量的值
# 回归树通过将数据集重复分割为不同的分支而实现分层学习，分割的标准是最大化每一次分离的信息增益。这种分支结构让回归树很自然地学习到非线性关系
# 决策树算法
# 决策树学习通常有三个步骤：特征选择；决策树生成；决策树剪枝
# 决策树学习通常是一个选择最优特征，并根据该特征对训练数据进行分割，使得各个子数据集有一个最好的分类的过程。这一过程对应着对特征空间的划分，也对应着决策树的构建
# 决策树算法的核心在于生成一棵决策树过程中，如何划分各个特征到树的不同分支上去
# 构建决策树常用的算法有：ID3、C4.5、C5.0和CART算法。其中ID3、C4.5和C5.0只能用于分类问题；CART既可以用于分类，也可以用于回归，因此更被广泛应用
# ID3(迭代分离3)
# ID3算法是由Ross Quinlan于1986年开发的。该算法建立了一个多路树，为每个节点(即贪婪地)寻找分类特征，从而为分类目标提供最大的信息增益。树生长到它们的最大尺寸，然后，通常采取一个剪枝步骤，以提高树的对位置数据的泛化能力
# ID3算法的核心是在决策树各个节点上使用信息增益作为选择特征的准则，信息增益大的优先选择，使用递归方法构建决策树。信息增益体现出特征与类的关联程度，即特征对类不确定性的影响程度
# C4.5
# C4.5是ID3的继承者，并且通过动态定义将连续属性值分割成一组离散间隔的离散属性（基于数字变量），从而消除了特征必须是分类的限制。C4.5将经过训练的树(即ID3算法的输出)转换成一组if-then规则的集合。然后对每条规则的这些准确性进行评估，以确定应用它们的顺序。如果规则的准确性没有提高的话，则需要决策树的树枝来解决
# C4.5算法是对ID3算法的改进，C4.5使用信息增益比来选择特征，以减少信息增益容易选择特征值多的特征的问题
# C5.0
# C5.0是Quinlan在专有许可下发布的最新版本。与C4.5相比，它使用更少的内存和构建更小的规则集，同时更精确
# CART（Classification And Regression Tree，分类与回归树）
# CART与C4.5非常相似，但它的不同之处在于它支持数值目标变量(回归)，不计算规则集。CART使用特征和阈值构造二叉树，在每个节点上获得最大的信息增益
# CART算法是根据基尼系数（Gini）来划分特征的，每次选择基尼系数最小的特征作为最优切分点
# Scikit-Learn使用CART算法的优化版本。
# CART的特点是：假设决策树是二叉树，内部结点特征的取值为是和否，右分支是取值为是的分支，左分支是取值为否的分支。这样的决策树等价于递归地二分每个特征，将输入特征空间划分为有限个单元，并在这些单元上确定预测的概率分布，也就是在输入给定的条件下输出的条件概率分布

# 决策树修剪
# 决策树生成只考虑了对训练数据更好的拟合，可以通过对决策树进行剪枝，从而减小模型的复杂度，达到避免过拟合的效果


# |参数  | 说明  |
# |--|--|
# |`criterion` |  用于衡量节点(分支)划分质量的指标，默认为`mse`(均方误差)，父节点和叶子节点之间的均方误差将被用来作为特征选择的标准，这种方法通过使用叶子节点的均值来最小化L2损失。其他取值还有`friedman_mse`(费尔德曼均方误差)，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差；`mae`(平均绝对误差)，这种指标使用叶节点的中值来最小化L1损失      |
# |`splitter`| 用于在每个节点上选择划分的策略，默认为`best`(最佳划分)。其他还有`random`(随机划分) |
# | `max_depth`| 决策树的最大深度，默认为None，表示将节点展开，直到所有叶子都是纯净的，或者直到所有叶子都包含少于`min_samples_split`个样本，即没有限制   |
# |`min_samples_split` | 拆分内部节点所需的最少样本数，默认为2，表示每个节点至少需要2个样本才能进行划分  |
# |`min_samples_leaf` | 在叶节点处需要的最小样本数，默认为1，表示每个叶子节点至少需要1个样本才能停止划分 |
# |`min_weight_fraction_leaf` |在所有叶节点处（所有输入样本）的权重总和中的最小加权分数，默认为0.0。如果未提供`sample_weight`，则样本的权重相等   |
# |`max_features` |寻找最佳划分时要考虑的特征数量，默认为None或`auto`，使用全部特征。其他取值还有`sqrt`(`sqrt(n_features)`)；`log2`(`log2(n_features)`)；也可使用`int`类型直接指定 |
# |`random_state` | 用于控制估算器的随机性，设置随机数生成器的种子  |
# |`max_leaf_nodes` |  用于控制决策树最多有多少个叶子节点，默认为None，叶子节点的数量不受限制   |
# | `min_impurity_decrease` |用于控制每个节点最少需要减少多少不纯度才能进行划分，默认值为0.0，表示每个节点至少需要减少0个不纯度才能进行划分  |
# | `ccp_alpha` | 用于最小代价复杂度修剪的复杂度参数，默认值为0.0，表示不执行修剪  |
# |`monotonic_cst` | 要对每个特征强制实施的单调性约束，默认为None，不应用任何约束。1表示单调递增；-1表示单调递减   |


# |属性  | 说明  |
# |--|--|
# |`feature_importances_` |  返回每个特征的重要程度，一维数组类型 |
# |`max_features_` |  拟合期间的特征数 |
# |`n_features_in_` |  拟合期间的特征名称 |

# |方法  | 说明  |
# |--|--|
# | `fit(X,y)` |  根据训练集构建决策树回归器 |
# | `predict(X)` | 预测X的类或回归值  |
# | `score(X,y)` | 返回模型的决定系数R^2  |
# | `get_depth()` |  返回决策树的深度 |
# | `get_n_leaves()` |  返回决策树的叶子数 |
# | `apply(X)` |  返回每个样本被预测为叶子的索引 |
# | `cost_complexity_pruning_path(X,y)` |  在最小化成本复杂性修剪期间计算修剪路径 |
# | `decision_path(X)` | 返回树中的决策路径  |


# Scikit-Learn回归树初体验

# # 创建正弦曲线数据集
# # 随机数种子，每次生成相同的随机数
# rng = np.random.RandomState(1)
# # rng.rand()：生成指定形状在[0,1)范围内均匀分布的随机数，下面生成80x1在[0,5)区间的随机数并排序
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# # ravel()：展开多维数组为一维数组，生成理想正弦函数的取值
# y = np.sin(X).ravel()
# # 添加噪声
# y[::5] += 3 * (0.5 - rng.rand(16))
# # 可视化
# plt.figure()
# plt.scatter(X, y, s=20, edgecolor='black', c='darkorange', label='data')
# # plt.show()
#
# from sklearn.tree import DecisionTreeRegressor
#
# # 训练决策树回归模型
# dt_reg1 = DecisionTreeRegressor(max_depth=2)
# dt_reg1.fit(X, y)
# dt_reg2 = DecisionTreeRegressor(max_depth=5)
# dt_reg2.fit(X, y)
#
# # 预测
# # 创建测试集
# # [:, np.newaxis]：用于将一维数组升维
# # [np.newaxis, :]：用于将一维数组升维后并转置
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# # 在测试集上预测
# y_pred1 = dt_reg1.predict(X_test)
# y_pred2 = dt_reg2.predict(X_test)
#
# # 可视化
# plt.plot(X_test, y_pred1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_pred2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import datasets
#
#
# # 数据集简介
# COLS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
#
# # 波士顿房价数据集
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# boston_df = pd.DataFrame(data)
# boston_df.columns = COLS
# # print(boston_df.head().to_string())
# target = raw_df.values[1::2, 2]
# boston_df['MEDV'] = target
# # print(boston_df.head().to_string())
#
#
# from sklearn.model_selection import train_test_split     # 数据集划分
# from sklearn.metrics import mean_squared_error as MSE    # 均方误差
# from sklearn.tree import DecisionTreeRegressor           # 回归树模型
#
# # 划分数据集
# # 特征
# X = boston_df.iloc[:, 0: 13]
# # 标签
# y = np.array(boston_df.iloc[:, -1])
#
# # 将数据集划分为训练集和测试集（随机采样20%的数据作为测试样本，其余作为训练样本）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # 训练模型
# # CART回归树
# dt_reg = DecisionTreeRegressor(max_depth=4)
# # 拟合训练集
# dt_reg.fit(X_train, y_train)
# # 在测试集上预测
# y_pred = dt_reg.predict(X_test)
# # 模型评估
# # 拟合程度：决定系数/相关系数R^2
# print(dt_reg.score(X_test, y_test))        # 0.867184400389605
# # 均方误差MSE
# print(MSE(y_test, y_pred))                 # 14.709066723474084
# # 均方根误差RMSE
# print(MSE(y_test, y_pred, squared=False))  # 3.8352401128839486
#
# # 可视化
# # 设置画板尺寸
# plt.figure(figsize=(15, 6))
# # 设置字体
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.title('波士顿房价预测曲线与实际曲线对比图', fontsize=15)
# # x轴数据
# x = range(len(y_test))
# # 实际价格曲线
# plt.plot(x, y_test, color='r', label='实际房价')
# # 预测价格曲线
# plt.plot(x, y_pred, color='g', ls='--', label='预测房价')
# # 显示图例
# plt.legend(fontsize=12, loc=1)
# plt.show()




