
# Scikit-Learn随机森林

# Scikit-Learn随机森林回归

# 随机森林是一种由决策树构成的集成算法，它在大多情况下都能有不错的表现。随机森林既可用于回归也可用于分类。随机森林回归在机器学习知识结构中的位置如下：
# 随机森林概述
# 随机森林是一种由决策树构成的（并行）集成算法，属于Bagging类型，随机森林通过组合多个弱分类器，最终结果通过投票或取均值，使得整体模型的结果具有较高的精确度和泛化性能，同时也有很好的稳定性，因此广泛应用在各种业务场景中
# 随机森林有如此优良的表现，主要归功于随机和森林。顾名思义，随机森林是一个比喻，它由若干决策树构成，每棵决策树都是其基本单元。至于随机，只是一个数学抽样概念。随机使它具有抗过拟合能力，森林使它更加精准
# 随机森林的基本思想在于集思广益，集中群众的智慧，广泛吸收有益的意见。这往往可以得到更好的解决方案。集思广益在机器学习中对应一个关键概念——集成学习
# 集成学习
# 集成学习（Ensemble Learning）通过训练学习多个个体学习器，当预测时通过结合策略将多个个体学习器的结果组合作为最终强学习器的结果输出
# 对于训练数据集，我们训练一系列个体学习器，再通过结合策略将它们集成起来，形成一个更强的学习器，这就是集成学习所做的事情
# 图
# 个体学习器是相对于集成学习来说的，其实我们在之前了解到的很多模型，例如决策树算法、朴素贝叶斯算法等，都是个体学习器
# 而集成可以分为同质集成和异质集成：
# - 同质集成：只包含同种类型的个体学习器，个体学习器称作基学习器。例如随机森林中全是决策树集成
# - 异质集成：包含不同类型的个体学习器，个体学习器称作组件学习器。例如同时包含决策树和神经网络进行集成
# 个体学习器代表的是单个学习器，集成学习代表的是多个学习器的结合
# 集成学习的核心问题有两个：
# - 使用什么样的个体学习器？
#   - 准确性：个体学习器不能太弱，需要有一定的准确性
#   - 多样性：个体学习器之间要存在差异性，即具有多样性
# - 如何选择合适的结合策略构建强学习器？
#   - 并行组合方式：例如随机森林
#   - 传统组合方式：例如Boosting树模型
# Bagging方法
# 这里我们只讲随机森林的并行集成模型，而Bagging是并行式集成学习方法最著名的代表
# Bagging方法全称为自助聚集（Bootstrap Aggregating），顾名思义，Bagging由Bootstrap与Aggregating两部分组成
# 自助采样法
# 要理解Bagging，首先要了解自助采样法（Bootstrap Sampling）
# 图
# 自助采样的过程为
# - 给定包含m个样本的数据集，先随机取出一个样本放入采样集中，再把该样本放回初始数据集，使得下次采样时该样本仍有可能被选中
# - 重复上述过程m轮，得到包含m个样本的采样集，初始数据集中有的样本在采样集中多次出现，有的则从未出现
# - 假设约63.2%的样本出现在采样集中，而未出现的约36.8%的样本可用作验证集来对后续的泛化性能进行包外估计
# Bagging方法是在自助采样基础上构建的，上述的采样过程我们可以重复T次，采样出T个包含m个样本的采样集，然后基于每个采样集训练出一个基学习器，然后将这些基学习器进行结合
# 在对预测输出进行结合时，Bagging通常对分类任务使用简单投票法，对回归任务使用简单平均法，这就是Bagging方法的基本流程
# 图
# 从偏差-方差分解的角度看，Bagging主要关注降低方差，因此它在不剪枝的决策树、神经网络等易受到样本扰动的学习器上效用更明显
# 随机森林算法
# 随机森林（Random Forest，RF）是一种基于树模型的Bagging的优化版本。核心思想依旧是Bagging，但是做了一些独特的改进——RF使用了CART决策树作为基学习器。具体过程如下：
# + 输入样本集$D$=$\{{ (x_1,y_1),(x_2,y_2),...,(x_m,y_m) \}}$
# + 对训练集进行第t(`t=1,2,...,T`)次随机采样，每次采样m次，得到包含m个样本的采样集$D_T$
# + 用采样集$D_T$训练第T个决策树模型$G_T(x)$，在训练决策树划分节点的时候，在节点上所有的样本特征中选择一部分样本特征，在这些随机选择的部分样本特征中选择一个最优的特征来做决策树的左右子树划分
# + 回归场景中，取T个基模型（决策树）的平均预测值；分类场景中，在T个基模型（决策树）中投出最多票数的类别为最终类别
# 构造随机森林的过程可以总结为以下四个步骤：
# 图
# 值得注意的是，随机森林中的每棵决策树的采用最佳节点划分，直到无需划分或不能划分为止，整个决策树在形成过程中没有剪枝操作
# 随机森林的核心特点是随机和森林，也是给它带来良好性能的最大支撑
# 随机主要体现在两个方面：
# + 样本扰动：直接基于自助采样法，使得一部分初始训练集样本出现在一个采样集中，并带来数据集的差异化
# + 属性扰动：对于基决策树的每个节点，先在该节点的特征属性集合中随机选择k个属性，然后再从这k个属性中选择一个最优属性进行划分。这一重随机性也会带来基模型的差异性
# 森林主要体现在：
# + 集成多个（差异化）采样集，训练得到多个（差异化）决策树，采用简单投票或者平均法来提高模型稳定性和泛化性能
# 随机森林的优缺点
# 优点：
# - 并行集成，训练速度快，可以有效控制过拟合
# - 每棵树都随机有放回选择部分样本，对存在缺失值或不平衡的数据集友好，可以平衡误差
# - 每棵树都随机选择部分特征，并可以自动识别非常重要的特征（自动特征选择）
# - 适用于高维（特征很多）数据，并且不用降维，无需做特征选择，因此成为降维的不错工具
# - 可以借助模型构建组合特征，鲁棒性强，可以维持不错的准确度
# 缺点：
# - 相比单一决策树，因其随机性，模型的可解释性更差
# - 在噪声过大的分类和回归数据集上还是可能会过拟合
# - 在多个分类变量的问题中，随机森林可能无法提高基学习器的准确性
# - 对于有不同取值的属性的数据，取值划分较多的属性会对随机森林产生更大的影响，所以随机森林在这种数据上产出的属性权值是不可信的

# 随机森林回归实践（加州房价预测）

from sklearn.datasets import fetch_california_housing as fch  # 加州房价数据集
from sklearn.model_selection import train_test_split          # 数据集划分
from sklearn.metrics import mean_squared_error as MSE         # 均方误差

# 特征
X = fch().data
# 标签
y = fch().target

# 将数据集划分为训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor            # 随机森林回归模型

# 随机森林回归器
regr = RandomForestRegressor(oob_score=True)
# 训练数据，拟合模型
regr.fit(X_train, y_train)
# 在测试集上预测
y_pred = regr.predict(X_test)

# 模型评估
# 相关系数/决定系数R^2
print(regr.score(X_test, y_test))           # 0.7980079760141825
# 均方误差MSE
print(MSE(y_test, y_pred))                  # 0.2633884899047907
# 均方根误差RMSE
print(MSE(y_test, y_pred, squared=False))   # 0.5132138831956816

from matplotlib import pyplot as plt

# 可视化
# 设置画板尺寸
plt.figure(figsize=(15, 6))
# 设置字体
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.title('加州房价预测曲线与实际曲线对比图', fontsize=15)
# x轴数据（由于数据较多，这里只取测试集的前100个数据）
x = range(len(y_test))[0:100]
# 实际价格曲线
plt.plot(x, y_test[0:100], color='r', label='实际房价')
# 预测价格曲线
plt.plot(x, y_pred[0:100], color='g', ls='--', label='预测房价')
# 显示图例
plt.legend(fontsize=12, loc=1)
plt.show()

# https://xietao.site/sklearn_RF
# https://www.cnblogs.com/shanger/articles/11901533.html


# Scikit-Learn随机森林分类

# 随机森林是一种基于集成学习（Ensemble Learning）的机器学习算法。随机森林既可用于回归也可用于分类。随机森林分类在机器学习知识结构中的位置如下：
# 图
# 随机森林的随机性主要体现在两个方面：一是决策树训练样本的随机选取，二是决策树节点划分属性特征的随机选取
# 这两个随机性的目的是降低森林估计器的方差。事实上，单个决策树通常表现出很高的方差，并且往往会过拟合。在森林中注入随机性产生的决策树具有一定的解耦预测误差（Decoupled Prediction Errors)。通过取这些预测的平均值或投票，可以抵消掉一些误差
# 随机森林属于集成学习中的Bagging（Bootstrap Aggregating）中的方法。它们之间的关系如下
# 图
# 随机森林分类通过引入随机性来构建多个决策树，再通过对这多个决策树的预测结果进行投票以产生最终的分类结果
# 随机森林分类算法可以应用于各种需要进行分类或预测的问题，例如，垃圾邮件识别、信用卡欺诈检测等，它也可以与其他机器学习算法进行结合，以进一步提高预测准确率

# 随机森林算法的构造过程如下：
# + 从原始数据集中有放回的随机选择一部分样本，构成一个子样本集，每棵决策树都在不同子样本集上进行训练，增加模型的多样性
# + 对于每棵决策树的每个节点，随机选择一部分属性，然后选择最佳划分属性，每棵决策树的每个节点都基于随机选择的部分属性，提高模型的鲁棒性
# + 在每个子样本集上构建决策树，在决策树生长的过程中，每个节点都基于随机选择的部分属性选择最佳划分属性，直到不能分裂为止
# + 建立大量决策树，形成随机森林

# 在随机森林中，不同决策树之间没有关联。当我们进行分类任务时，新的输入样本进入，就让森林中的每一棵决策树分别进行判断和分类，每个决策树会得到一个自己的分类结果，决策树的分类结果中哪一个分类最多，那么随机森林就会把这个结果当做最终的结果

# 随机森林分类的优缺点
# 优点：
# + 抗过拟合能力强：采用随机选择样本数据和特征的方式，可以有效地避免过拟合问题
# + 泛化能力强：通过对多个决策树的结果进行投票，可以获得更好的泛化性能
# + 对数据特征的选取具有指导性：在构建决策树时会对特征进行自动选择，这可以为后续的特征选择提供指导
# + 适用于大规模数据集：可以有效地处理大规模数据集，并且训练速度相对较快
# 缺点：
# + 需要大量的内存和计算资源：由于需要构建多个决策树，因此需要更多的内存和计算资源
# + 需要调整参数：性能很大程度上取决于参数的设置，如树的数量、每个节点的最小样本数等，这些参数的设置需要一定的经验和实验
# + 对新样本的预测性能不稳定：由于是通过投票多个决策树的结果来进行预测，因此对新样本的预测性能可能会受到影响


# 案例：葡萄酒分类
from sklearn.datasets import load_wine                  # 葡萄酒数据集
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.ensemble import RandomForestClassifier     # 随机森林分类器

# 加载数据集
wine = load_wine()
# 特征
X = wine.data
# 标签
y = wine.target

# 划分训练集(80%)和测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 随机森林分类器
clf = RandomForestClassifier(random_state=0)
# 训练数据，拟合模型
clf.fit(X_train, y_train)
# 在测试集上预测
print(y_test)                # [0 2 1 0 1 1 0 2 1 1 2 2 0 1 2 1 0 0 1 0 1 0 0 1 1 1 1 1 1 2 0 0 1 0 0 0]
print(clf.predict(X_test))   # [0 2 1 0 1 1 0 2 1 1 2 2 0 1 2 1 0 0 2 0 1 0 0 1 1 1 1 1 1 2 0 0 1 0 0 0]
# 模型评估：准确度评分
print(clf.score(X_test, y_test))  # 0.9722222222222222


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def decision_boundary_plot(model, X, y):
    # np.meshgrid()：定义X/Y坐标轴上的起始点和结束点以及点的密度，将区域边界规定为特征值的min-0.5及max+0.5，生成坐标点阵，每个相邻坐标点之间的距离为0.02
    X0, X1 = np.meshgrid(
        np.arange(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 0.02),
        np.arange(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 0.02)
    )
    # ravel()：将高维数组降为一维数组
    # np.c_[]：将两个数组以列的形式拼接起来形成矩阵，这里将上面每个网格点的X和Y坐标组合
    X_grid_matrix = np.c_[X0.ravel(), X1.ravel()]
    # 通过训练好的模型，预测平面上这些网格点的分类
    y_grid_pred = model.predict(X_grid_matrix)
    y_pred_matrix = y_grid_pred.reshape(X0.shape)
    # ListedColormap()：自定义色彩列表
    cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_sample = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    # plt.pcolormesh(X,Y,Z)：根据分类值来选择colormap中对应的颜色进行背景填充，cmap用于设置填充轮廓；alpha用于设置填充透明度
    plt.pcolormesh(X0, X1, y_pred_matrix, alpha=0.90, cmap=cmap_background)
    # plt.scatter()：绘制散点图，edgecolors用于设置边缘/边框颜色
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_sample, edgecolors='k', s=20)


# 鸢尾花分类
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.ensemble import RandomForestClassifier     # 随机森林分类器
from sklearn import datasets

# 加载数据集
iris = datasets.load_iris()

# 使用鸢尾花前两个特征的全部分类（三分类）
X = iris.data[:, :2]
y = iris.target

# 随机森林分类器
clf = RandomForestClassifier()
# 使用全部数据拟合模型
clf.fit(X, y)
# 准确度评分
print(clf.score(X, y))   # 0.9266666666666666

# 绘制决策边界
decision_boundary_plot(clf, X, y)
plt.show()


iris = datasets.load_iris()

# 使用鸢尾花前两个种类的前两个特征（二分类）
X = iris.data[iris.target < 2, :2]
y = iris.target[iris.target < 2]
# 使用鸢尾花前两个特征的全部分类（三分类）
# X = iris.data[:, :2]
# y = iris.target
# 全部特征
# X = iris.data
# y = iris.target

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 随机森林分类器
clf = RandomForestClassifier(max_depth=3, min_samples_leaf=2)
# 训练数据，拟合模型
clf.fit(X_train, y_train)
# 在测试集上预测（二分类）
print(y_test)                # [0 1 0 1 1 0 1 0 0 0 1 1 0 1 1 1 0 1 0 0]
# 三分类：[2 0 0 1 0 1 2 0 0 2 2 2 0 2 1 2 0 2 1 0 2 1 1 1 2 1 1 0 2 1]
print(clf.predict(X_test))   # [0 1 0 1 1 0 1 0 0 0 1 1 0 1 1 1 0 1 0 0]
# 三分类：[1 0 0 1 0 1 2 0 0 1 2 1 0 2 1 2 0 2 1 0 2 2 1 2 1 1 1 0 1 1]
# 模型评估：准确度评分（二分类）
print(clf.score(X_test, y_test))       # 1.0
# 三分类：0.7666666666666667
# 全部特征：0.9333333333333333

# 绘制决策边界
decision_boundary_plot(clf, X, y)
plt.show()

# 参数调优与选择
# 在随机森林算法众多参数中，影响随机森林的核心参数有
# + max_features：生成单棵决策树划分节点时考虑的最大特征数
#   增加`max_features`一般能提高单个决策树模型的性能，但会降低树和树之间的差异性，且可能降低算法的速度
#   不同太小的`max_features`会影响单棵树的性能，进而影响整体的集成效果，需要适当地平衡和选择最佳`max_features`
#   `max_features`一般取`float`类型，表示取特征总数的百分比，常见的选择区间是`[0.5,0.9]`
# + n_estimators：决策树的棵树
#   较多的子树可以让模型有更好的稳定性和泛化能力，但同时会让模型的学习速度变慢
#   一般会在计算资源能支撑的情况下，选择稍大的子树棵树，可能会设置为＞50的取值，可根据计算资源调整
# + max_depth：决策树的最大深度
#   过大的树深，因为每颗子树都过度学习，可能会导致过拟合问题
#   如果模型样本量多特征多，我们会限制最大树深，提高模型的泛化能力，常见的选择在`[4,12]`之间
# + min_samples_split：内部节点再划分所需最小样本数
#   如果样本量不大，不需要调整这个值。如果样本量数量级非常大，我们可能会设置这个值为16、32、64等
# + min_samples_leaf：叶子节点最少样本数
#   为了提高泛化能力，我们可能会设置这个值＞1

from sklearn.model_selection import GridSearchCV    # 网格搜索与交叉验证

# 模型候选参数
params = {
    'criterion': ['gini', 'entropy'],      # 最优划分策略
    'max_features': [idx / 10 for idx in range(5, 10)],   # 单棵决策树划分节点时考虑的最大特征数
    'n_estimators': [50, 100, 150],        # 决策树的数量
    'max_depth': range(3, 13),             # 限制决策树的最大深度
    "min_samples_split": [2, 4],           # 将中间节点样本数量小于min_samples_split的剪掉
    'min_samples_leaf': range(2, 5)        # 将样本数量小于min_samples_leaf的叶子节点剪掉
}

# 网格搜索：搜索所有组合，评估每种组合
grid = GridSearchCV(
    RandomForestClassifier(random_state=0),
    param_grid=params,
    scoring='accuracy',
    cv=5
)

# 拟合所有组合
grid.fit(X_train, y_train)

# 最优评分（二分类）
print(grid.best_score_)       # 0.9875
# 三分类：0.7916666666666667
# 最佳参数（二分类）
print(grid.best_params_)      # {'criterion': 'gini', 'max_depth': 3, 'max_features': 0.5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
# 三分类：{'criterion': 'entropy', 'max_depth': 3, 'max_features': 0.5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
# 获取最佳评分时的模型（二分类）
print(grid.best_estimator_)   # RandomForestClassifier(max_depth=3, max_features=0.5, min_samples_leaf=2, n_estimators=50, random_state=0)
# 三分类：RandomForestClassifier(criterion='entropy', max_depth=3, max_features=0.5, min_samples_leaf=2, n_estimators=50, random_state=0)

from sklearn.metrics import accuracy_score

# 基于最佳参数建模
best_clf = grid.best_estimator_
# 训练模型
best_clf.fit(X_train, y_train)
# 在测试集上预测
y_pred = best_clf.predict(X_test)
# 准确度评分（二分类）
print(accuracy_score(y_test, y_pred))   # 0.9
# 三分类：0.7666666666666667

# 绘制决策边界
decision_boundary_plot(best_clf, X, y)
plt.show()





