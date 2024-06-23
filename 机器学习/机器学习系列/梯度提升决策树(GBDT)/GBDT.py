

# https://www.showmeai.tech/article-detail/193

# GBDT梯度提升决策树

# 梯度提升决策树（GBDT）是集成学习Boosting提升中的一种重要算法。GBDT在机器学习知识结构中的位置如下：
# 图
# 梯度提升决策树（Gradient Boosting Decision Tree，GBDT）是一种迭代的决策树算法，它通过构造一组弱学习器（决策树），并把多颗决策树的结果累加起来作为最终的预测输出。该算法将决策树与集成思想进行了有效的结合
# Boosting方法
# 我们已经知道，Bagging方法在训练过程中，各基分类器之间无强依赖，可以进行并行训练
# 与Bagging方法不同，Boosting方法在训练基分类器时采用串行的方式，各个基分类器之间有依赖
# Boosting的基本思路是将基分类器层层叠加，每一层在训练的时候，对前一层基分类器分错的样本，给予更高的权重。预测时，根据各层分类器的结果的加权得到最终结果

# GBDT算法的原理
# 梯度提升决策树的算法流程如下：
# + 初始化：设定一个初始预测值，通常为所有样本目标变量的均值，这个初始预测值代表了我们对目标变量的初始猜测
# + 迭代训练：GBDT是一个迭代算法，通过多轮迭代来逐步改进模型。在每一轮迭代中，GBDT都会训练一棵新的决策树，目标是减少上一轮模型的残差。残差是预测值与真实值之间的误差，新的树将学习如何纠正这些残差
#   + 计算残差：在每轮迭代开始时，计算当前模型对训练数据的预测值与实际观测值之间的残差。这个残差代表了前一轮模型未能正确预测的部分
#   + 拟合残差：以当前残差为新的学习目标变量，训练一棵新的决策树。这棵树将尝试纠正上一轮模型的错误，以减少残差
#   + 更新模型：将新训练的决策树与之前的模型进行组合，更新预测值。具体地，将新树的预测结果与之前模型的预测结果累加，得到更新后的模型
# + 终止/集成：当达到预定的迭代次数或残差变化小于阈值时停止迭代。将所有决策树的预测结果相加，得到最终的集成预测结果。这个过程使得模型能够捕捉数据中的复杂关系，从而提高预测精度
# GBDT的核心点在于不断迭代，每一轮迭代都尝试修正上一轮模型的错误，逐渐提高模型的预测性能
# 例如，我们用GBDT去预测年龄：
# 图 说明
# GBDT回归
# 回归任务下，GBDT在每一轮迭代时对每个样本都会有一个预测值，此时的损失函数为均方差损失函数：
# 图
# 损失函数的负梯度计算如下：
# 图
# 可以看到，当损失函数选用均方误差损失时，每次拟合的值就是真实值减预测值，即残差

# GBDT的训练过程
# 以下通过案例（根据行为习惯预测年龄）帮助我们深入理解梯度提升决策树（GBDT）的训练过程
# 假设训练集有4个人（A、B、C、D），他们的年龄分别是14、16、24、26。其中A、B分别是高一和高三学生；C、D分别是应届毕业生和工作两年的员工
# 下面我们将分别使用回归树和GBDT，通过他们的日常行为习惯（购物、上网等）预测每个人的年龄
# 1、使用回归树训练
# 回归树训练得到的结果如图所示：
# 图
# 2、使用GBDT训练
# 由于我们的样本数据较少，所以我们限定叶子节点最多为2（即每棵树都只有一个分枝），并且限定树的棵树为2
# 梯度提升决策树（GBDT）的训练过程如下：
# 1）第一棵树：假设初始值为平均年龄20，得到的结果如图所示：
# 图
# 上图中，A、B的购物金额不超过1k，C、D的购物金额超过1k，因此被分为左右两个分支，每个分支使用平均年龄作为预测值
# 分别计算A、B、C、D的残差（实际值减预测值）：
# + A残差=14-15=-1
# + B残差=16-15=1
# + C残差=24-25=-1
# + D残差=26-25=1
# 以A为例，这里A的预测值是指前面所有树预测结果的累加和，当前由于只有一棵树，所以直接是15，其他同理
# 2）第二棵树：拟合前一棵树的残差-1、1、-1、1，得到的结果如图所示：
# 图
# 上图中，A、C的上网时间超过1h，B、D的上网时间不超过1h，因此被分为左右两个分支，每个分支使用平均残差作为预测值
# 分别计算A、B、C、D的残差（实际值减预测值）：
# + A残差=-1-(-1)=0
# + B残差=1-1=0
# + C残差=-1-(-1)=0
# + D残差=1-1=0
# 第二棵树学习第一棵树的残差，在当前这个简单场景下，已经能够保证预测值与实际值（上一轮残差）相等了，此时停止迭代
# 3）迭代终止后，最后就是集成，累加所有决策树的预测结果作为最终GBDT的预测结果
# 图
# 本案例中，我们最终得到GBDT的预测结果为第一棵树的预测结果加第二棵树的预测结果
# + A：真实年龄14岁，预测年龄15+(-1)=14
# + B：真实年龄16岁，预测年龄15+1=16
# + C：真实年龄24岁，预测年龄25+(-1)=24
# + D：真实年龄26岁，预测年龄25+1=26
# 综上所述，GBDT需要将多棵树的预测结果累加，得到最终的预测结果，且每轮迭代都是在当前树的基础上，增加一棵新树去拟合前一个树预测值与真实值之间的残差

# 梯度提升与梯度下降
# 负梯度方向是梯度下降最快的方向。梯度下降与梯度提升两种迭代优化算法都是在每一轮迭代中，利用损失函数负梯度方向的信息，更新当前模型。但两者完全不同
# 梯度下降中，模型是以参数化形式表示，从而模型的更新等价于参数的更新
# 图
# 梯度提升中，模型并不需要进行参数化表示，而是直接定义在函数空间中，从而大大扩展了可以使用的模型种类
# 图
# |比较项 | 梯度下降 |  梯度提升  |
# |模型定义空间|  参数空间  | 函数空间 |
# |优化规则| $\theta_t$=$\theta_{t-1}$+$\Delta\theta_t$  | $f_t(x)$=$f_{t-1}(x)$+$\Delta f_t(x)$  |
# |损失函数| $L$=$\sum_{t}^{}l(y_t,f(x;\theta_t))$  | $L$=$\sum_{t}^{}l(y_t,F(x_t))$  |
# |描述| 参数更新的方向为负梯度方向，第t次迭代的参数等于第t-1次迭代的参数加上第t次迭代的参数增量  |  函数更新的方向为负梯度方向，第t次迭代的函数等于第t-1次迭代的函数加上第t次迭代的函数增量  |
# |最终结果| 最终的参数等于每次迭代的参数增量的累加和，$\theta_0$为初始值 | 最终的函数等于每次迭代的函数增量的累加和，$f_0(x)$为初始值 |

# 随机森林与GBDT
# 随机森林与GBDT都是集成树模型算法，GBDT与随机森林的异同点如下：
# 相同点：
# + 都是集成模型，由多棵树构成，最终的结果都是由多棵树一起决定
# + 随机森林和GBDT都是使用CART树，可以是分类树或者回归树
# 不同点：
# + 训练过程中，随机森林中的树可以并行生成；而GBDT中的树只能串行生成
# + 随机森林随机抽取部分特征构建树；而GBDT使用全部特征构建树
# + 随机森林的结果是取多棵树的平均或多数表决；而GBDT的结果是多棵树的结果累加
# + 随机森林对异常值不敏感；而GBDT对异常值较敏感
# + 随机森林是降低模型的方差；而GBDT是降低模型的偏差

# GBDT的优缺点
# 优点：
# + 可解释性强。每棵决策树都可以看作一个规则集合，模型的预测结果可通过查看各棵树的决策路径进行解释
# + 鲁棒性强。决策树的局部学习特性使得GBDT对异常值较为稳健，不易受个别噪声点影响
# + 准确性高，泛化性能好。通过梯度提升策略，GBDT能够逐步减少预测残差，构建出具有高预测精度的模型
# + 运算效率高。适用于高维数据，预测阶段，因为每棵树的结构都已确定，可并行化计算，计算速度快
# + 支持多种任务。GBDT既可以用于回归任务，也可以通过设置不同的目标函数应用于分类任务
# 缺点：
# + GBDT在高维稀疏的数据集上，效率较差，且效果表现不如SVM或神经网络
# + 适合数值型特征，在NLP或文本特征上表现弱
# + 训练过程无法并行，工程加速只能体现在单颗树构建过程中
# + 虽然单颗决策树对异常值鲁棒，但若异常值影响了残差计算，可能会导致后续决策树过度拟合这些异常点
# + 若不加以限制，随着迭代次数增加，模型复杂度增大，可能导致过拟合。需通过设置最大深度、学习率、早停等策略进行控制


# GBDT回归实践（加州房价预测）

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing as fch
from sklearn.model_selection import train_test_split    # 数据集划分

# 加载数据集
# 特征
X = fch().data
# 标签
y = fch().target

# 划分训练集（80%）与测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import GradientBoostingRegressor  # GBDT回归

# GBDT回归器，默认使用均方差损失函数（最小二乘）
reg = GradientBoostingRegressor(learning_rate=0.1, random_state=0)
# 训练模型，拟合数据
reg.fit(X_train, y_train)
# 在测试集上预测
y_pred = reg.predict(X_test)

from sklearn.metrics import mean_squared_error as MSE   # 均方误差

# 模型评估
# 拟合程度、决定系数R^2
print(reg.score(X_test, y_test))           # 0.7773617035611023
# 均方误差
print(MSE(y_test, y_pred))                 # 0.29031029808451114
# 平均绝对误差
print(MSE(y_test, y_pred, squared=False))  # 0.5388045082258602

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


# GBDT分类实践（鸢尾花分类）

from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier  # GBDT分类

# 加载数据集
iris = datasets.load_iris()

# 使用鸢尾花前两个种类的前两个特征（二分类）
# X = iris.data[iris.target < 2, :2]
# y = iris.target[iris.target < 2]
# 使用鸢尾花前两个特征的全部分类（三分类）
X = iris.data[:, :2]
y = iris.target
# 全部特征
# X = iris.data
# y = iris.target

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# GBDT分类器
clf = GradientBoostingClassifier(max_depth=10, criterion='squared_error', random_state=0)
# 训练数据，拟合模型
clf.fit(X_train, y_train)
# 在测试集上预测（二分类）
print(y_test)                # [1 0 0 1 1 1 1 0 1 1 0 0 1 1 1 1 1 1 1 1]
# 三分类：[1 2 0 0 0 2 2 1 0 0 1 0 0 1 2 1 2 2 2 0 2 0 2 0 1 2 2 2 1 0]
print(clf.predict(X_test))   # [1 0 0 1 1 1 1 0 1 1 0 0 1 1 1 1 1 1 1 1]
# 三分类：[1 2 0 0 0 2 2 2 0 0 2 0 0 0 2 1 2 2 2 0 2 0 1 0 1 2 2 1 1 0]

# 决策树最大深度
print(clf.max_depth)         # 3
# 决策树最小叶子数
print(clf.min_samples_leaf)  # 1

# 模型评估：准确度评分（二分类）
print(clf.score(X_test, y_test))       # 1.0
# 三分类：0.8333333333333334
# 全部特征：0.9


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


# 绘制决策边界
# decision_boundary_plot(clf, X, y)
# plt.show()


# GBDT参数调优与选择

# GBDT回归与GBDT分类两者的参数基本相同，这些参数中，其中核心的我们一般分为两类：第一类是弱学习器（CART树）的核心参数；第二类是Boosting框架的核心参数

# 弱学习器核心参数
# GBDT使用了弱学习器CART回归/决策树，因此它的参数优化基本来源于回归/决策树的参数优化。在弱学习器中，影响GBDT性能的核心参数有
# + max_features：生成单棵决策树划分节点时考虑的最大特征数
# + max_depth：决策树的最大深度
# + min_samples_split：内部节点再划分所需最小样本数
# + min_samples_leaf：叶子节点最少样本数
# + max_leaf_nodes：最大叶子节点数，通过限制最大叶子节点数（剪枝），可以防止过拟合。如果特征较少，可以不考虑这个值；如果特征较多，可以加以限制

# Boosting框架核心参数
# 影响GBDT性能的Boosting框架核心参数有
# + n_estimators：弱学习器（决策树）的数量，也可以说是弱学习的最大迭代次数。`n_estimators`过大，容易过拟合；`n_estimators`过小，容易欠拟合。通常与参数`learning_rate`一起考虑
# + learning_rate：每个弱学习器的正则化项系数、步长/学习率，取值范围为`(0,+∞]`。对于同样的训练集拟合效果，较小的学习率意味着我们需要更多的弱学习器（迭代次数），通常与`n_estimators`一起考虑
# + subsample：子采样，子样本数的比例，取值范围为`(0,1]`，默认为1，不使用子采样。需要注意的是，这里的子采样和随机森林不一样，随机森林是有放回抽样，而这里是不放回抽样。如果取值小于1，则只有一部分样本用于GBDT决策树拟合。选择小于1的比例可以减小方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐区间为`[0.5,0.8]`
# + init：初始弱学习器$f_0(x)$，默认情况下，使用训练集样本和预测类DummyEstimator预测平均目标值。如果为`zero`，初始化原始预测为0
# + loss：GBDT算法中的损失函数。分类模型与回归模型的损失函数不同
#   + 回归：可选参数有均方差损失`squared_error`（默认）、绝对损失`absolute_error`、Huber损失`huber`和分位数损失`quantile`。一般地，如果数据的噪音点不多，使用均方差损失较好。如果噪音点较多，则推荐使用抗噪损失`huber`。如果需要对训练集进行分段预测，则采用`quantile`
#   + 分类：可选参数有对数似然损失`log_loss`（默认）和指数损失`exponential`。推荐使用对数似然损失，它对二元分类和多元分类各自都有较好的优化。指数损失等同于Adaboost算法
# + alpha：仅GBDT回归器的参数，Huber损失和分位数损失的分位数，取值范围为`(0.0,1.0)`。如果噪音点较多，可以适当降低这个分位数值


# 案例（鸢尾花分类）
from sklearn.model_selection import GridSearchCV    # 网格搜索与交叉验证

# 模型候选参数（参数空间）
params = {
    'max_features': [idx / 10 for idx in range(5, 10)],   # 单棵决策树划分节点时考虑的最大特征数
    'max_depth': range(3, 13),             # 限制决策树的最大深度
    "min_samples_split": [2, 4],           # 将中间节点样本数量小于min_samples_split的剪掉
    'min_samples_leaf': range(2, 5),       # 将样本数量小于min_samples_leaf的叶子节点剪掉
    'n_estimators': [50, 100, 150],        # 决策树的数量、最大迭代次数
    'subsample': [0.5, 0.6, 0.7, 0.8],     # 子采样
}

# 网格搜索：搜索所有组合，评估每种组合
grid = GridSearchCV(
    GradientBoostingClassifier(random_state=0),
    param_grid=params,
    scoring='roc_auc',   # roc_auc只适用于二分类，三分类修改为准确度accuracy
    cv=5
)

# 拟合所有组合
grid.fit(X_train, y_train)

# 最优ROC-AUC评分（二分类）
print(grid.best_score_)       # 1.0
# 三分类（准确度Accuracy）：0.7666666666666667
# 最佳参数（二分类）
print(grid.best_params_)      # {'max_depth': 3, 'max_features': 0.5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50, 'subsample': 0.5}
# 三分类：{'max_depth': 9, 'max_features': 0.5, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 50, 'subsample': 0.5}
# 获取最佳评分时的模型（二分类）
print(grid.best_estimator_)   # GradientBoostingClassifier(max_features=0.5, min_samples_leaf=2, n_estimators=50, random_state=0, subsample=0.5)
# 三分类：GradientBoostingClassifier(max_depth=9, max_features=0.5, min_samples_leaf=3, n_estimators=50, random_state=0, subsample=0.5)

# 基于最佳参数建模
from sklearn.metrics import roc_auc_score

# 基于最佳参数建模
best_clf = grid.best_estimator_
# 训练模型
best_clf.fit(X_train, y_train)
# 在测试集上预测
y_pred = best_clf.predict(X_test)
# 平均准确度（二分类）
print(best_clf.score(X_test, y_test))   # 0.95
# 三分类：0.7
# ROC曲线下面积AUC（二分类：直接使用y_test和y_pred）
print(roc_auc_score(y_test, y_pred, multi_class='ovo'))   # 0.9615384615384616
# ROC曲线下面积AUC（三/多分类：使用y_test和预测置信度）
# print(roc_auc_score(y_test, best_clf.predict_proba(X_test), multi_class='ovr'))   # 0.8916820671206636

# 绘制决策边界
decision_boundary_plot(best_clf, X, y)
plt.show()

# 绘制ROC曲线（只适用于二分类）
from sklearn.metrics import RocCurveDisplay

# 设置正常显示符号
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# plot_chance_level=True：绘制对角线
display = RocCurveDisplay.from_predictions(
    y_test,
    y_pred,
    plot_chance_level=True
)
plt.title("ROC curve（AUC）")
plt.show()



