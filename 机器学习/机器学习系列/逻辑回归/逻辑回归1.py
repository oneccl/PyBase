
# Sigmoid函数图像

# import numpy as np
# import matplotlib.pyplot as plt
#
# z = np.arange(-5, 5, 0.01)
# y = 1/(1+np.exp(-z))
#
# plt.plot(z, y)
# plt.show()


# 逻辑回归中的梯度下降法实现
import numpy as np
from sklearn.metrics import accuracy_score     # 精确率

# 封装使用梯度下降法求解逻辑回归的最优解的类（训练数据、曲线拟合）
class GradLogisticRegression(object):

    def __init__(self):
        # theta初始点列向量
        self._theta = None
        # 系数
        self.coef_ = None
        # 截距
        self.intercept_ = None

    # 定义Sigmoid函数
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit_grad(self, X_train, y_train, alpha=0.01, n_iters=1e4):

        # 定义代价函数（损失函数）
        def cost(X, Y, m, theta):
            y_hat = self._sigmoid(X.dot(theta))
            try:
                return -(1 / m) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1-y_hat))
            except:
                return float('inf')

        # 定义代价函数的梯度函数
        def grad(X, Y, m, theta):
            return (1 / m) * X.T.dot(self._sigmoid(X.dot(theta)) - Y)

        # 梯度下降迭代算法
        def gradient_descent(X, Y, m, theta, alpha, n_iters, diff=1e-8):
            i_iter = 0
            while i_iter < n_iters:
                gradient = grad(X, Y, m, theta)
                last_theta = theta
                theta = theta - alpha * gradient
                if abs(cost(X, Y, m, theta) - cost(X, Y, m, last_theta)) < diff: break
                i_iter = i_iter+1
            print(f"迭代次数: {i_iter}")
            return theta

        # 构建X
        X = np.hstack([np.ones((len(X_train), 1)), X_train])
        # 初始化theta向量为元素全为0的向量
        theta = np.zeros(X.shape[1])

        self._theta = gradient_descent(X, y_train, len(X), theta, alpha, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # 给定待预测数据集X_pred，返回X_pred的概率向量
    def predict(self, X_pred):
        # 构建X_p
        X_p = np.hstack([np.ones((len(X_pred), 1)), X_pred])
        # 返回0、1之间的浮点数（概率）
        probability = self._sigmoid(X_p.dot(self._theta))
        # 将概率转换为0和1，True对应1，False对应0（返回概率结果和分类结果）
        return np.array(probability >= 0.5, dtype='int')

    # 准确率评分
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)


# 莺尾花数据集
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

# # 查看数据的所有属性和方法
# print(dir(iris))    # ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']
# # 特征数据的形状大小
# print(iris.data.shape)      # (150, 4)
# # 特征名称
# print(iris.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# # 分类标签
# print(pd.Series(iris.target).value_counts())    # 依次为50个0、50个1、50个2
# '''
# 0    50
# 1    50
# 2    50
# Name: count, dtype: int64
# '''
# # 分类名称
# print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
#
# 将数据保存到DataFrame
# iris_df = pd.DataFrame(data=iris.data, columns=[e[:-4] for e in iris.feature_names])
# iris_df['class'] = iris.target
# print(iris_df.to_string())
# '''
#    sepal length   sepal width   petal length   petal width   class
# 0            5.1           3.5            1.4           0.2      0
# 1            4.9           3.0            1.4           0.2      0
# 2            4.7           3.2            1.3           0.2      0
# 3            4.6           3.1            1.5           0.2      0
# 4            5.0           3.6            1.4           0.2      0
# '''
# # 查看数据整体信息（缺失值检查）：无缺失值
# print(iris_df.info())
#
# # 可视化（2D）
# def show(X, y, data):
#     plt.plot(X[y == 0, 0], X[y == 0, 1], 'rs', label=data.target_names[0])
#     plt.plot(X[y == 1, 0], X[y == 1, 1], 'bx', label=data.target_names[1])
#     plt.plot(X[y == 2, 0], X[y == 2, 1], 'go', label=data.target_names[2])
#     plt.xlabel(data.feature_names[0])
#     plt.ylabel(data.feature_names[1])
#     plt.title("鸢尾花数据集（2D）")
#     plt.legend()
#     plt.rcParams['font.sans-serif'] = 'SimHei'  # 消除中文乱码
#     plt.show()
#
# # 前2列特征（平面只能展示2维）花萼长度(cm)、花萼宽度(cm)
# X = iris.data[:, :2]
# # 后2列特征
# # X = iris.data[:, 2:4]
# # 分类（目标）标签
# y = iris.target
# show(X, y, iris)


# 验证我们封装的逻辑回归
# 使用鸢尾花的前两个种类的前两个特征（二分类）
X = iris.data
y = iris.target
# X = X[y < 2, :2]
# y = y[y < 2]

# 划分训练集（80%）和测试集（20%）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # 梯度下降法逻辑回归分类器
# log_reg = GradLogisticRegression()
# # 训练模型
# log_reg.fit_grad(X_train, y_train)
# print(log_reg.coef_)         # [3.03026986 -5.08061492]
# print(log_reg.intercept_)    # -0.6938379190860661
# # 预测
# y_pred = log_reg.predict(X_test)
# # 预测结果与真实结果比较
# print(y_pred)    # [0 1 0 1 1 1 0 1 1 1 1 1 1 0 0 0 0 0 0 0]
# print(y_test)    # [0 1 0 1 1 1 0 1 1 1 1 1 1 0 0 0 0 0 0 0]
# # 模型评估
# print(log_reg.score(X_test, y_test))    # 1.0

# 绘‍‍制线性决策边界直线
# # 定义求X2的函数
# def X2(X1):
#     return (-log_reg.intercept_ - log_reg.coef_[0] * X1) / log_reg.coef_[1]
#
# # 构建X1
# X1 = np.linspace(4.5, 7, 1000)
# X2 = X2(X1)
#
# # 使用鸢尾花的前两个种类的前两个特征可视化
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
# plt.plot(X1, X2)
# plt.show()

# 逻辑回归二分类
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 使用鸢尾花的前两个种类的前两个特征（二分类）
X = iris.data
y = iris.target
X = X[y < 2, :2]
y = y[y < 2]

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 逻辑回归分类器（默认求解器lbfgs、分类方式OvR）
lr = LogisticRegression()
# 训练模型
lr.fit(X_train, y_train)
# 在测试集上预测
y_pred = lr.predict(X_test)
print(y_pred)    # [0 1 0 1 1 1 0 1 1 1 1 1 1 0 0 0 0 0 0 0]
print(y_test)    # [0 1 0 1 1 1 0 1 1 1 1 1 1 0 0 0 0 0 0 0]

# 系数，类型<class 'numpy.ndarray'>
print(lr.coef_)         # [[2.8540218  -2.79274819]]
# 截距
print(lr.intercept_)    # [-6.83672757]

# 模型评估（准确率：预测正确的样本数占总预测样本数的比例）
print(lr.score(X_test, y_test))    # 1.0


# 绘制不规则决策边界（利用等高线原理）
def decision_boundary(model, axis):
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
    # plt.contour(X,Y,Z)：绘制等高线，X、Y是数据点的坐标，Z为每个坐标对应的高度值（数据点的分类）
    # 绘制等高线
    plt.contour(X0, X1, y_pred_matrix, linewidths=2, colors='#20B2AA')


# 逻辑回归二分类绘制
# decision_boundary(log_reg, axis=[4.5, 7.0, 2, 4.5])
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
# plt.show()

# 逻辑回归多分类（三分类）
# 使用鸢尾花前两个特征的全部分类（三分类）
X = X[:, :2]
y = y

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 逻辑回归分类器
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
# 训练模型
lr.fit(X_train, y_train)

# 在测试集上预测
y_pred = lr.predict(X_test)
print(y_pred)    # [2 2 0 2 0 2 0 2 2 2 2 2 2 2 2 0 2 2 0 0 2 2 0 0 2 0 0 2 1 0]
print(y_test)    # [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0]

# 模型评估（准确率：预测正确的样本数占总预测样本数的比例）
print(lr.score(X_test, y_test))    # 0.6
# 全部特征：0.9666666666666667

# 使用全部特征逻辑回归多分类预测鸢尾花类别
print(lr.predict([[5.2, 3.6, 4.0, 1.8]]))    # [2]

# LogisticRegression(multi_class='multinomial', solver='newton-cg')
# LogisticRegression(solver='liblinear', multi_class='ovr')
# Sklearn-Learn内置二分类、多分类API，分类器接收一个评估器estimator对象，其他使用同LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ovo = OneVsOneClassifier(LogisticRegression(multi_class='multinomial', solver='newton-cg'))
# ovr = OneVsRestClassifier(LogisticRegression(solver='liblinear', multi_class='ovr'))

# K近邻二分类
from sklearn.neighbors import KNeighborsClassifier

# 使用鸢尾花的前两个特征的前两个分类（二分类）
X = X[y < 2, :2]
y = y[y < 2]

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# KNN分类器
knn_clf = KNeighborsClassifier()
# 训练
knn_clf.fit(X_train, y_train)
# 绘制决策边界
decision_boundary(knn_clf, axis=[4.5, 7.0, 2, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
plt.show()

# K近邻多分类（三分类）

# 使用鸢尾花前两个特征的全部分类（三分类）
X = X[:, :2]
y = y

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# KNN分类器
knn_clf = KNeighborsClassifier()
# 训练
knn_clf.fit(X_train, y_train)
# 在测试集上预测
y_pred = knn_clf.predict(X_test)
print(y_pred)   # [1 1 0 2 0 2 0 2 1 2 2 2 2 2 2 0 2 1 0 0 1 1 0 0 2 0 0 2 1 0]
print(y_test)   # [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0]
# 测试与标签数据的平均准确度
print(knn_clf.score(X_test, y_test))    # 0.6666666666666666
# 全部特征：0.9666666666666667

# K近邻在鸢尾花数据集的前两个特征的三分类决策边界绘制
# 使用鸢尾花前两个特征的全部分类（三分类）
X = X[:, :2]
y = y

# KNN分类器
knn_clf = KNeighborsClassifier()
# 训练
knn_clf.fit(X, y)
# 绘制决策边界
decision_boundary(knn_clf, axis=[4.5, 7.0, 2, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='green')
plt.show()





