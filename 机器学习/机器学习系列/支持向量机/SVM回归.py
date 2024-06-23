
# https://www.cnblogs.com/wang_yb/p/17938132
# https://www.showmeai.tech/article-detail/196
# https://www.zybuluo.com/Duanxx/note/433281
# https://tangshusen.me/2018/10/27/SVM/

# Scikit-Learn支持向量机回归

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.svm import SVR                             # 支持向量机回归
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.metrics import mean_squared_error as MSE   # 均方误差


# 创建sin函数样本数据
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
# 在标签中增加噪音
y[::5] += 3 * (0.5 - np.random.rand(8))
# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(X, y)
plt.show()

# 拟合SVR回归模型
svr_lin = SVR(kernel='linear')
svr_lin.fit(X, y)
svr_lin_pred = svr_lin.predict(X)
svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X, y)
svr_rbf_pred = svr_rbf.predict(X)
# svr_poly = SVR(kernel='poly')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=0.1, coef0=1)
svr_poly.fit(X, y)
svr_poly_pred = svr_poly.predict(X)

# 绘制拟合曲线
svrs = [(svr_lin, svr_lin_pred), (svr_rbf, svr_rbf_pred), (svr_poly, svr_poly_pred)]
labels = ['Linear', 'RBF', 'Polynomial']
colors = ['m', 'c', 'g']

# 创建多图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 10), sharey=True)
for idx, (svr, svr_pred) in enumerate(svrs):
    # 拟合图
    axes[idx].plot(X, svr_pred, color=colors[idx], lw=2, label=f'{labels[idx]} model')
    # 散点图
    # 支持向量机散点图
    axes[idx].scatter(X[svr.support_], y[svr.support_], facecolor="none", edgecolor=colors[idx], s=50, label=f'{labels[idx]} support vectors')
    # 非支持向量机散点图
    axes[idx].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)], y[np.setdiff1d(np.arange(len(X)), svr.support_)], facecolor="none", edgecolor="k", s=50, label='Other support vectors')
    # 图例
    axes[idx].legend(loc='upper right')

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()

# 加州房价预测
from sklearn.datasets import fetch_california_housing as fch

# 加载数据集
X = pd.DataFrame(fch().data)
y = fch().target

# 划分训练集（70%）与测试集（30%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# 支持向量机回归器
svr = SVR(kernel='rbf')
# 训练模型
svr.fit(X_train, y_train)
# 在测试集上预测
y_pred = svr.predict(X_test)
# 支持向量的索引
print(svr.support_)    # [0 1 2 ... 14445 14446 14447]
# # 优化拟合模型所运行的迭代次数
# print(svr.n_iter_)     # 2131373911
# # 系数（仅在线性内核的情况下可用）
# print(svr.coef_)
'''
[[ 4.03749816e-01  1.31519316e-02 -6.00080161e-02  4.63479872e-01
  -2.93040212e-05  6.89952551e-02 -4.20757026e-01 -4.37328088e-01]]
'''
# 截距
print(svr.intercept_)  # [-37.59767043]

# 模型评估
# 拟合程度、决定系数R^2
print(svr.score(X_test, y_test))           # -0.3895967189192997
# 均方误差
print(MSE(y_test, y_pred))                 # 1.8526614779520805
# 平均绝对误差
print(MSE(y_test, y_pred, squared=False))  # 1.3611250779969049

# 根据模型预测
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



