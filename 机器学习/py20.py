
# 机器学习模型评估与优化

# 1、机器学习模型的评估指标
"""
根据机器学习任务的不同，我们有多种模型评估指标
"""
'''
1）分类任务
对于二分类任务，常见的评估指标有准确率（accuracy）、精确度（precision）、召回率（recall）、F1 分数和 ROC AUC 分数等
对于多分类任务，可以采用多分类版本的上述指标，或者混淆矩阵等
2）回归任务
对于回归任务，常见的评估指标有均方误差（MSE）、均方根误差（RMSE）、平均绝对误差（MAE）等
3）聚类任务
对于聚类任务，常见的评估指标有轮廓系数、Davies-Bouldin 指数等
Scikit-learn 提供了丰富的函数来计算这些指标
'''

# 2、机器学习模型的交叉验证方法
'''
交叉验证是一种评估模型性能的方法。最常见的形式是k-折交叉验证，数据被分成k个子集，模型在k-1个子集上进行训练
在剩下的一个子集上进行测试。这个过程重复k次，每次选择一个不同的子集作为测试集，然后取k次的平均结果
'''
# Scikit-learn 的 cross_val_score 函数可以方便地实现交叉验证：
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(clf, X, y, cv=5)

# 3、机器学习模型的优化技巧
'''
模型优化主要包括特征选择、调参、模型选择等
1）特征选择：根据特征的重要性，选择对模型预测性能贡献大的特征。
2）调参：通过调整模型的超参数来改进模型的性能
3）模型选择：选择在验证集上表现最好的模型
Scikit-learn 提供了 GridSearchCV 和 RandomizedSearchCV 用于超参数调整，提供了 SelectFromModel 用于特征选择
'''

# 4、机器学习模型的持久化与部署
'''
训练好的模型可以通过序列化保存到磁盘，并在需要的时候加载回来
Scikit-learn 推荐使用 Python 的 pickle 模块或 joblib 模块来持久化模型
'''
# 模型部署则需要结合实际业务场景；可能的部署方式包括嵌入应用、REST API、实时流处理等

from sklearn.externals import joblib

# 保存模型
joblib.dump(clf, 'model.pkl')

# 加载模型
clf = joblib.load('model.pkl')

