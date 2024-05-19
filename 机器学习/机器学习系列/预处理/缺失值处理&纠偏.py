

# 数据预处理：缺失值处理

# # 填充缺失值
#
# from sklearn.impute import SimpleImputer
#
# # 1）单变量填充
# # 均值(mean)、中位数(median)、众数(most_frequent)和常量(constant)填充
# # 下面以均值填充为例
# data = np.array([[1, 2, 3], [4, np.NaN, 6], [7, 8, np.NaN]])
# print(data)
# '''
# [[ 1.  2.  3.]
#  [ 4. nan  6.]
#  [ 7.  8. nan]]
# '''
# # SimpleImputer(missing_values,strategy,fill_value)
# # missing_values：指定缺失值  strategy：填充策略  fill_value：当填充策略为常量(constant)时使用
# imputer = SimpleImputer(missing_values=np.NaN, strategy="mean")
# data = imputer.fit_transform(data)
# print(data)
# '''
# [[1.  2.  3. ]
#  [4.  5.  6. ]
#  [7.  8.  4.5]]
# '''
#
# # 2）多变量填充
#
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
#
# data = np.array([[1, 2, 3], [4, np.NaN, 6], [7, 8, np.NaN]])
# print(data)
# '''
# [[ 1.  2.  3.]
#  [ 4. nan  6.]
#  [ 7.  8. nan]]
# '''
# imp = IterativeImputer(max_iter=10, random_state=0)
# data1 = imp.fit_transform(data)
# print(data1)
# '''
# [[1.         2.         3.        ]
#  [4.         5.00203075 6.        ]
#  [7.         8.         8.99796726]]
# '''
#
# # 这种方式非常灵活，在拟合的时候可以选择多种模型，默认为BayesianRidge()循环插补估计器
# # 下面以决策树回归模型为例
# from sklearn.tree import DecisionTreeRegressor
#
# imp = IterativeImputer(DecisionTreeRegressor(), max_iter=10, random_state=0)
# data2 = imp.fit_transform(data)
# print(data2)
# '''
# [[1. 2. 3.]
#  [4. 8. 6.]
#  [7. 8. 6.]]
# '''
#
#
# # 3）K近邻（KNN）填充
#
# from sklearn.impute import KNNImputer
#
# data = np.array([[1, 2, 3], [4, np.NaN, 6], [7, 8, np.NaN], [10, 11, 12]])
# print(data)
# '''
# [[ 1.  2.  3.]
#  [ 4. nan  6.]
#  [ 7.  8. nan]
#  [10. 11. 12.]]
# '''
# # uniform：统一的权重。每个邻域内的所有点的权重是相等的
# imputer = KNNImputer(n_neighbors=2, weights="uniform")
# data = imputer.fit_transform(data)
# print(data)
# '''
# [[ 1.  2.  3.]
#  [ 4.  5.  6.]
#  [ 7.  8.  9.]
#  [10. 11. 12.]]
# '''


# # 数据预处理：纠偏
# from sklearn.preprocessing import PowerTransformer
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
#
# # 使用管道构建线性回归模型工作流
# pipe = Pipeline([
#     ('ss', StandardScaler()),
#     ('pt', PowerTransformer(method='yeo-johnson')),
#     ('lr', LinearRegression())
# ])


