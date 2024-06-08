

# Sklearn模型保存与加载

# 在实际应用中，训练一个模型需要花费很多时间。为了方便使用，可以将训练好的模型导出到磁盘上，在使用的时候，直接加载使用即可
# 模型导出的过程称为对象序列化，而加载还原的过程称为反序列。Sklearn模型的保存与加载主要有两种方式：
# 1、Pickle
# Pickle（二进制协议）模块主要用于Python对象的序列化和反序列化，Pickle是二进制序列化格式
import pickle

# 保存模型：将模型clf保存到文件clf.pkl
with open('clf.pkl', 'wb') as file:
    pickle.dump(clf, file)

# 加载模型
with open('clf.pkl', 'rb') as file:
    model = pickle.load(file)

# 2、Joblib
# Joblib是Sklearn自带的一个工具。通常情况下，Joblib的性能要优于Pickle，尤其是当数据量较大的时候
# sklearn.externals.joblib函数存在于Sklearn0.21及以前的版本中，在最新的版本中，该函数已被弃用改为直接导入Joblib
import joblib

# 保存模型：将模型clf保存到文件clf.pkl
joblib.dump(clf, 'clf.pkl')

# 加载模型
model = joblib.load('clf.pkl')

# 可以看到，与Pickle相比，Joblib提供了更简单的工作流程。Pickle要求将文件对象作为参数传递，而Joblib可以同时处理文件对象和字符串文件名



