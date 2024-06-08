

# 模型评估指标

# A、回归问题模型评估指标

# 回归与分类算法的模型评估其实是相似的法则：找真实标签和预测值的差异
# 在分类算法中，这个差异只有一种角度来评判，那就是是否预测到了正确的分类，其结果只存在0（分类错误）或100（分类正确），并不存在中间过程的衡量标准
# 而在回归类算法中，我们一般从两种不同的角度来看待回归的效果：第一，我们是否预测到了正确的数值；第二，我们是否拟合到了足够的信息
# 这两种角度分别对应着不同的模型评估指标
# 1）均方误差（MSE）
# 均方误差（MSE）可用于验证是否预测到了正确的数值
# 它的本质是我们的预测值与真实值之间的差异，也就是从第一种角度来评估我们回归的效果。所以RSS既是我们的损失函数，也是我们回归模型的重要评估指标之一
# 但是，RSS有一个致命的缺点：它是一个无界的和，可以无限地大。我们只知道，我们想要求解最小的RSS，从RSS的公式来看，它不能为负，所以RSS越接近0越好
# 但我们没有一个概念，究竟多小才算好？多接近0才算好？为了应对这种状况，Scikit-Learn使用了RSS的变体：均方误差MSE（Mean Squared Error，L2损失）来衡量我们的预测值和真实值的差异
#
# 均方误差即预测值和真实值之差平方的期望值，本质是在RSS的基础上除以了样本总量，得到了每个样本量上的平均误差。有了平均误差，我们就可以将每个样本的平均误差和我们的标签的取值范围在一起比较，以此获得一个较为可靠的评估依据
# 在Scikit-Learn当中，我们有两种方式调用这个评估指标：一种是使用Scikit-Learn专用的模型评估模块里的类mean_squared_error，另一种是调用交叉验证的类 cross_val_scoresco并使用里面的参数来设置使用均方误差
# Scikit-Learn均方误差
from sklearn.metrics import mean_squared_error as MSE

# mse = MSE(y_pred, y_test)

# MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据精确度越高
# 2）均方根误差（RMSE）
# 一些教材也使用均方根误差（RMSE）代替MSE。均方根误差是均方误差的算术平方根，可使用参数squared=False指定
# 3）平均绝对误差（MAE）
# 在线性回归中，MSE（L2损失）计算简便，由于它的惩罚是平方和，所以含有异常值的损失（Loss）会非常大，因此，MSE对异常值敏感。当预测值和真实值接近时，MSE较小；反之非常大。使用MSE会导致异常点有更大的权重，因此数据有异常点时，使用平均绝对误差（MAE）损失函数更好
# 平均绝对误差MAE（Mean Absolute Error，L1损失）是另一种用于回归模型的损失函数
#
# 平均绝对误差表示预测值和观测值之间绝对误差的平均值。平均绝对误差认为每个样本的差异在平均值上的权重都相等
# 4）决定系数（R^2）
# 决定系数（R^2）可用于验证是否拟合了足够的信息
# 在统计学中，决定系数反映了因变量y的波动有多少百分比能被自变量x（特征）的波动所描述。简单来说，该参数可以用来判断统计模型对数据的拟合能力（说服力）
# 在Scikit-learn中，回归模型的性能分数就是利用R^2对拟合效果打分的，具体的实现函数是score()


# 交叉验证
# 交叉验证是一种模型选择的方法。交叉验证可以分为以下三种：
#   简单交叉验证：即将数据按照一定比例分为训练集和测试集
#   S折交叉验证：将已给数据切分为S个互不相交、大小相同的子集，将S-1个子集的数据作为训练集来训练模型，剩余的一个测试模型，重复S次，选择S次中平均测试误差最小的模型
#   留一交叉验证：即S=n。往往在数据缺乏的时候使用，因为数据很少没法再分了
# 由于交叉验证是用来模型选择的，所以是将不同的模型（如SVM，LR等）运用上述方法，然后比较误差大小，选择误差最小的模型
# 需要注意的是，上述三种方法是针对数据量不充足的时候采用的交叉验证方法，如果数据量充足（如100万），一种简单的方法就是将数据分为3部分：
#   训练集：用来训练模型
#   验证集：用于模型选择
#   测试集：用于最终对学习方法的评估
# 选择验证集上有最小预测误差的模型
# 测试集的目的是评估模型在训练集上的效果。核心也是看模型在测试集上的表现，并根据测试集结果来评估模型的优劣
# 交叉验证主要用于评估模型的预测性能，尤其是训练好的模型在新数据上的表现，可以在一定程度上减小过拟合。交叉验证可以完整的利用数据信息，而不是只将数据简单的分为训练集和测试集
# 1）简单交叉验证
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# 通过训练集来训练模型，然后通过测试集表现来评估该模型
# 2）S折交叉验证
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

clf = LogisticRegression()
lr = LinearRegression()
# score1 = cross_val_score(clf, X, y, cv=10, scoring='r2')
# score2 = cross_val_score(lr, X, y, cv=10, scoring='r2')
# print(f'逻辑回归 10折交叉验证平均决定系数R^2: {score1.mean()}')
# print(f'线性回归 10折交叉验证平均决定系数R^2: {score2.mean()}')
# 例如10折交叉验证，首先，将全部样本划分成10个大小相等的样本子集；然后依次遍历这10个子集，每次把当前子集作为验证集（测试集），其余9份作为训练集，进行模型的训练和评估；最后把10次评估指标的平均值作为最终的评估指标
# 3）留一交叉验证
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
# for train, test in loo.split(X, y):
#     print(train.shape, test.shape)
# 留一交叉验证每一次把n-1的数据集作为训练集，1作为验证集。即训练集样本量为n-1，测试集样本量为1
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
# https://scikit-learn.org.cn/view/663.html

# S折交叉验证设置参数使用均方误差：
from sklearn.model_selection import cross_val_score

# score = cross_val_score(reg, X, y, cv=10, scoring="neg_mean_squared_error")
# print(score.mean())

# 这里需要注意的是，虽然均方差永远为正，但是在cross_val_score的scoring参数下，均方误差作为评判标准时，确是计算负均方误差neg_mean_squared_error。这是因为Scikit-Learn在计算模型评估指标时，会考虑指标本身的性质
#
# 均方误差本身是一种误差，所以被Scikit-Learn划分为一种损失（Loss）。在Scikit-Learn当中，所有损失都使用负数表示。因此，这里均方误差也显示为负数。真正的均方误差MSE的数值其实就是neg_mean_squared_error去掉负号的数字


# B、单项分类模型评估指标与综合分类模型评估指标

# 科学家门捷列夫说“没有测量，就没有科学”，在AI场景下我们同样需要定量的数值化指标来指导我们更好地应用模型对数据进行学习和建模
# 事实上，在机器学习领域，对模型的测量和评估至关重要。选择与问题相匹配的评估方法，能帮助我们快速准确地发现在模型选择和训练过程中出现的问题，进而对模型进行优化和迭代

# 模型评估的目标

# 模型评估的目标是选出泛化能力强的模型完成机器学习任务。实际的机器学习任务往往需要进行大量的实验，经过反复调参、使用多种模型算法（甚至多模型融合策略）来完成自己的机器学习问题，并观察哪种模型算法在什么样的参数下能够最好地完成任务
# 泛化能力强的模型能很好地适用于未知的样本，模型的错误率低、精度高。机器学习任务中，我们希望最终能得到准确预测未知标签的样本（即泛化能力强）的模型
# 但是我们无法提前获取未知的样本，因此我们会基于已有的数据进行切分来完成模型训练和评估，借助于切分出的测试数据进行评估，可以很好地判定模型状态（过拟合/欠拟合），进而迭代优化
# 在建模过程中，为了获得泛化能力强的模型，我们需要一整套方法及评价指标：
# + 评估方法：为保证客观地评估模型，对数据集进行的有效划分实验方法
# + 性能指标：量化的评估模型效果的指标

# 模型评估方法

# 模型评估方法主要涉及到对完整数据集不同的有效划分方法，保证我们后续计算得到的评估指标是可靠有效的，进而进行模型选择和优化
# 1）留出法（Hold-out）
# 留出法是机器学习中最常见的评估方法之一，它会从训练数据中保留出验证样本集，这部分数据不用于训练，而用于模型评估
# 使用留出法划分数据集需要注意：
# + 划分时一般不宜随机划分，可以采用分层抽样的方式选择测试数据，以保证数据分布比例的平衡
# + 单次划分不一定能得到合适的测试集，一般多次重复按照划分、训练、测试求误差的步骤，取误差的平均值
# + 划分的验证集不能太大或太小，否则评估将失去意义，常用做法是选择1/5~1/3左右的数据当作验证集用于评估
# 2）交叉验证法（Cross Validation）
# 留出法的数据划分可能会带来偏差。在机器学习中，另外一种比较常见的评估方法是交叉验证法：K折交叉验证对K个不同分组训练的结果进行平均来减少方差
# 因此模型的性能对数据的划分就不那么敏感，对数据的使用也会更充分，模型评估结果更加稳定，可以很好地避免上述问题
# 使用交叉验证法划分数据集需要注意：
# + 当数据量较小时，选择较大的K；当数据量较大时，选择较小的K。这样可以提高评估的效率
# + K一般取5或10，当K取样本总数m时，称为留一法，每次的测试集都只有一个样本，需要进行m次训练和预测
# 3）自助采样法（Bootstrap Sampling）
# 部分场景下，数据量较少，很难通过已有的数据来估计数据的整体分布。因为数据量不足时，计算的统计量反映不了数据分布，这时可以使用自助采样法
# 自助采样法是一种用小样本估计总体值的一种非参数方法，在进化和生态学研究中应用十分广泛。自助采样法通过有放回抽样生成大量的伪样本，通过对伪样本进行计算，获得统计量的分布，从而估计数据的整体分布
# 自助采样法的过程为
# + 对m个样本进行m次有放回采样得到初始训练集，初始数据集中有的样本多次出现，有的则从未出现，从未未出现的剩余样本作为测试集
# + 上述的采样过程我们可以重复T次，采样出T个包含m个样本的训练集，然后基于每个训练集训练出一个基学习器，然后将这些基学习器进行结合

# 有了有效的模型评估方法，我们还需要量化的度量指标来精准评估与判断模型性能。以下是单项分类模型常用的评估指标：

# 1、单项分类模型评估指标
# 单项分类问题评估指标主要有：准确率、精确率和召回率等，而这些指标都是基于混淆矩阵进行计算的
# 1）混淆矩阵（Confusion Matrix）
# 混淆矩阵（Confusion Matrix）可以直观地展示模型预测结果与实际标签之间的对应关系。它是一个表格矩阵，习惯上，通常矩阵的行表示实际的类别标签，矩阵的列表示模型预测的类别标签
# 对于二分类模型，如果把预测情况与实际情况的所有结果两两组合，结果将会呈现4种情况：
# 图
# 例如，现在有10张人类的照片，其中男性5张，女性5张，假设我们已经训练了一个根据图像识别性别的模型，当我们使用它来预测性别时，会出现以下4种情况：
# + 实际为男性，且预测为男性（正确）
# + 实际为男性，但预测为女性（错误）
# + 实际为女性，且预测为女性（正确）
# + 实际为女性，但预测为男性（错误）
# 这4种情况构成了经典的混淆矩阵，如下图：
# 图
# + P（Positive Sample）：正例样本的数量
# + N（Negative Sample）：负例样本的数量
# + TP（True Positive）：正确预测到的正例的数量
# + FP（False Positive）：把负例预测成正例的数量
# + FN（False Negative）：把正例预测成负例的数量
# + TN（True Negative）：正确预测到的负例的数量
# 这种表示方法可分解为两部分看：预测是否正确（T/F），预测的类(P/N)
# 根据混淆矩阵可以得到评价单项分类模型的指标有：准确率、精确率/查准率、召回率/查全率

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://scikit-learn.org.cn/view/485.html
# 2）准确率（Accuracy）
# 准确率（Accuracy）是指预测正确的正负例样本数占样本总数的比例，是最常用的指标，可以总体衡量一个预测的性能。一般情况（数据类别均衡）下，模型的准确率越高，说明模型的效果越好
# 图
# 虽然准确率可以衡量总体的正确率，但是在数据类别不均衡的情况下，这个评估指标并不合理
# 例如，发病率为0.1%的医疗场景下，如果只追求准确率，模型可以把所有人判定为没有病的正常人，准确率高达99.9%，但这个模型实际是不可用的。为了更好地应对上述问题，衍生出了一系列其他评估指标：精确率和召回率
# 3）精确率/查准率（Precision）
# 精确率（Precision）又称查准率，表示在模型预测为正例的样本中，真正为正例的样本所占的比例。一般情况下，精确率越高，说明模型的效果越好
# 图
# 精确率与准确率类似但完全不同。精确率表示对正例样本结果中的预测准确程度，而准确率则表示对整体样本的预测准确程度，既包括正例样本，也包括负例样本
# 精确率偏向的思路是：宁愿漏掉，不可错杀。例如，在垃圾邮件识别的场景中，因为不希望很多的正常邮件被误杀，这样会造成严重的困扰。因此，精确率将是一个被侧重关心的指标
# 4）召回率/查全率（Recall）
# 召回率（Recall）又称查全率，表示模型正确预测为正例的样本数量占总的正例样本数量的比例。一般情况下，召回率越高，说明有更多的正例样本被模型预测正确，模型的效果越好
# 图
# 召回率偏向的思路是：宁愿错杀，不可漏掉。例如，在金融风控领域中，因为希望系统能够筛选出所有存在风险的行为或用户，然后交给人工鉴别，漏掉一个可能造成灾难性后果。因此，召回率将是一个被侧重关心的指标

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# https://scikit-learn.org.cn/view/482.html
# # 以下案例使用混淆矩阵评估鸢尾花数据集上SVM分类器的输出质量：
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

# 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 使用过于规范化（C太低）的模型运行分类器，以查看对结果的影响
clf = svm.SVC(kernel='linear', C=0.01, probability=True)
clf.fit(X_train, y_train)
# 用训练好的模型进行预测
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# normalize：是否归一化，默认None(不归一化)，字符串true表示归一化
cm = confusion_matrix(y_test, y_pred, normalize=None)
# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot(cmap=plt.cm.Blues)
print(disp.confusion_matrix)
'''
[[13  0  0]
 [ 0 10  6]
 [ 0  0  9]]
'''
plt.title("Unnormalized Confusion Matrix")
plt.show()
# 在混淆矩阵中，横轴是预测值，纵轴是真实值，对角线元素表示预测标签等于真实标签的点数，而非对角线元素则是分类器未正确标记的点。混淆矩阵的对角线值越高越好，表明正确的预测越多
# 在类别不均衡的情况下，使用归一化可能带来有趣的结果，可以对哪个类被错误分类具有更直观的解释

# 模型评估报告
# Sklearn模型分类评估报告提供了模型在各个类别上的详细性能指标。包括精确率、召回率、F1分数等评估指标，这些指标能够帮助我们更全面地了解模型的性能
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# 生成模型评估报告，返回str类型
report = classification_report(y_test, y_pred)
print(report)

# 2、综合分类模型评估指标
# 分类模型的评估与回归模型的侧重点不同，回归模型针对连续型的数据，而分类模型针对的是离散的数据
# 因此，分类模型的评估指标也与回归模型不同，回归模型的评估指标包括均方误差（MSE）、均方根误差（RMSE）和平均绝对误差（MAE）等，分类模型的评估指标通常包括准确率、精确率、召回率和F1分数等
# 不过，这些指标衡量的都是预测值与真实值之间的数值差异
# 在上篇中，我们已经介绍了单项分类模型评估指标：准确率、精确率和召回率，
# 本文主要介绍综合分类问题评估指标：F1分数、ROC曲线和AUC曲线等，而这些指标都是基于单项分类模型评估指标的
# 1）F1-Score与Fβ-Score
# 如果我们把精确率（Precision）和召回率（Recall）之间的关系用图来表达，就是下面的PR曲线：
# 图
# 可以发现两者是“两难全”的关系。理论上来说，精确率和召回率都是越高越好，但更多时候它们两个是矛盾的，经常无法保证二者都很高
# 为了综合两者的表现，在两者之间找到一个平衡点，就引入了一个新指标Fβ-Score
# 图
# 可以根据不同的业务场景来调整β值。当β为1时，Fβ-Score就是F1分数（F1-Score），此时，综合平等地考虑了精确率和召回率评估指标，当F1分数较高时则说明模型性能较好
# 图
# 当β＜1时，更关注精确率；当β＞1时，更关注召回率
# 2）灵敏度与特异度
# ROC和AUC是两个更加复杂的评估指标。它们都基于两个指标：灵敏度（Sensitivity）和特异度（Specificity）
# 灵敏度也称真正例率（True Positive Rate，TPR），特异度也称假正例率（False Positive Rate，FPR）
# 灵敏度（TPR）
# 图
# 1-特异度（FPR）
# 图
# 可以看到，灵敏度和召回率是一模一样的，只是换了个名称而已。另外，需要注意的是，由于我们只关心正例样本，所以需要查看有多少负例样本被错误地预测为正例样本，因此以上所说的特异度特指1-特异度，而不是真正的特异度
# 如上图所示，TPR和FPR分别是基于实际表现1和0出发的，也就是说它们分别在实际的正例样本和负例样本中来观察相关概率问题。正因为如此，所以无论样本是否平衡，都不会被影响
# 例如，总样本中，90%是正例样本，10%是负例样本。我们知道用准确率是有水分的，但是用TPR和FPR不一样。这里，TPR只关注90%正例样本中有多少是被负例覆盖的，而与那10%毫无关系，同理，FPR只关注10%负例样本中有多少是被正例覆盖的，也与那90%毫无关系
# 所以，如果我们从实际表现的各个结果角度出发，就可以避免样本不平衡的问题了，这也是为什么选用TPR和FPR作为ROC/AUC的指标的原因
# 图
# 另外，我们也可以从另一个角度理解：条件概率。假设X为预测值，Y为真实值。那么就可以将这些指标按如下条件概率表示
# <center>准确率=P(Y=1|X=1)&ensp;召回率=灵敏度=P(X=1|Y=1)&ensp;1-特异度=P(X=0|Y=0)</center>
# 从上面三个公式可以看到：如果我们以实际结果为条件（召回率，特异度），那么就只需考虑一种样本；而如果以预测值为条件（准确率），那么我们需要同时考虑正例样本和负例样本
# 所以以实际结果为条件的指标都不受样本不平衡的影响，相反以预测结果为条件的就会受到影响
# 3）ROC曲线
# ROC（Receiver Operating Characteristic）曲线，又称接受者操作特征曲线。该曲线最早应用于雷达信号检测领域，用于区分信号与噪声。后来人们将其用于评价模型的预测能力，ROC曲线是基于TPR和FPR得出的
# 图
# + (0，0)：即TPR=FPR=0，意味着TP=FP=0，分类器把每个正例都预测为负例
# + (0，1)：即TPR=1，FPR=0，意味着TP=0且FP=0，分类器将所有样本都预测正确
# + (1，1)：即TPR=FPR=0，意味着TP=FP=0，分类器把每个正例都预测为负例
# + (1，0)：即TPR=0，FPR=1，意味着TP=1且FP=1，分类器将所有样本都预测错误
# FPR表示模型预测的虚报程度，而TPR表示模型预测的覆盖程度。我们所希望的是：虚报的越少越好，覆盖的越多越好。也就是TPR越高，同时FPR越低（即ROC曲线越陡峭），那么模型的性能就越好
# 此时，说明模型在保证能够尽可能地准确识别小众样本的基础上，还保持一个较低的误判率，即不会因为要找出小众样本而将很多大众样本给误判
# 一般来说，如果ROC曲线是光滑的，那么基本上可以判断模型没有太大的过拟合
# 图
# 4）AUC曲线
# ROC曲线的确能在一定程度上反映模型的性能，但它并不是那么方便，因为曲线越陡峭这个说法还比较抽象，不够定量化，因此还是需要一个定量化的标量指标来反映这个事情。ROC曲线的AUC值恰好就做到了这一点
# 图
# AUC（Area Under ROC Curve）是ROC曲线下面积，AUC值越大，就能够保证ROC曲线越陡峭，ROC曲线越靠近左上方
# 比较有意思的是，如果我们连接对角线，它的面积正好是0.5。对角线的实际含义是：正负样本随机覆盖率应该都是50%。ROC曲线越陡越好，所以理想值就是1，一个正方形；而最差的随机预测也有0.5，所以一般AUC的值是介于0.5到1之间的
# AUC指标的一般判断标准如下：
# + 0.5–0.7：效果较差
# + 0.7–0.85：效果一般
# + 0.85–0.95：效果良好
# + 0.95–1：效果优秀（但一般不太可能）
# AUC曲线的物理意义是正例样本的预测结果大于负例样本的预测结果的概率，本质是AUC反应的是分类器对样本的排序能力
# 以下面的样本为例，AUC表示模型将某个随机正类样本排列在某个随机负类样本之前的概率。模型的预测从左到右以升序排列：
# 图
# 3、分类模型评估指标总结
# 下面对常用的分类模型评估指标进行如下总结：
# + 准确率（Accuracy）：适用于正负样本数量相差不大的情况
# + 精确率/查准率（Precision）：注重准，适用于正负样本差异很大的情况，不能用于抽样情况下的效果评估
# + 召回率/查全率（Recall）：注重全，适用于正负样本差异很大的情况，不受抽样影响
# + F1分数：描述了精准率和召回率的关系，在准与全之间平衡
# + ROC：对正负样本不平衡的数据集不敏感
# + AUC：计算与排序有关，因此对排序敏感，对预测分数不敏感

# Sklearn模型评估指标
# Sklearn提供了包括但不限于以上各种模型评估指标API，详情参考官方文档：https://scikit-learn.org.cn/view/93.html
# 例如，下面是根据预测分数计算接收器工作特性曲线（ROC）下的面积（AUC）的API：
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
# https://scikit-learn.org.cn/view/500.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# https://scikit-learn.org.cn/view/501.html
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score

# 准确率
print(accuracy_score(y_test, y_pred))  # 0.8421052631578947
# 精确率
print(precision_score(y_test, y_pred, average="micro"))  # 0.8421052631578947
# 召回率
print(recall_score(y_test, y_pred, average="micro"))  # 0.8421052631578947
# F1分数
print(f1_score(y_test, y_pred, average="micro"))  # 0.8421052631578947

# 根据模型预测置信度
y_score = clf.predict_proba(X_test)
print(y_score.shape)    # (38, 3)

# AUC（方式1）
# 汇总所有类的贡献，计算所有类的平均AUC指标
micro_auc_ovr = roc_auc_score(y_test, y_score, multi_class="ovr", average="micro")
print(micro_auc_ovr)    # 0.9889196675900277

from sklearn.preprocessing import label_binarize    # 独热编码，仅支持一维输入

# AUC（方式2）
# 以OvR的方式通过对目标进行二值化，一个给定类被视为正类，其余类都被视为负类
y_true = label_binarize(y_test, classes=clf.classes_)
print(y_true.shape)     # (38, 3)
# 根据置信度和样本真实标签计算FPR/TPR/临界点
# ravel()：将多维数组扁平为一维数组
fpr, tpr, thresholds = roc_curve(y_true.ravel(), y_score.ravel())
print(auc(fpr, tpr))    # 0.9889196675900277
# roc_curve不支持多分类，只能计算特定类别的ROC
# fpr, tpr, thresholds = roc_curve(y_true[:, 2], y_score[:, 2])
# print(auc(fpr, tpr))    # 0.9808429118773947

# 基于FPR（X轴）和TPR（Y轴）绘制ROC曲线
from matplotlib import pyplot as plt

plt.plot(fpr, tpr)
plt.title("ROC-AUC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

# 基于ROC曲线可视化API绘制
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
# https://scikit-learn.org.cn/view/585.html
from sklearn.metrics import RocCurveDisplay

# plot_chance_level=True：绘制对角线
display = RocCurveDisplay.from_predictions(
    y_true.ravel(),
    y_score.ravel(),
    plot_chance_level=True
)
plt.title("ROC-AUC")
plt.show()




