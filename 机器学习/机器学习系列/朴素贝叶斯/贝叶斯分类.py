
# Scikit-Learn朴素贝叶斯

# 贝叶斯分类法是基于贝叶斯定理的统计学分类方法。它通过预测一组给定样本属于一个特定类的概率来进行分类。贝叶斯分类在机器学习知识结构中的位置如下：

# 贝叶斯分类
# 贝叶斯分类的历史可以追溯到18世纪，当时英国统计学家托马斯·贝叶斯发展了贝叶斯定理，这个定理为统计决策提供了理论基础
# 不过，贝叶斯分类得到广泛实际应用是在20世纪80年代，当时计算机技术的进步使得大规模数据处理成为可能

# 在众多机器学习分类算法中，贝叶斯分类和其他绝大多数分类算法都不同
# 例如，KNN、逻辑回归、决策树等模型都是判别方法，也就是直接学习出输出Y和特征X之间的关系，即决策函数$Y$=$f(X)$或决策函数$Y$=$P(Y|X)$
# 但是，贝叶斯是生成方法，它直接找出输出Y和特征X的联合分布$P(X,Y)$，进而通过$P(Y|X)$=$\frac{P(X,Y)}{P(X)}$计算得出结果判定

# 贝叶斯分类是一类分类算法的总称，这类算法均以贝叶斯定理为基础，故统称为贝叶斯分类。而朴素贝叶斯（Naive Bayes）分类是贝叶斯分类中最简单，也是常见的一种分类方法
# 朴素贝叶斯算法的核心思想是通过特征考察标签概率来预测分类，即对于给定的待分类样本，求解在此样本出现的条件下各个类别出现的概率，哪个最大，就认为此待分类样本属于哪个类别
# 例如，基于属性和概率原则挑选西瓜，根据经验，敲击声清脆说明西瓜还不够成熟，敲击声沉闷说明西瓜成熟度好，更甜更好吃
# 所以，坏瓜的敲击声是清脆的概率更大，好瓜的敲击声是沉闷的概率更大。当然这并不绝对——我们千挑万选的沉闷瓜也可能并没熟，这就是噪声了。当然，在实际生活中，除了敲击声，我们还有其他可能特征来帮助判断，例如色泽、根蒂、品类等
# 朴素贝叶斯把类似敲击声这样的特征概率化，构成一个西瓜的品质向量以及对应的好瓜/坏瓜标签，训练出一个标准的基于统计概率的好坏瓜模型，这些模型都是各个特征概率构成的
# 这样，在面对未知品质的西瓜时，我们迅速获取了特征，分别输入好瓜模型和坏瓜模型，得到两个概率值。如果坏瓜模型输出的概率值更大一些，那这个瓜很有可能就是个坏瓜

# 贝叶斯定理
# 先验概率与后验概率
# 贝叶斯定理（Bayes Theorem）也称贝叶斯公式，其中很重要的概念是先验概率、后验概率和条件概率
# 1）先验概率
# 事件发生前的预判概率。可以是基于历史数据的统计，可以由背景常识得出，也可以是人的主观观点给出。一般都是单独事件概率
# 例如，如果我们对西瓜的色泽、根蒂和纹理等特征一无所知，按照常理来说，好瓜的敲声是沉闷的概率更大，假设是60%，那么这个概率就被称为先验概率
# 2）后验概率
# 事件发生后的条件概率。后验概率是基于先验概率求得的反向条件概率。概率形式与条件概率相同
# 例如，我们了解到判断西瓜是否好瓜的一个指标是纹理。一般来说，纹理清晰的西瓜是好瓜的概率更大，假设是75%，如果把纹理清晰当作一种结果，然后去推测好瓜的概率，那么这个概率就被称为后验概率
# 3）条件概率
# 一个事件发生后另一个事件发生的概率。一般的形式为P(B|A)，表示事件A已经发生的条件下，事件B发生的概率
# $$P(B|A)=\frac{P(AB)}{P(A)}$$
# 4）贝叶斯公式
# 贝叶斯公式是基于假设的先验概率与给定假设下观察到不同样本数据的概率提供了一种计算后验概率的方法。朴素贝叶斯模型依托于贝叶斯公式
# $$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$
# 贝叶斯公式中
# P(A)是事件A的先验概率，一般都是人主观给定的。贝叶斯中的先验概率一般特指它
# P(B)是事件B的先验概率，与类别标记无关，也称标准化常量，通常使用全概率公式计算得到
# P(B|A)是条件概率，又称似然概率，一般通过历史数据统计得到
# P(A|B)是后验概率，后验概率是我们求解的目标

# 由于P(B)与类别标记无关，因此估计P(A|B)的问题最后就被我们转化为基于训练数据集样本先验概率P(A)和条件概率P(B|A)的估计问题
# 贝叶斯公式揭示了事件A在事件B发生条件下的概率与事件B在事件A发生条件下的概率的关系

# 贝叶斯定理的推导
# 根据条件概率公式可得
# $$P(AB)=P(B|A)P(A)$$
# 同理可得
# $$P(BA)=P(A|B)P(B)$$
# 设事件A与事件B互相独立，即$P(AB)$=$P(BA)$，则有
# $$P(B|A)P(A)=P(A|B)P(B)$$
# 由此可得
# $$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$

# 朴素贝叶斯
# 基于贝叶斯定理的贝叶斯模型是一类简单常用的分类算法。在假设待分类项的各个属性相互独立的前提下，构造出来的分类算法就称为朴素的，即朴素贝叶斯算法
# 所谓朴素，就是假定所有输入事件之间相互独立。进行这个假设是因为独立事件间的概率计算更简单，当然，也更符合我们的实际生产生活
# 朴素贝叶斯模型的基本思想是，对于给定的待分类项$X{x_1,x_2,...,x_n}$，求解在此项出现的条件下各个类别$P(y_i|X)$出现的概率，哪个最大，就认为此待分类项属于哪个类别
# 朴素贝叶斯算法的定义是，设$X{x_1,x_2,...,x_n}$为一个待分类项，每个$x_i$为X的一个特征属性，且特征属性之间相互独立。设$C{y_1,y_2,...,y_i}$为一个类别集合，分别计算
# $$P(y_1|X),P(y_2|X),...,P(y_i|X)$$
# 取其中条件概率最大的对应类别作为分类结果
# $$P(y_k|X)=max{P(y_1|X),P(y_2|X),...,P(y_i|X)} \;\;\; X \in y_k$$
# 求解后验概率$P(y_k|X)$的关键在于求解其中的各个条件概率，其求解核心为：对于已知待分类项集合（训练集），统计得到在各类别下各个特征属性的条件概率估计
# 在朴素贝叶斯中，待分类项的每个特征属性都是独立的，根据贝叶斯公式
# $$P(y_i|X)=\frac{P(X|y_i)P(y_i)}{P(X)}$$
# 因为分母P(X)是已存在事件X的概率，所以对于任何待分类项来说P(X)都是固定常数。因此，在求后验概率$P(y_i|X)$的时候，我们只需要考虑分子即可
# 又因为各个特征属性是相互独立的，因此，对于分子就有
# $$\begin{aligned}
# P(X|y_i)P(y_i)&=P(x_1|y_i)P(x_2|y_i) \cdots P(x_n|y_i)P(y_i) \\ &= P(y_i)\prod_{j=1}^{n}P(x_j|y_i)
# \end{aligned}$$
# 由上式可得
# $$P(X|y_i)=\prod_{k=1}^{n}P(x_k|y_i)$$
# 对于先验概率$P(y_i)$，可以近似理解为在训练集D中$y_i$出现的概率：
# $$P(y_i)=\frac{|y_i|}{D}$$
# 对于条件概率$P(x_i|y_i)$，可以近似理解为在类别$y_i$中，特征元素$x_i$出现的概率：
# $$P(x_i|y_i)=\frac{|训练样本为y_i时x_i出现的次数|}{|y_i训练样本数|}$$
# 总结来说，朴素贝叶斯模型的分类过程为：获取训练样本，确定特征属性，对每个类别计算$P(y_i)$，对每个特征属性计算所有划分的条件概率$P(x_i|y_i)$，
# 对每个类别计算$P(x_i|y_i)P(y_i)$，以$P(x_i|y_i)P(y_i)$中最大项作为X的所属类别

# 深入理解贝叶斯分类
# 以下通过根据气象情况预测出行案例帮助我们理解贝叶斯分类的原理过程
#
# 已知某人的出行和出行时的气象记录如下：
# |天气 |温度 |湿度 | 刮风| 出行|
# |--|--|--|--|--|
# |雨 |热 |高 | 有| 是|
# |晴 |凉 |低 | 有| 是|
# |雨 |中 |低 | 无| 否|
# |雨 |凉 |高 | 有| 否|
# |晴 |热 |中 | 无| 是|
# |晴 |热 |高 | 有| 否|
# 由上述表格可知，数据的特征共有4个：天气、温度、湿度和刮风，类别共有2个：是否出行
# 下面我们来预测一下，在气象是雨、热、高、有的情况下，这个人是否会出行？

# 这是一个典型的分类问题。转化为数学问题就是：比较`p(是|雨,热,高,有)`与`p(否|雨,热,高,有)`的概率，通过判断两个概率大小得出是否出行
# 根据朴素贝叶斯公式，可得
# $$p(是|雨,热,高,有)=\frac{p(雨,热,高,有|是)*p(是)}{p(雨,热,高,有)}$$
# 其中，`p(雨,热,高,有|是)`表示已知出行发生的条件下气象为`雨,热,高,有`的条件概率，`p(是)`表示出行的先验概率，`p(雨,热,高,有)`表示气象为`雨,热,高,有`的先验概率
# 通过朴素贝叶斯公式，我们可以将无法直接求解的因转换为求解已知的三个量的果，将待求的量转化为其它可求的量，这就是贝叶斯公式所做的事情

# 由于朴素贝叶斯假设各个特征之间相互独立，因此有
# $$p(雨,热,高,有|是)=p(雨|是)*p(热|是)*p(高|是)*p(有|是)$$
# $$p(雨,热,高,有)=p(雨)*p(热)*p(高)*p(有)$$
# 根据上式，我们只需要分别计算出等式右边的概率，也就得到了左边的概率
# 当样本量很大时，根据中心极限定理，样本的抽样分布服从正态分布，频率近似于概率，所以，这里我们直接进行统计即可
# 下面我们按照分子、分母求解三个量
# 分子
# $$p(是)=3/6 \\
# p(雨|是)=1/3 \\
# p(热|是)=2/3 \\
# p(高|是)=1/3 \\
# p(有|是)=2/3$$
# 分母
# & p(雨)=3/6
# & p(热)=3/6
# & p(高)=3/6
# & p(有)=4/6
# 分子 4/81 * 1/2 = 2/81
# 分母 1/12
# p(是|雨,热,高,有)=24/81=8/27

# 同理，计算
# 分子
# & p(否)=3/6 \notag \\
# & p(雨|否)=2/3 \notag \\
# & p(热|否)=1/3 \notag \\
# & p(高|否)=2/3 \notag \\
# & p(有|否)=2/3 \notag
# 分母
# & p(雨)=3/6 \notag \\
# & p(热)=3/6 \notag \\
# & p(高)=3/6 \notag \\
# & p(有)=4/6 \notag
# 分子 8/81 * 1/2 = 4/81
# 分母 1/12
# p(否|雨,热,高,有)=48/81=16/27

# p(是|雨,热,高,有)<p(否|雨,热,高,有)
# 最大后验概率为p(否|雨,热,高,有)，因此，这个人大概率不会出行


# 文档分词与词汇权重（TF-IDF）

# 文档分词
# 文本分类主要做的是如何提取文本中的主要信息。那么，如何衡量哪些信息是主要信息呢？
# 我们知道，一篇文档是由若干词汇组成的，也就是文档的主要信息是词汇。从这个角度来看，我们就可以用一些关键词来描述文档
# 这种处理文本的方式叫做词袋(bag of words)模型，该模型会忽略文本中的词汇出现的顺序以及相应的语法，将文档看做是由若干单词组成的，且单词之间相互独立，没有关联

# 要想提取文档中的关键词，就需要对文档进行分词。分词的方法一般是：基于统计和机器学习，需要人工标注词性和统计特征，训练分词模型
# 值得注意的是，分词后词汇中必定存在一些普遍使用的停用词，这些停用词对文档分析作用不大，因此，在文档分析之前需要将这些词去掉
# 另外，分词阶段还需要处理同义词，很多时候一个事物有多个不同的名称，例如，番茄和西红柿等
# 中文分词与英文分词是不同的，它们需要不同的分词库进行处理：
# - 中文分词：jieba
# - 英文分词：NLTK

# 词汇权重（TF-IDF）
# 有了分词，那么，哪些关键词对文档才是重要的呢？
# 例如，可以通过词汇出现的次数，次数越多就表示越重要。更为合理的方法是计算词汇的TF-IDF值
# TF-IDF
# TF-IDF（Term Frequency-Inverse Document Frequency，词频-逆文档频率）是一种统计方法，用来评估一个词汇对一篇文档的重要程度
# 词汇的重要性随着它在文档中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降
# - TF（Term Frequency）：即词频，指某个特定的词汇在该文档中出现的次数
#   $$TF_\omega=词汇\omega出现的次数/文档中的总词汇数$$
# - IDF（Inverse Document Frequency）：即逆向文档频率，指一个词汇在文档中的类别区分度。它认为一个词汇出现在文档的数量越少，这个词汇对该文档就越重要，就越能通过这个词汇把该文档和其他文档区分开
#   $$IDF_\omega=\log \frac{语料库中的文档总数}{包含词汇\omega的文档数+1}$$
#   IDF是一个相对权重值，$\log$的底数可以自定义。其中分母加1是为了避免分母为0（有些单词可能不在文档中出现）
# 某一特定文档内的高频词汇，以及该词汇在整个文档集合中的低频率文档，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词汇，保留重要的词汇
# $$TF-IDF = TF×IDF$$
# 以下是一个示例：
# 假设一篇文章中共有2000个单词，“中国”出现100次。假设全网共有1亿篇文章，其中包含“中国”的有200万篇。求单词“中国”的TF-IDF值
# $$
# & TF(中国) = 100 / 2000 = 0.05 \notag \\
# & IDF(中国) = \log_{10} \frac{1亿}{200万+1} = 1.7 \notag \\
# & TF-IDF(中国) = 0.05 × 1.7 = 0.085
# $$
# 通过计算文档中词汇的TF-IDF值，我们就可以提取文档中的特征属性。即就是将TF-IDF值较高的词汇，作为文档的特征属性


# TfidfVectorizer文本特征提取
# Sklearn提供了计算TF-IDF值的API：TfidfVectorizer
# 官方对该API的描述如下：
# TfidfVectorizer将原始文档集合转换为TF-IDF特征矩阵。相当于CountVectorizer后接TfidfTransformer。CountVectorizer将文本文档集合转换为词频/字符频数矩阵；TfidfTransformer将词频/字符频数矩阵转换为标准化的TF或TF-IDF矩阵
# TfidfVectorizer的一般使用步骤如下：
# # train_matrix = vectorizer.fit_transform(X_train)
# # test_matrix = vectorizer.transform(X_test)
# TfidfVectorizer将原始文本转化为TF-IDF特征矩阵，从而为后续的文本相似度计算奠定基础。
# 以下是一个示例：
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# # 定义3个文档
# docs = [
#     'I am a student.',
#     'I live in Beijing.',
#     'I love China.',
# ]
# # 计算TF-IDF值
# # TF-IDF矢量化器
# vectorizer = TfidfVectorizer()
# # 拟合模型
# tfidf_matrix = vectorizer.fit_transform(docs)
# # 获取所有不重复的特征词汇
# print(vectorizer.get_feature_names_out())  # ['am' 'beijing' 'china' 'in' 'live' 'love' 'student']
# # 不知道你有没有发现，这些特征词汇中不包含`i`和`a`，你能解释这是为什么吗？
# # 获取特征词汇的TF-IDF矩阵值
# print(tfidf_matrix.todense())
# print(tfidf_matrix.toarray())
# # 获取特征词汇与列的对应关系
# print(vectorizer.vocabulary_)  # {'am': 0, 'student': 6, 'live': 4, 'in': 3, 'beijing': 1, 'love': 5, 'china': 2}
#
# # 与英文文档不同，中文文档的词汇之间没有像英文那样的自然空格分割，因此，需要额外处理，要将中文文档转换为类似英文文档中自然空格分割的格式
# # 以下是一个示例：
#
# import jieba
#
# # 定义3个文档
# docs = [
#     "我是一名学生。",
#     "我居住在北京。",
#     "我爱中国。"
# ]
# # 中文文档预处理
# # 使用中文分词库jieba进行分词
# doc_words = [jieba.lcut(doc) for doc in docs]
# new_docs = [' '.join(words) for words in doc_words]
# print(new_docs)  # ['我 是 一名 学生 。', '我 居住 在 北京 。', '我 爱 中国 。']
# # 计算TF-IDF值
# # TF-IDF矢量化器
# vectorizer = TfidfVectorizer()
# # 拟合模型
# tfidf_matrix = vectorizer.fit_transform(new_docs)
# # 获取所有不重复的特征词汇
# print(vectorizer.get_feature_names_out())  # ['一名' '中国' '北京' '学生' '居住']
# # 同样，这些特征词汇中不包含“我”、“是”、“在”和“爱”，你能解释这是为什么吗？
# # 获取特征词汇的TF-IDF矩阵值
# print(tfidf_matrix.todense())
# print(tfidf_matrix.toarray())
# # 获取特征词汇与列的对应关系
# print(vectorizer.vocabulary_)  # {'一名': 0, '学生': 3, '居住': 4, '北京': 2, '中国': 1}

# 案例：新闻分类与预测
# 停用词文件
p = r'stopwords\baidu_stopwords.txt'
# 训练集数据（包括4个文件：财经类、娱乐类、健康类、体育类）
train_dir = 'train_data'
# 测试集数据（包括4个文件：财经类、娱乐类、健康类、体育类）
test_dir = 'test_data'

# 文件加载与文档分词
# 加载停用词文件
def load_stopwords(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    words = [line.strip() for line in lines]
    return words

stop_words = load_stopwords(p)
# print(stop_words)
# print(len(stop_words))
# print(type(stop_words))

import jieba

# 加载数据文件、文档分词及分词处理
def load_docs(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    titles = []
    labels = []
    for line in lines:
        line_arr = line.strip().split('---')
        if len(line_arr) != 2:
            continue
        label, title = line_arr
        # 分词
        words = jieba.lcut(title)
        words = [word.strip() for word in words if word.strip() != '']
        # 分词处理：中文分词类比英文添加空格分割
        title = ' '.join(words)
        titles.append(title)
        labels.append(label.strip())
    return titles, labels

import os

# 加载新闻数据集
def load_files(_dir):
    # 获取指定目录下的全部
    ls = os.listdir(_dir)
    titles_ls = []
    labels_ls = []
    # 数据整合
    for file in ls:
        path = os.path.join(_dir, file)
        titles, labels = load_docs(path)
        titles_ls += titles
        labels_ls += labels
    return titles_ls, labels_ls


# 训练集
train_titles, train_labels = load_files(train_dir)
# 测试集
test_titles, test_labels = load_files(test_dir)

# 计算单词权重（TF-IDF）
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF矢量化器
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
# 在训练集上拟合并转换（计算训练集单词权重（TF-IDF））
train_titles_tfidf = tf.fit_transform(train_titles)
# 在测试集上转换（计算测试集单词权重（TF-IDF））
test_titles_tfidf = tf.transform(test_titles)

# 训练朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB

# 多项式朴素贝叶斯分类器（使用默认参数）
clf = MultinomialNB()
# 拟合模型
clf.fit(train_titles_tfidf, train_labels)

# 模型评估
# 准确度评分
print(clf.score(test_titles_tfidf, test_labels))  # 0.961082910321489

# 预测新闻类型
# 在测试集上预测
labels_pred = clf.predict(test_titles_tfidf)

# # 根据标题预测新闻类型
# def predict(title):
#     words = jieba.lcut(title)
#     words = [word.strip() for word in words if word.strip() != '']
#     text = ' '.join(words)
#     text_tfidf = tf.transform([text])
#     label_pred = clf.predict(text_tfidf)
#     return label_pred[0]
#
#
# print(predict('东莞市场采购贸易联网信息平台参加部委首批联合验收'))  # 财经
# print(predict('红薯的好处 常吃这种食物能够帮你减肥'))  # 健康

# Sklearn模型保存与加载
# 在实际应用中，训练一个模型需要花费很多时间。为了方便使用，可以将训练好的模型导出到磁盘上，在使用的时候，直接加载使用即可
# 模型导出的过程称为对象序列化，而加载还原的过程称为反序列化
# Sklearn模型的保存与加载主要有两种方式：

import joblib

# 保存模型：将模型clf保存到文件clf.pkl
joblib.dump(clf, 'clf.pkl')

# 加载模型
model = joblib.load('clf.pkl')


import pickle

# 保存模型：将模型clf保存到文件clf.pkl
with open('clf.pkl', 'wb') as file:
    pickle.dump(clf, file)

# 加载模型
with open('clf.pkl', 'rb') as file:
    model = pickle.load(file)


# # 根据标题预测新闻类型
# def predict(title):
#     words = jieba.lcut(title)
#     words = [word.strip() for word in words if word.strip() != '']
#     text = ' '.join(words)
#     text_tfidf = tf.transform([text])
#     label_pred = model.predict(text_tfidf)
#     return label_pred[0]
#
#
# print(predict('东莞市场采购贸易联网信息平台参加部委首批联合验收'))  # 财经




