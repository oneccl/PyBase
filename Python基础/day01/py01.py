
# Python：是一门面向对象、解释型脚本语言
# Python解释器：python.exe：将Python语言翻译成计算机CPU能识别的机器指令语言
# Python的编程方式分为2种：交互式(cmd命令框)、脚本式(.py文件)
"""
一、Python的诞生与发展
Python是由Guido van Rossum(龟叔)于1991年在荷兰创造的编程语言。最初，它被设计为一种可读性强、易于学习的语言。Guido van Rossum的目标是创造一种能够提高程序员生产力的语言。他命名它为Python，以纪念他喜爱的电视剧《Monty Python's Flying Circus》
Python从诞生之初就受到了广泛的关注和使用。它的简洁、可读性强以及丰富的生态系统使其成为了一个受欢迎的编程语言。Python的发展不断演进，引入了新的功能和语法，同时保持了向后兼容性，以确保现有的代码可以在新版本上运行
二、Python的特点
Python具有许多独特的特点，使其成为开发者钟爱的编程语言之一：
1）简洁而清晰的语法
Python采用简洁而清晰的语法，使得代码易于阅读和理解。这种语法风格被称为"Pythonic"，强调代码的可读性和简洁性
2）动态类型
Python是一种动态类型语言，它允许你在运行时更改变量的类型。这使得代码编写更加灵活，但也需要开发者更加注意类型相关的错误
3）面向对象
Python是一种面向对象的语言，它支持类、对象和继承等面向对象的概念。面向对象编程使得代码的组织和复用更加方便
4）丰富的标准库
Python附带了一个广泛而丰富的标准库，涵盖了各种功能，例如文件操作、网络通信、数据库访问等。这使得开发者能够快速地构建应用程序，而无需从头开始编写底层功能
三、Python的应用领域
Python在各个领域都有广泛的应用，包括但不限于：
1）Web开发：Python的简洁语法和丰富的Web框架（如Django和Flask）使其成为Web开发的热门选择
2）数据科学：Python在数据科学领域有着强大的支持，众多库（如NumPy、Pandas和Matplotlib）使得数据分析和可视化变得简单而高效
3）人工智能与机器学习：Python拥有流行的机器学习库（如Scikit-learn和TensorFlow），使其成为开发人员和研究人员进行人工智能和机器学习实验的首选语言
4）科学计算：Python具有强大的科学计算能力，因此被广泛用于科学研究、数值计算和工程计算等领域
四、Python 2.x与3.x的主要区别
Python有两个主要版本：2.x和3.x。这两个版本之间有一些重要的区别：
语法差异：Python 3.x引入了一些新的语法特性，并修复了一些在Python 2.x中存在的语法问题。例如，Python 3.x中的print语句被替换为print函数，并且引入了新的除法运算符
字符串处理：Python 2.x中的字符串处理有些限制，而Python 3.x中的字符串处理更加统一和一致。在Python 3.x中，字符串默认使用Unicode编码
# Python 2.x中的print语句
print "Hello, World!"
# Python 3.x中的print函数
print("Hello, World!")
"""

# 1、注释（单行注释：#；多行注释'''或"""回车）
"""
Python多行注释
"""

# 2、关键字、保留字

# 3、标识符：字母、数字、下划线组成；不能以数字开头；大小写敏感
# 3.1、变量与常量
# 变量：不需要申明类型，变量名 = 值；命名格式：小驼峰(userName)、小写下划线(user_name)
# 常量：语法跟变量完全一致，全大写
userName = "Tom"
user_name = "Tom"
# 多变量赋值
a = b = c = 10
print(a, b, c)
d, e, f = 2, 3, 5
print(d, e, f)
# 变量的3大属性：
print(id(a), type(a), a)    # id内存地址 type类型 值
# 3.2、垃圾回收机制(GC)
# Python解释器自带的GC主要运用"引用计数(reference counting)"来跟踪和回收垃圾（不可用变量）
# 在"引用计数"的基础上，还可以通过"标记-清除(mark and sweep)"解决容器对象可能产生的循环引用的
# 问题，并且通过"分代回收(generation collection)"以空间换时间的方式来进一步提高垃圾回收的效率
# 3.3、获取输入函数：input("提示信息")
# num = input("请输入一个整数：")
# 输出函数：print(var, end='可选，若多个变量以什么分隔')
# print(num)      # 换行输出
a = 1
b = 2
print(a,)         # 不换行输出
print(a, b)

# 4、行与缩进
"""
Python最具特色的就是使用严格的缩进来表示代码块，不需要使用{}
一条语句多行输出：使用 +\ 实现
一行多条语句输出：语句间使用 ; 分割
"""
# 代码块/组：缩进相同的一组语句，首行以关键字开始，以:结束

# 5、Python数据类型（6大数据类型）
"""
基本数据类型(不可变)：数字（Number）、字符串（String）
复合数据类型(可变)：列表、元组、字典、集合
"""
# 5.1、Number数字（4种）
# 1）int长整型：Python3无long类型
x = 10
print(x)
# 2）bool布尔：只有True和False两个取值；True对应整数1，False对应整数0；布尔运算：and、or、not三种
''''
等同于False的值：False、None、0、0.0、""、()、[]、{}
'''
b = True
print(b)
# 3）float浮点数：双精度
pi = 3.14
print(pi)
# 4）complex复数：a+bj：a实部，b虚部
z = 3 + 4j
print(z)

# 5.2、字符串：使用''或""包裹，是字节数组
s = "Alice"
print(s[0])       # 访问字符，索引从0开始，可为负（从-1开始）
print(s[1:3])     # [sta,end)截取/切片
print(s[2:])      # [sta,后面全部]
print(s * 2)      # 连接、重复、复制
print('A' in s)   # in、not in：检查是否存在sub_str或字符
print(len(s))     # 字符串长度
s[2] = 'l'        # 字符串是不可变变量，不支持直接通过下标修改
# 字符串拼接
print("Hello！" + s)
print("Hello！", s)
print(f"Hello！{s}")
# 连接字符串优先使用join()
print('.'.join(['www', 'baidu', 'com']))    # '连接符'.join([e1,...])：字符串拼接：www.baidu.com
# 字符串格式化
print("%s 今年 %d 岁！" % (s, 18))    # %s字符串 %d长整型 %f浮点型 %.nf浮点数(保留n位)
print('{}*{}={}'.format(2, 3, 2*3))  # str.format()方法格式化
print(f"He name is {s}")             # s-string：可嵌入{变量/表达式}
print(r"C:\Users\cc\Desktop\a.txt")  # r-string：原始字符串，所有字符串都是按照字面意思使用(不需要转义)
print(u"unicode")                    # u-string：表示unicode编码的字符串，防止中文乱码问题(Python3中所有字符串默认都是unicode字符串)
print(b"10")                         # 表示字符串是bytes二进制类型

# %ns 字符串前面填充n个空格
print("%10s" % ('A'))
# %-ns 字符串后面填充n个空格
print("%-10s" % ('A'))

# 5舍6入
print("%.0f" % (10.6))      # 11
print("%.0f" % (10.5))      # 10
# 字符串格式化添加%：%%
print("%.0f%%" % (10.5))    # 10%

# 在Python3中，bytes和str的转换方式为：str.encode('utf-8')、bytes.decode('utf-8')
# 常用字符串内置函数：
s1 = 'ods_user_info'
print(s1.strip())               # 删除所有空格
print(s1.upper())               # 全转大写
print(s1.lower())               # 全转小写
print(s1.replace('ods_', ""))   # user_info，替换
print(s1.split("_"))            # ['ods', 'user', 'info']，分割
print(s1.capitalize())          # Ods_user_info，仅首字母大写
print(s1.startswith('ods'))     # True，是否以什么开头
print(s1.endswith('ods'))       # False，是否以什么结尾
print(s1.find('A'))             # -1 find(sub): 找子串：没找到返回-1
print('  AB C '.strip())        # 字符串去除前后空格
print('Hello'.index('o'))       # 返回字符或子串的索引（不存在报错）
print('hello world'.title())    # 每个单词首字母大写
print('hello'.ljust(10, '*'))   # hello***** 左对齐，右侧填充至指定长度
print('hello'.rjust(10, '*'))   # *****hello 右对齐，同上
print('hello'.center(15, '*'))  # *****hello***** 居中对齐，同上

# 5.3、List列表
'''
定义：存储多个数据，且可为不同数据类型，元素间用,分割，使用[]包裹
特点：
有序，有索引，可更新，元素可重复
'''
ls = [10, 2.78, 'Tom', True, ["a", "b"]]    # 定义
print(ls[0])       # 访问元素，索引从0开始，可为负（从-1开始）
print(ls[1:3])     # [sta,end)截取/切片
print(ls[2:])      # [sta,后面全部]，包括sta
print(ls[:2])      # [前面全部,end)，不包括end
print(ls * 2)      # 连接、重复、复制
print(ls + ls)     # 连接、组合、拼接
print(len(ls))     # 长度、大小
print(10 in ls)    # 元素是否存在
# 添加元素
ls.append("New")
print(ls)
# 修改元素
ls[1] = 12
# 删除元素
ls.remove(["a", "b"])
print(ls)
del ls[4]
print(ls)
# print(list.pop())   # 移出最后一个元素，并将这个元素返回
# ls.clear()          # 清空列表
ls.count(10)        # 返回元素出现的次数
ls.index(10)        # 返回元素的索引
ls.reverse()        # 反转
# ls.sort()           # 排序：默认从小到大；参数reverse=True时，从大到小
# sort()与sorted()区别
'''
sort()：仅列表的排序方法
sorted()：可迭代序列排序，不改变原对象本身
'''
ls.copy()           # 浅拷贝，相当于ls[:]
import copy
new_ls = copy.deepcopy(ls)    # 深拷贝
# 遍历
for i in ls:
    print(i)
for index, e in enumerate(ls):
    print(f"{index} : {e}")
# 列表排序
ls1 = [4, 3, 5]
print(ls1.sort())    # 默认升序，参数reverse=True可降序
sorted(ls1)          # 内置函数排序，默认升序，参数reverse=True可降序，返回新列表，不对原列表做任何修改
# 列表反转
print(ls.reverse())

# 1）列表作为栈使用：先进后出
stack = [3, 4, 5]
stack.append(6)    # 进栈
print(stack)
stack.pop()        # 出栈
print(stack)
# 2）列表作为队列使用：先进先出
from collections import deque
queue = deque([2, 3, 5])
queue.append(8)    # 入列
print(queue)
queue.popleft()    # 出列
print(queue)

# 5.4、元组：值不允许修改、删除
'''
定义：存储多个数据，且可为不同数据类型，元素间用,分割，使用()包裹
特点：
有序，有索引，不可更新，元素可重复
'''
t1 = (10, 3.14, "Jack", True)
t2 = 10, 3.14, "Jack", True    # 任意无符号的对象，以逗号隔开，默认为元组
t3 = ()             # 空元组
t4 = (20,)          # 一个元素的元组需要在元素后添加逗号
print(t1[0])        # 访问元素，索引从0开始，可为负（从-1开始）
print(t1[1:3])      # [sta,end)截取/切片
print(t1[2:])       # [sta,后面全部]
print(t1 * 2)       # 连接、重复、复制
print(t1 + t4)      # 连接、组合、拼接
print(len(t1))      # 长度、大小
t1.count(10)        # 返回元素出现的次数
t1.index(10)        # 返回元素的索引
print(10 in ls)     # 元素是否存在
# 修改：元组不允许修改元素
# 删除：元组不允许删除元素
# 遍历
for j in t1:
    print(j)
for k, v in (['k', 'v'],):
    print(f"{k} = {v}")   # k = v

# 5.5、字典：d = {key1 : value1, key2 : value2, ...} key唯一（后面相同替换前面）、类型不变；value任意
'''
定义：存放一组有对应关系的数据
特点：
1）无序，可更新，有索引，Key不能重复
2）使用{}包裹，元素由键值对K:V组成，元素间使用,分割
'''
d = {'name': 'Tom', 'age': 18}
print(d['name'])      # 根据Key访问元素
print(d.get('name'))  # get(key, 0)：获取key对应的value，若为空用0填充
d['age'] = 19         # 修改元素（根据Key）
d['addr'] = "美国"     # 添加元素
del d['age']          # 删除元素
print(d)
# d.clear()             # 清空字典
print(len(d))         # 元素数量（Key的数量）
print(d.keys())       # 获取所有Key
print(d.values())     # 获取所有Value
print(d.items())      # 获取所有K和V，K-V以元组形式显示

# 字典元素合并
d1 = {'A': 10}
d2 = {'B': 11}
# 方式1
res = {}
res.update(d1)
res.update(d2)
print(res)
# 方式2
res = {**d1, **d2}
print(res)
'''
{'A': 10, 'B': 11}
'''

# 字典排序
d = {'u': 4, 'w': 2, 'v': 1}
# 默认按键K升序
print(sorted(d.items()))
# [('u', 4), ('v', 1), ('w', 2)]
# 根据值降序
print(sorted(d.items(), key=lambda item: item[1], reverse=True))
# [('u', 4), ('w', 2), ('v', 1)]

# 遍历
for k, v in d.items():
    print(f"{k} : {v}")
for tup in d.items():
    print(tup)   # 以元组(key, value)形式显示

# 5.6、集合（set）：元素不可重复
'''
定义：多个元素的无序组合，每个元素唯一，集合元素不可修改（为不可变数据类型）
特点：
1）无序，无索引，元素不可重复
2）集合一旦被创建，无法更新内容，但可以添加
3）可求交集、并集、差集等
4）使⽤{}或set()，若要创建空集合只能使⽤set()，因为默认{}是⽤来创建空字典的
'''
s1 = {10, 3.14, 'Tom', True}
s1.add("New")       # 添加元素
s1.remove("Tom")    # 删除元素
# 修改：集合不允许修改元素
print(len(s1))      # 集合大小、元素数量
print(10 in s1)     # 元素是否存在
s1.clear()          # 清空集合
set1 = {1, 3, 5, 7, 8}
set2 = {2, 3, 4, 6, 8}
# set集合的方法：交集，注意：返回set类型
print(set1.intersection(set2))
print(set1 & set2)
# set集合的方法：并集，注意：返回set类型
print(set1.union(set2))
print(set1 | set2)
ls1 = [1, 2, 2, 3]
ls2 = [2, 3, 4]
# 利用集合实现列表去重
print(set(ls1))         # {1, 2, 3} list去重，注意：返回set类型，需要再转为list:
print(list(set(ls1)))
# set集合的方法：差集：set_a.difference(set_b): 属于a_set不属于b_set的元素
print(set(ls1).difference(set(ls2)))     # {1} 注意：返回set类型
# 遍历
for e in s1:
    print(e)
for index, e in enumerate(s1):
    print(f"{index} : {e}")

# 6、类型转换
print("x的类型为：", type(x))     # 查看类型

print(str(x))            # 强转为string
print(int(pi))           # 强转为int
print(float(x))          # 强转为float
print(bool(s))           # 字符串转bool，若不为空True，若为空False
print(list(s))           # 字符串转list
print(set(ls))           # 列表转集合，去重
print(list(set(ls)))     # 集合转列表

name = ['A', 'B', 'C']
age = [20, 19, 21]
# 二元组列表转字典
d = dict(zip(name, age))
print(d)                  # {'A': 20, 'B': 19, 'C': 21}
# 字典转二元组列表
print(list(d.items()))    # [('A', 20), ('B', 19), ('C', 21)]
# 元组转列表
t = ('k', 'v')
print(list(t))
# 列表转元组
print(tuple(name))

# str()与repr()区别
"""
str()：面向用户，目的是可读性，返回友好可读性高的字符串
repr()：面向开发人员，目的是准确性，其返回值表示Python解释器内部的定义，可以使用eval()还原对象
"""

# 7、运算符
'''
1）算数运算符：+、-、*、/、%
2）比较运算符：>、>=、<、<=、==、!=（<>）
3）逻辑运算符：and(与)、or(或)、not(非)
4）成员运算符：in、not in
5）位运算符：&(按位与)、|(按位或)、~(按位取反)
6）is与==：is比较的是内存地址；==比较的是值
'''
print(not 2 > 1)
print('1' in '123')
m = 10         # 1010
n = 3          # 0011
print(m & n)   # 0010 输出2
p = "abc"
q = "abc"
print(p is q)  # True
print(p == q)  # True

# 8、Python随机数
# 方式1：random包
import random
print(random.randint(1, 10))            # [1, 10]随机整数
# 方式2：numpy包
import numpy as np
print(np.random.randint(1, 10))         # [1, 10]随机整数
print(np.random.uniform(1, 10))         # [1, 10]均匀分布随机数
print(np.random.normal(5, 1, [2, 2]))   # 正态分布随机数，均值5标准差1，2行2列

# 9、对象比较与拷贝
ls = [1, 2]

# 1）比较：==比较对象的值(内容)，is比较对象的id
# Python对象三要素：id(身份标识)、type(数据类型)、value(值)

# 2）拷贝：
import copy

# A、浅拷贝：创建新对象，仅复制原对象的第一层属性成员，若原对象的属性是基本类型，则复制值；若原对象的属性是引用类型，则复制引用
# 浅拷贝后两个对象的第一层属性成员互不影响，但第二层及以下的属性成员会共享同一个内存地址，会互相影响
ls_copy = copy.copy(ls)
print(ls_copy)               # [1, 2]
print(ls is ls_copy)         # False
ls_copy.append(3)
print(ls)                    # [1, 2]

# B、深拷贝：创建新对象，递归地复制原对象的所有层级的属性成员，完全独立，互不影响
ls_deepcopy = copy.deepcopy(ls)
print(ls_deepcopy)           # [1, 2]
print(ls is ls_deepcopy)     # False
ls_deepcopy.append(3)
print(ls)                    # [1, 2]

# 10、Python断言
# assert语句：用于表达式判断，若为True则程序运行；否则，程序停止运行，抛出AssertionError错误
# 语法：
'''
assert 表达式, 错误提示信息
'''
# 类似if语句：
'''
if not 表达式:
    raise AssertionError
'''
# 基本使用
def game(age):
    assert age >= 18, "未满18禁止进入游戏！"
    print("享受欢乐游戏时光！")

game(17)

