
# 数据处理基础库Numpy、Pandas
import numpy as np

# Numpy 科学数学库
"""
NumPy是Python中最常用的数值计算库之一，它提供了高效的多维数组对象以及对这些数组进行操作的函数
NumPy的核心是ndarray（N-dimensional array）对象，它是一个用于存储同类型数据的多维数组
Numpy通常与SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用，用于替代MatLab
SciPy是一个开源的Python算法库和数学工具包；Matplotlib是Python语言及其Numpy的可视化操作界面
"""
# None与NaN的区别：
'''
None：None表示空值，类型是NoneType
NaN（np.nan）：当使用numpy或pandas处理数据时，会将表中空缺项(为空)转换为NaN（类型是float）
'''
# 创建ndarray对象：使用np.array()函数可以创建一个ndarray对象，例如np.array([1, 2, 3])
# 数组属性：可以通过访问ndarray对象的属性，如shape、dtype和size，获取数组的形状、数据类型和元素个数

# 1、ndarray：n维数组对象：创建数组、数组属性
# A、创建ndarray对象方式：
# 1）array()函数：array()常用参数：
# dtype: 指定元素的数据类型；
arr1 = np.array([2, 3, 5])         # array()函数：创建ndarray对象
print(arr1)
arr2 = np.array([[1, 2], [3, 4]])  # 多维
print(arr2)
# B、numpy数组ndarray属性：
print(arr2.ndim)         # 维度数量，称为秩，即轴的数量
print(arr2.shape)        # 返回矩阵n行m列：(n,m)
print(arr2.size)         # 返回矩阵元素总数
print(arr2.dtype)        # 元素类型

# 2）asarray()：从已有的数组创建
ls = [3, 4, 5]
arr3 = np.asarray(ls)
print(arr3)

# 3）arange()函数：创建数值范围
# numpy.arange(start, stop, step, dtype)  [start,stop)
# start: 起始值，默认0；stop: 终止值；step: 步长；dtype: 类型
arr4 = np.arange(5)
print(arr4)

# 数组索引和切片：可以使用索引和切片操作访问数组的元素，如array[0]和array[1:3]

# 2、NumPy数组的切片和索引
print(arr4[1:3:1])       # 切片 [sta,end,step)
print(arr4[0])           # 索引 从0开始

arr5 = np.array([[1, 2], [3, 4], [5, 6]])
print(arr5[arr5 > 3])    # 布尔索引

# 数组运算：NumPy提供了许多用于数组运算的函数，如逐元素运算、矩阵运算、统计运算等

# 3、NumPy迭代数组: nditer()
for e in np.nditer(arr5):
    print(e)     # 每个元素迭代

# 4、NumPy数组操作
# 1）修改数组形状
# reshape(arr, shape, order='C') 不改变数据的条件下修改形状，order=C按行，F按列，A原顺序
a = np.arange(6)
print(a)                                # [0 1 2 3 4 5]
print(np.reshape(a, [2, 3]))            # 方式1
print(a.reshape(2, 3))                  # 方式2
'''
[[0 1 2]
 [3 4 5]]
'''

# 2）flat 数组元素迭代器
a = np.arange(4).reshape(2, 2)
for r in a:
    print(r)    # 按行迭代
for e in a.flat:
    print(e)    # 每个元素迭代

# 3）flatten() 扁平化、展开（不影响原数组）
print(a.flatten())   # [0 1 2 3]
# ravel() 扁平化、展开（影响原数组）
print(a.ravel())     # [0 1 2 3]

# 4）翻转数组
# np.transpose(arr)  对换数组的维度（等同于arr.T）
# np.swapaxes(arr, axis1, axis2)  交换数组的两个轴，axis1宽度方向 axis2深度方向
print(np.transpose(a))
'''
[[0 2]
 [1 3]]
'''
print(np.swapaxes(a, 0, 0))
'''
[[0 1]
 [2 3]]
'''

# 5、NumPy字符串函数
# 1）连接、拼接
print(np.char.add('abc', 'de'))                    # abcde
# 2）多重连接
print(np.char.multiply('abc', 2))                  # abcabc
# 3）将字符串居中，并使用指定字符在左侧和右侧进行填充
print(np.char.center('abcd', 10, fillchar='*'))    # ***abcd***
# 4）首字母转大写
print(np.char.capitalize('abc'))                   # Abc
# 5）每个单词的首字母转大写
print(np.char.title('it is abc'))                  # It Is Abc
# 6）全转大写
print(np.char.upper('abc'))                        # ABC
# 7）全转小写
print(np.char.lower('ABC'))                        # abc
# 8）分割、切分
print(np.char.split('www.baidu.com', sep='.'))     # ['www', 'baidu', 'com']
# 9）以\n、\r、\r\n作为分隔符切分
print(np.char.splitlines('line1\nline2'))          # ['line1', 'line2']
# 10）替换
print(np.char.replace('Alice', 'li', 'tt'))        # Attce

# 6、NumPy数学函数
print(np.ceil(5.2))                   # 向上取整（可传入）
print(np.floor(5.8))                  # 向下取整
print(np.around(5.542, decimals=1))   # 四舍五入，decimals=0（默认）保留位数
print(np.sin(30*np.pi/180))           # 正弦值（通过乘pi/180转化为弧度）
print(np.cos(30*np.pi/180))           # 余弦值
print(np.tan(30*np.pi/180))           # 正切值
print(np.arcsin(30))                  # 反正弦值（返回值以弧度为单位）
print(np.arccos(30))                  # 反余弦值
print(np.arctan(30))                  # 反正切值
print(np.degrees(np.arctan(30)))      # 转换为角度单位

# 7、NumPy算数运算函数
print(np.reciprocal(1 / 4))             # 倒数
print(np.power(2, 3))                   # x的y次方
print(np.sqrt(9))                       # 开方
print(np.mod(7, 3))                     # x%y

# 8、NumPy统计函数
# 数组的聚合操作：可以对数组进行聚合操作，如求和、求平均值、求最大值等
print(np.max([7, 9, 4, 2]))             # 最大值
print(np.min([7, 9, 4, 2]))             # 最小值
print(np.ptp([7, 9, 4, 2]))             # 最大值-最小值

ls = [80, 65, 70, 80, 95, 55, 90, 75]
print(np.sum(ls))                       # 总和
print(np.median(ls))                    # 中位数
print(np.mean(ls))                      # 算数平均数

# 加权平均数（将各值乘以相应的权数，然后求和得到总体值，再除以单位数）
# (1*4+2*3+3*2+4*1)/(4+3+2+1) = 2.0
print(np.average([1, 2, 3, 4], weights=[4, 3, 2, 1]))
print(np.std([1, 2, 3, 4]))             # 标准差
print(np.var([1, 2, 3, 4]))             # 方差

# 9、NumPy排序、筛选函数
# 1）排序（升序）
print(np.sort([5, 7, 2, 3]))            # [2 3 5 7]
print(list(np.sort([5, 7, 2, 3])))      # [2, 3, 5, 7]
# 2）分区排序
print(np.partition([3, 4, 2, 1], 3))         # [2 1 3 4] 比3小的排前面，比3大的排后面
print(np.partition([3, 4, 2, 1], (2, 3)))    # [1 2 3 4] 比2小的排前面，比3大的排后面，2和3之间的在中间
# 3）条件筛选
ls = [2, 3, 5, 7]
print(np.extract(np.mod(ls, 2) == 0, ls))    # [2] 选择满足条件的元素

