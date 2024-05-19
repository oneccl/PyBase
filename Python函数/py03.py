# 函数：可以提高应用的模块性、代码的复用性

# 1、函数定义、返回值
"""
def 函数名(参数列表):
    '''函数文档说明'''
    function_suite函数体
    return [expression]（可以使用return返回值，没有return默认返回None）
"""
def add(x, y):
    return x + y

# 2、函数调用
print(add(2, 3))

# 3、函数的参数传递：
# 3.1、必须/位置参数：参数必须数量和顺序正确，如上面add()函数

# 3.2、关键字参数：使用关键字参数允许函数调用时参数的顺序与声明时不一致
# 3.3、默认参数：若没有传递参数，则会使用默认参数
def greet(name, msg='Hi！'):
    print(msg, name)

print(greet('Tom'))
print(greet(msg='Hello！', name='Jerry'))

# 3.4、可变参数：
'''
Python函数参数*args和**kwargs：用于在Python中编写变长参数的函数
*args收集额外的参数组成元组；**kwargs收集额外的参数组成字典
实际起作用的语法是*和**；args和kwargs只是约定俗成的名称
'''
# *args  1个*：该参数会以元组形式输入，若没有指定参数，则输入为一个空元组
def multiply(*args):
    res = 1
    for n in args:
        res *= n
    return res

print(multiply(2, 3, 4))  # 2*3*4=24

# **args  2个*：该参数会以字典形式输入
def method(**args):
    print(args)

method(a=2, b=3)  # {'a': 2, 'b': 3}

# 4、局部变量与全局变量（命名空间与作用域）
# 命名空间：命名空间是名称(标识符)的集合，用于确保名称唯一以避免命名冲突，已定义的每个名称都有与之对应的命名空间，不同的命名空间完全隔离
# 局部变量：在函数内部定义的变量，函数运行时会开辟临时的局部命名空间，作用范围：仅限于函数内部
# 全局变量：在函数外部定义的变量，代码运行时会创建存储变量名与值关系的空间，作用范围：整个程序
count = 0

def increment():
    # global：在函数内部使用global声明的全局变量，可以在函数内部修改；局部想对全局变量的值进行修改时使用
    global count
    count += 1
    print(count)

increment()  # 1
increment()  # 2

# nonlocal：当需要修改函数外层函数包含的名称对应的值时，需要使用nonlocal关键字
def func1():
    p = 1
    def func2():
        nonlocal p
        p = 0
    func2()
    print(f"func1内的p值为{p}")

func1()      # func1内的p值为0

# 5、lambda表达式：匿名函数（函数式编程特性）
'''
Python匿名函数：lambda表达式；Python使用lambda表达式创建匿名函数，匿名函数是一种不使用def关键字定义的函数
'''
# 定义：lambda 形参列表: 表达式(lambda体)
add = lambda x, y: x + y

print(add(2, 3))  # 5

# 6、Python内置高阶函数：map、reduce、filter
'''
Python高阶函数：是指接收一个或多个函数作为参数，并返回一个新函数的函数
'''
# map：map函数接受一个函数和一个可迭代对象作为参数，并对可迭代对象中的每个元素应用函数，返回一个新的可迭代对象
# 将函数对象作用于list每个元素上，返回一个map对象
nums = [1, 2, 3, 4]
squared = map(lambda e: e ** 2, nums)
print(list(squared))     # [1, 4, 9, 16]  求平方

# 案例eg:
ls = [['A', 'B'], ['A', 'B', 'C']]
ls_map = map(len, ls)    # 应用map()函数，len方法用于求长度大小
# 返回map类型，map转为list
l = list(ls_map)
print(l)                 # [2, 3]
# 获取子list最大长度
max_len = max(l)
print(max_len)           # 3
# 与lambda表达式的结合使用：给子list长度不够的补充None，这里list不支持使用apply方法，换用map方法
# ls.apply(lambda sub_ls: sub_ls + [None]*(max_len-len(sub_ls)))
ls = list(map(lambda sub_ls: sub_ls + [None] * (max_len - len(sub_ls)), ls))
print(ls)                # [['A', 'B', None], ['A', 'B', 'C']]

# reduce：reduce函数接受一个函数和一个可迭代对象作为参数，对可迭代对象中的元素进行聚合操作，返回最终结果
from functools import reduce
mul = reduce(lambda x, y: x * y, nums)
print(mul)      # 24  求积
# filter：filter函数接受一个函数和一个可迭代对象作为参数，根据函数的返回值是True还是False来过滤可迭代对象中的元素，返回一个新的可迭代对象
even = filter(lambda e: e % 2 == 0, nums)
print(list(even))      # [2, 4]  筛选偶数

# 7、Python闭包
'''
闭包是指在一个函数内部定义的函数，并且内部函数可以访问外部函数的变量；闭包可以捕获和保持外部函数的状态，即使外部函数已经执行完毕
闭包的用途：可以在全局使用局部的属性
'''
'''
outer_func是一个外部函数，它接受一个参数x。内部函数inner_func在外部函数内部定义，并且可以访问外部函数的变量x；通过调用
outer_func并将返回的内部函数对象赋值给add_three，可以创建一个新的函数add_three，该函数在调用时会将其参数与外部函数的参数相加
详细解释：outer_func函数的形参传入了3，此时就会在outer_func函数的命名空间声明x=3；内部函数inner_func
使用了外部函数的x变量，所以inner_func函数的返回的结果为调用outer_func函数传入的参数x加上再调用inner_func
函数传入的参数y；之所以要返回inner_func，是因为要把内部函数的内存地址返回出去，这样才能被变量所接收并调用
'''
def outer_func(x):
    def inner_func(y):
        return x + y
    return inner_func

add_three = outer_func(3)
print(add_three(5))      # 8
'''
函数对象：既是函数也是变量；函数不加括号可作变量使用，打印的是内存地址
'''
print(add_three)         # <function outer_func.<locals>.inner_func at 0x000001D90943DBD0>
#                          outer_func函数        内部的inner_func函数    inner_func的内存地址

# 8、Python装饰器
'''
装饰器是一种用于在函数或类的定义前面添加额外功能的语法；装饰器可以在不修改原函数或类定义的情况下，为其添加额外的行为
使用装饰器是因为在开发过程中，需要遵守开放（对拓展功能是开放的）封闭（对修改源代码是封闭的）原则
'''
"""
1）装饰器公式：
def 装饰器函数(被装饰的函数对象):
    def 闭包函数(*args, **kwargs):
        装饰之前的行为
        接收的变量 = 被装饰的函数(*args, **kwargs)
        装饰之后的行为
        return 接受的变量
    return 闭包函数对象
    
2）装饰器使用、语法糖
res = 装饰器函数(被装饰的函数对象)
语法糖简化：在被装饰的函数上添加注解@装饰器函数名（装饰器函数一定要在被装饰器函数的上面定义）
@装饰器函数名
def 被装饰的函数():
    ......
    
定义一个装饰器uppercase_decorator，它接受一个函数作为参数，并返回一个新的函数对象；装饰器中的wrapper函数可以
在调用原始函数之前或之后执行额外的操作。使用@符号将装饰器应用到greet函数上，使其在调用时自动经过装饰器处理
"""
def uppercase_decorator(func):
    # func=greet函数的内存地址
    def wrapper(text):
        # 执行被装饰的函数greet
        result = func(text)
        # 装饰之后添加的功能
        return result.upper()
    return wrapper

@uppercase_decorator
def greet(name):
    return "Hello, " + name + "！"

print(greet("Alice"))     # HELLO, ALICE！

# debug装饰器用于将debug属性添加到MyCla类中，并将其值设置为True
def debug(cls):
    cls.debug = True
    return cls

@debug
class MyCla:
    pass

print(MyCla.debug)        # True

# 9、常用Python内置函数

# 9.1、数学相关
# 1）abs(x) 取绝对值
print(abs(-1))           # 1
# 2）divmod(x,y) 结合除法和余数运算，返回包含商和余数的元组
print(divmod(5, 2))      # (2, 1)
# 3）pow(x,y) x的y次方
print(pow(2, 3))         # 8
# 4）round(x,n) 四舍五入，保留n位小数
print(round(2.718, 2))   # 2.72
# 5）min()、max()、sum() 返回序列最小值、最大值、和
# 6）oct(x)、hex(x)、bin(x) 转8进制、16进制、2进制
# 7）chr(i) 返回输入数字(范围为0-256)对应的字符
print(chr(65))           # A

# 9.2、序列相关
# 1）len(obj) 返回字符串或序列的长度
# 2）range(sta,end,step) 创建序列
# 3）zip(iter1,...) 拉链，对应位置的元素生成一个新的元组类型的迭代器
x = ['a', 'b', 'c']
y = [2, 3, 5]
print(list(zip(x, y)))          # [('a', 2), ('b', 3), ('c', 5)]
# 4）sorted(iter) 对所有可迭代的对象进行排序，reverse参数: 排序方式：True降序，False升序(默认)
print(sorted(y, reverse=True))  # [5, 3, 2]
# 5）reversed(sep) 反转序列，生成一个新的序列
print(list(reversed(y)))        # [5, 3, 2]
# 6）enumerate(iter) 枚举，将一个可遍历的对象组合为一个索引和数据的序列
print(list(enumerate(y)))       # [(0, 2), (1, 3), (2, 5)]
# eg：
for e in enumerate(x):
    print(e)                    # (0, 'a')
                                # (1, 'b')
                                # (2, 'c')
for i, j in enumerate(ls):
    print('标签'+str(i)+' : '+j)
# 标签0 : a
# 标签1 : b
# 标签2 : c
# 7）any()、all()内置函数
# all(iter) 判断给定的可迭代对象元素中是否含有元素为0、''、False，含有返回False，其它返回True(包括空迭代器)
'''
all()：判断可迭代对象所有元素是否都是真值（非0、非空、非None、非False等），返回bool类型
any()：判断可迭代对象元素是否存在真值（非0、非空、非None、非False等），返回bool类型
'''
import numpy as np

# 注意：np.nan是真值
print(bool(np.nan))     # True

print(all([True, False, 3]))    # False 含有False
print(all(['', 3]))             # False 含有''
print(any(['', False, 0, None]))            # False
print(any(['', False, 0, None, np.nan]))    # True

# 基本使用
nums = [2, 3, 5, 7, 11]

# 判断列表元素是否全部奇数
print(all(num % 2 != 0 for num in nums))    # False
# 判断列表元素是否存在偶数
print(any(num % 2 == 0 for num in nums))    # True

# 8）iter()、next()、slice()见py01.py迭代器与生成器

# 9.3、对象相关
# 1）dir(obj) 收集对象的信息，返回参数的属性、方法列表
# 2）id(obj) 返回对象的内存地址
# 3）hash(obj) 返回对象的哈希值
# 4）type(obj) 返回对象的类型
print(type(10))                 # int
print(type(3.14))               # float
print(type('abc'))              # str
# 5）getattr(obj, obj_field_or_method_str, default)
#            对象   对象属性或方法的字符串    默认返回值
# 功能：用于调用一个对象的属性或方法
# eg：
class Person(object):
    field = 'name'

    def method(self, age):
        print("method方法执行啦!")
        return f"{self.field} : {age}"

p = Person()

# 调用属性
print(getattr(p, 'field'))         # name
# 调用执行方法（传参外面使用()）
getattr(p, 'method')(18)           # method方法执行啦!
# 调用执行方法，并打印返回值（传参外面使用()）
print(getattr(p, 'method')(18))    # method方法执行啦!
                                   # name : 18

# getattr()与getattribute()区别
# Python中所有类默认继承了Object类，Object提供了很多原始的内建属性和方法
'''
getattr(obj,attr,default)：Python内置的一个函数，可用来获取对象的属性和方法
getattribute(self,attr)：__getattribute__()是Python属性访问拦截器，当类的属性被实例访问时，优先调用__getattribute__()方法，无论属性存不存在
当__getattribute__()找不到对象属性时，才会调用__getattr__()方法；若直接使用类名.属性调用类属性时，则不会调用__getattribute__()方法
'''

# 10、递归函数
'''
在一个函数的调用过程中，直接或间接地调用了函数本身的函数
'''
# eg：二分查找：给定一个有序列表和需要查找的值，先获取列表中索引为中间的值，然后将查询的值与其对比
# 当查询的值大于中间值时，那么就以中间为分割，往右查；当查询的值小于中间值时，那么就以中间为分割，往左查
# 当查询的值等于中间值时，即为找到

def find_from_list(num, ls):
    mid_index = int(len(ls) / 2)
    mid_num = ls[mid_index]
    if mid_num > num:
        ls = ls[:mid_index]
        find_from_list(num, ls)
    elif mid_num < num:
        ls = ls[mid_index+1:]
        find_from_list(num, ls)
    else:
        print("找到了！")

find_from_list(7, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 11、回调函数
"""
将一个函数作为参数传递给另一个函数，并在需要时调用，可以降低函数间的耦合度
"""
# 1）异常处理
def auto_task(err_callback):
    try:
        # print("没有错误")
        print(f"出现错误: {1 / 0}")
    except Exception as e:
        err_callback(e)

def err_deal(err):
    print(f"处理错误: {err}")

auto_task(err_deal)

# 2）定时器与时间循环
import time

def timer_callback(n):
    print(f"定时器启动，执行第 {n} 次".center(20, '*'))

def set_time(timer, callback):
    n = 0
    while True:
        time.sleep(timer)
        n += 1
        # 使用回调函数对定时器进行处理
        callback(n)

set_time(3, timer_callback)

# 3）异步编程：通常用于处理较长时间运行是任务结果
import asyncio

async def long_time_task():
    # 模拟等待一段时间
    await asyncio.sleep(5)
    return 'result'

def callback(result):
    print(f"处理任务结果: {result}")

async def run_task():
    result = await long_time_task()
    callback(result)

asyncio.run(run_task())

# 4）迭代器与生成器：见py02.py

# 12、值传递与引用传递
# 1）情况1
a, b = 10, 11

def swap(x, y):
    x, y = y, x
    return f"{x} {y}"

print(swap(a, b))      # 11 10
print(a, b)            # 10 11
# 结论1：swap()函数并不会改变实际参数a，b的值，Python函数参数是按照值传递的

# 2）情况2
ls = [10, 11]

def append(seq):
    seq.append(12)
    return seq

print(append(ls))      # [10, 11, 12]
print(ls)              # [10, 11, 12]
# 结论2：append()函数改变了实际参数ls的值，Python函数参数是按照引用传递的

# 总结：凡是对原对象操作的函数，都会影响传递的实际参数；凡是生成了新对象的操作，都不会影响传递的实际参数
# 值传递：是传递变量的值，不会改变函数外面变量的值。不可变对象（如String、Tuple、Number等）用的是值传递
# 引用传递：是传递对象的地址，会改变对象本身的值。可变对象（如List、Dict、Set等）用的是引用传递

