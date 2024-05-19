# 1、流程控制（顺序/选择/分支结构）
"""
1.1、if语句：elif 相当于Java中的else if
if 条件1:
   语句1
elif 条件2:
   语句2
else:
   语句3

1.2、match-case语句：相当于Java中的switch-case语句
match 变量:
   case 值1:
       语句1
   case 值2:
       语句2
   case _:
       默认执行的语句：相当于Java中的default

1.3、三元表达式/三目运算符
变量 = bool表达式为True的取值 if bool表达式 else bool表达式为False的取值
"""
score = 88.0
if score >= 85:
    print("优秀")
elif score >= 75:
    print("一般")
elif score >= 60:
    print("及格")
else:
    print("不及格")

x = 10
res = '负数' if x < 0 else '非负数'
print(res)

# 2、循环语句
'''
2.1、while循环（Python中没有do...while循环）
while 条件:
   语句
   
2.2、while-else语句
while 条件:
   语句
else:
   当while条件为False时执行else语句（while循环结束后执行的语句）如果循环中遇到break，则不会执行else语句
   
2.3、for循环
for 变量 in 可迭代对象:
   语句
   
2.4、for-else语句
for 变量 in 可迭代对象:
   语句
else:
   for循环结束后执行的语句，如果循环中遇到break，则不会执行else语句
   
2.5、循环控制语句
1）break语句：跳出整个循环
2）continue语句：跳出当前循环，执行下一次循环
3）pass语句：用作占位符，不执行任何操作
'''
# range()函数：用于生成序列
'''
range(n)：[0,n)
range(m,n)：[m,n)
range(m,n,s)：[m,n) 其中s为步长
'''
# 计算10！
n = 10
k = 1
res = 0
while k <= n:
    res *= k
    k += 1

for i in range(10):
    if i % 2 != 0:
        continue
    elif i > 0:
        print(i)
        break

# 9*9乘法表
for i in range(1, 10):
    for j in range(1, i+1):
        print('%d * %d = %d' % (i, j, i*j), end='\t')
    print()  # 换行

# 找素数（质数）
j = 2
zs = []
while j < 10:
    count = 0
    for k in range(1, j+1):
        if j % k == 0:
            count += 1
    if count == 2:
        zs.append(j)
    j += 1

print(zs)

# 序列解包（并行遍历）
ls1 = [2, 3, 5]
ls2 = [6, 8, 10]
for (a, b) in zip(ls1, ls2):
    print(a, " : ", b)

# 3、推导式
'''
3.1、列表推导式
列表推导式可以通过循环和条件判断表达式配合使用，列表推导式返回一个列表，整个表达式需要写在[]内部
语法1：[表达式 for 变量 in 列表 [if 过滤条件]]
语法2：[表达式 [if 过滤条件 else 默认值] for 变量 in 列表]
解释：返回过滤条件成立、经过表达式处理的结果列表
列表推导式支持多层嵌套：[表达式 for 变量1 in 列表1 for 变量2 in 列表2 ...]
'''
# 过滤掉长度<=3的字符串列表，并将剩下的全转大写
names = ['Bob', 'Tom', 'Alice', 'Jerry']
res_names = [name.upper() for name in names if len(name) > 3]
print(res_names)        # ['ALICE', 'JERRY']

tuples = [(x, y) for x in range(3) for y in range(3)]
print(tuples)           # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

'''
3.2、字典推导式
字典推导式可以通过循环和条件判断表达式配合使用，字典推导式返回一个字典，整个表达式需要写在{}内部
语法1：{key表达式: value表达式 for key,value in 字典.items() [if 过滤条件]}
解释1：返回过滤条件成立，经过key表达式(结果key)、value表达式(结果value)处理的结果字典
语法2：{key表达式: value表达式1 [if 过滤条件 else value表达式2] for key,value in 字典.items()}
解释2：返回过滤条件成立，经过key表达式(结果key)、value表达式1(结果value)处理的结果与过滤条件不成立，经过key表达式(结果key)、value表达式2(结果value)处理的结果组成的字典
语法3：{key表达式: value表达式 for语句}
解释3：根据for语句中的变量拼凑key表达式和value表达式
'''
d1 = {'a': 10, 'B': 20, 'c': 17, 'D': 13}
# 1）获取字典中key是小写且value是偶数的键值对
d_res1 = {k: v for k, v in d1.items() if k.islower() and v % 2 == 0}
print(d_res1)           # {'a': 10}
# 2）将字典中所有小写key转大写，并将原来所有大写key对应的value加1
d_res2 = {k.upper(): v if k.islower() else v + 1 for k, v in d1.items()}
print(d_res2)           # {'A': 10, 'B': 21, 'C': 17, 'D': 14}

# 3）将字典中key、value互换位置
d3 = {1: 'a', 2: 'b', 3: 'c'}
dict5 = {k: v for v, k in d3.items()}
print(dict5)            # {'a': 1, 'b': 2, 'c': 3}

# 4）将两个长度相同的列表合并成字典
name = ['A', 'B', 'C']
age = [20, 19, 21]
dict3 = {name: age for name, age in zip(name, age)}
print(dict3)            # {'A': 20, 'B': 19, 'C': 21}

# 5）将字典中大小写相同的key对应的value聚合（结果中key以小写显示）
d2 = {'a': 2, 'B': 3, 'b': 5, 'A': 7, 'D': 11}
dict4 = {k: d2.get(k, 0) + d2.get(k.upper(), 0) for k in map(str.lower, d2.keys())}
print(dict4)            # {'a': 9, 'b': 8, 'd': 11}

# 6）将cookies字符串转化为字典
cookies = "mid=jy0ui55o; dep=GW; JSESSIONID=abcMkt; ick_login=a9b557b8"
# 方式1：
dict1 = {cookie.split("=")[0]: cookie.split("=")[1] for cookie in cookies.split(";")}
print(dict1)            # {'mid': 'jy0ui55o', ' dep': 'GW', ' JSESSIONID': 'abcMkt', ' ick_login': 'a9b557b8'}
# 方式2：其中([k, v],)为一个元素的元组
dict2 = {k: v for t in cookies.split(";") for k, v in (t.split("="),)}
print(dict2)            # {'mid': 'jy0ui55o', ' dep': 'GW', ' JSESSIONID': 'abcMkt', ' ick_login': 'a9b557b8'}

'''
3.3、集合推导式
集合推导式可以通过循环和条件判断表达式配合使用，集合推导式返回一个集合，整个表达式需要写在{}内部
语法：{表达式 for 变量 in 集合 [if 过滤条件]}
解释：返回过滤条件成立、经过表达式处理的结果集合
'''
# 过滤掉长度<=3的字符串集合，并将剩下的全转大写
names = {'Bob', 'Tom', 'Alice', 'Jerry'}
set_names = {name.upper() for name in names if len(name) > 3}
print(set_names)        # {'JERRY', 'ALICE'}

'''
3.4、元组推导式（生成器表达式）
元组推导式可以通过循环和条件判断表达式配合使用，元组推导式返回一个元组，整个表达式需要写在()内部
语法：(表达式 for 变量 in 元组 [if 过滤条件])
解释：返回过滤条件成立、经过表达式处理的结果元组
'''
# 过滤掉长度<=3的字符串，并将剩下的全转大写
names = ('Bob', 'Tom', 'Alice', 'Jerry')
tup_names = (name.upper() for name in names if len(name) > 3)
print(tup_names)         # <generator object <genexpr> at 0x00000231712F05F0>
print(tuple(tup_names))  # ('ALICE', 'JERRY')

# 4、迭代器与生成器
'''
4.1、迭代器
在Python中，迭代器（iterator）是一种用于遍历集合元素的对象；迭代器提供了一种统一的访问集合元素的方式，无需关注底层数据结构
迭代器可以通过iter()函数来创建，然后使用next()函数来逐个获取/访问元素；当没有更多元素可供迭代时，会抛出StopIteration异常
'''
fruits = ["apple", "banana", "orange"]
it = iter(fruits)
while True:
    try:
        fruit = next(it)
        print(fruit)
    except StopIteration:
        break

'''
4.2、生成器
生成器（generator）是一种特殊的迭代器，使用了yield关键字的函数称为生成器；生成器可以在迭代过程中逐个生产值
而不是一次性返回所有结果；当在函数中使用yield语句时，函数的执行将暂停，并将yield后面的表达式作为当前迭代的值返回
每次调用生成器的next()方法或使用for循环进行迭代时，生成器函数可以逐个产生值，调用一个生成器函数，返回一个迭代器对象
'''
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()

# 调用生成器的next()方法逐个生产值
print(next(fib))    # 0
print(next(fib))    # 1
print(next(fib))    # 1

# 使用for循环逐个生产值
for i in range(10):
    print(next(fib))

# Python迭代器与生成器区别
'''
生成器就是一种特殊的迭代器，能做到迭代器能做的所有事
生成器是高效的，使用生成器表达式取代列表解析，会非常节省内存
生成器除了创建和保持程序状态的自动生成，当发生器终结时，会抛出出StopIteration异常
'''

