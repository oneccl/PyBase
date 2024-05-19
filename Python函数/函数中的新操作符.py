"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/12/2
Time: 16:35
Description:
"""
# Python函数中的操作符：/、*、...

# Python新操作符：/与*
'''
def func(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2): ...
         -----------   -----------     ----------
           |                |               |
           |       Positional or keyword    |
           |                                Keyword only
           Positional only
'''
# /和*是可选的
# /前面的参数都是仅位置参数(Positional-Only Parameter)，即参数只能通过位置参数的形式传入函数，不能通过关键字的形式传入函数
# /和*之间的是位置或关键字参数(Positional-or-Keyword Parameter)，即参数可以通过位置参数的形式参入函数，也可以通过关键字的形式传入函数
# *后面的参数是仅关键字参数（Keyword-Only Parameter)，即只能通过关键字传入参数
# 总结：
# /符号之前的所有参数，都必须以位置参数穿参，不能使用关键字参数传参
# *符号之后的所有参数，都必须以关键字方式传参，不能使用位置参数传参
# /与*符号之间的所有参数，既能使用关键字方式传参，也能使用位置参数传参

def f1(p1, p2, /, pk, *, k1, k2):
    print(p1 + p2 + pk + k1 + k2)

f1(2, 3, pk=1, k1=5, k2=4)    # 15
f1(2, 3, 1, k1=5, k2=4)       # 15

# Python不支持*在/前，虽然方法本身不会报错，但函数还没调用就会执行报错：SyntaxError: invalid syntax
# def f2(p1, p2, *, pk, /, k1, k2):
#     print(p1 + p2 + pk + k1 + k2)

# 打包与解包
# *操作的本质是对可迭代对象的解包
# 打包
scores = 80, 90, 900, 120
# 解包：被解包的序列中的元素数量必须与赋值符号=左边元素的数量完全一样
a, *b, c = scores
print(a, b, c)    # 80 [90, 900] 120
# 打包
scores = [80, 90]
# 解包：不能将*操作符用于表达式的右边
x, y = scores     # 80 90
print(x, y)

# *args将多个参数打包成一个元组
# **kwargs将多个K-V参数打包成一个字典

# Python新操作符：…
# 操作符...表示函数没有实现任何代码，跟pass作用类似
def m(): ...
# 等价于
def n(): pass



