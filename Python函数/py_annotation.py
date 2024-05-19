"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/9/20
Time: 21:54
Description:
"""

# Python类型注解、泛型、注解

# 1、Python类型注解

# 1）变量注解
# Python是动态语言，例如在声明变量时，不需要显示声明其类型，程序运行时提供了隐式推断
a = 10
print(a + 1)
# 优缺点：
'''
优点：自由灵活，简化代码
缺点：若某些变量类型有错，IDE工具无法早期进行纠错，只能在程序运行阶段才能暴露问题
'''
# 例如：
# a = '10'
# print(a + 1)
# Python在3.5之后引入了类型注解，其作用是可以明确的声明变量类型
# 例如：
a: str = '10'
print(int(a) + 1)

# 2）函数注解
def say_hello(name: str) -> str:
    return f"Hello {name}"

print(say_hello('Alice'))

# 带默认值的函数
def add(x: int = 0, y: int = 0) -> float:
    return x + y

print(add(2, 3))

# 3）复合注解：容器类型
# typing模块是Python的标准库之一，提供了一些复杂的容器类型，例如List，Dict，Tuple，Set
# Python3.9+版本不再需要typing模块，内置的容器类型直接支持复合注解（使用小写）
from typing import List, Dict, Tuple, Set

# 元素为float的列表
scores: list[float] = [80, 85, 90.5, 78.5]
print(scores)
# Key为str，Value为int的字典
stu: dict[str, int] = {'Tom': 18}
print(stu)
# 二元组
stu: tuple[str, int] = ('Tom', 18)
print(stu)
# Seq可迭代类型
from typing import Sequence as Seq

items: Seq[object] = [2, 'A', (5, 3), {'K': 'V'}]
print(items)

# 容器类型别名
Vector2D = tuple[int, int]
vector: Vector2D = (1, 2)
print(vector)

# 4）其他类型
# Optional可选类型，Union多种类型，Literal字面枚举类型，Any任何类型
from typing import Optional, Union, Literal, Any

# Optional：此函数可能返回None，也可能返回bool，可以使用Optional
def m1(f: int = 0) -> Optional[bool]:
    return True if f == 1 else False

# Union：返回多种类型中的一种
# 该函数可以返回str、int、float中的任一类型
def m2() -> Union[str, int, float]:
    pass

# Literal
Mode = Literal['r', 'rb', 'w', 'wb']
def m3(file: str, mode: Mode) -> str:
    pass

# 使用
m3('path', 'r')

# Any：任意类型
def m4() -> Any:
    pass

# 2、Python泛型

# 泛型可以巧妙地对类型进行了参数化，同时又保留了函数处理不同类型时的灵活性

# 若一个函数既要可以处理str又要能处理int
U = Union[str, int]
def m5(a: U, b: U) -> tuple:
    return (a, b)

print(m5('Tom', 18))     # ('Tom', 18)
print(m5(17, 18))        # (17, 18)
# 存在问题：类型检查通过，但参数的类型可以混合用

# TypeVar：用于定义泛型
from typing import TypeVar

# 定义泛型T，T必须是str或int其中一种
T = TypeVar('T', str, int)
def m6(a: T, b: T) -> list[T]:
    return [a, b]

# print(m6('Tom', 18))  # 混合使用类型检查不通过

# 定义泛型K和V，K和V的类型没有限制
K = TypeVar("K")
V = TypeVar("V")
def m7(key: K, d: dict[K, V]) -> V:
    return d[key]

d1 = {'id': 100}
d2 = {18: 'age'}
print(m7('id', d1))     # 100
print(m7(18, d2))       # age

# 3、Python注解

# Python在3.0之后添加了新特性Decorators，以@为标记修饰function和class，类似Java注解
# Python注解是通过装饰器实现的，@注解本质是一个函数
# Decorators用于修饰约束function和class，分为带参数和不带参数，影响原有输出

# 1）不带参数注解
def annotation1(func):
    def enhance(*args):
        print('前置增强！')
        func(*args)
    return enhance

@annotation1
def use1():
    print('annotation1 test')

use1()
'''
前置增强！
annotation1 test
'''

# 2）带参数注解（**kwargs）
def annotation2(**kwargs):
    def enhance(func):
        for item in kwargs.items():
            print(item)
        return func
    return enhance

@annotation2(age=18, name='Tom')
def use2():
    print('annotation2 test')

use2()
'''
('age', 18)
('name', 'Tom')
annotation2 test
'''

