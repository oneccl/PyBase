"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/9/21
Time: 22:56
Description:
"""

# Python单例模式

# 单例模式是最常见的一种设计模式，该模式确保系统中一个类有且仅有一个实例
# 1）使用模块
# 模块是天然单例的，因为模块只会被加载一次，加载后，其他脚本若导入使用时，会从sys.modules中找到已加载好的模块，多线程下也是如此
# 编写Singleton.py脚本
class MySingleton():
    def __init__(self, name, age):
        self.name = name
        self.age = age
# 其他脚本导入使用
# from Singleton import MySingleton
# single1 = MySingleton('Tom', 18)
# single2 = MySingleton('Bob', 20)
#
# print(single1 is single2)     # True

# 2）使用装饰器
# 可以通过装饰器装饰需要支持单例的类
from threading import RLock

def Singleton(cls):
    single_lock = RLock()
    instance = {}
    def singleton_wrapper(*args, **kwargs):
        with single_lock:
            if cls not in instance:
                instance[cls] = cls(*args, **kwargs)
        return instance[cls]
    return singleton_wrapper

@Singleton
class MySingleton(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

# 该方式线程不安全，需要加锁校验

single1 = MySingleton('Tom', 18)
single2 = MySingleton('Bob', 20)

print(single1 is single2)     # True

# 3）使用__new__()方法
# Python的__new__()方法是用来创建实例的，可以在其创建实例的时候进行控制

class MySingleton(object):
    single_lock = RLock()
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __new__(cls, *args, **kwargs):
        with MySingleton.single_lock:
            if not hasattr(MySingleton, '_instance'):
                MySingleton._instance = object.__new__(cls)
        return MySingleton._instance

single1 = MySingleton('Tom', 18)
single2 = MySingleton('Bob', 20)

print(single1 is single2)     # True


