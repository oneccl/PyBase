

# 1、Object与Type

# 对象（Object）和类型（Type）是Python中两个最基本的概念，它们是构筑Python语言大厦的基石
# 所有的数据类型，值，变量，函数，类，实例等一切可操作的基本单元在Python中都是对象（Object），每个对象都有三个基本属性：ID、类型和值
a = 1
print(id(a), type(a), a)    # 1958094307568 <class 'int'> 1
# id()内建方法获取对象的唯一编号，它是一个整数，通常就是对象的内存地址。type()内置方法获取对象的类型（Type）
# 一个对象可能有一个或多个基类（Bases），当一个对象表示数据类型时，比如int对象，它就具有了__bases__属性
print(int.__bases__)    # (<class 'object'>,)
# type和bases定义了该对象与其他对象间的关系，实际上对象内的type和bases均指向其他对象，是对其他对象的地址引用

# 一个对象必有Type属性，同样Type是不能脱离开对象存在的
# type()内置方法获取对象的类型。我们也可以使用对象.__class__来获取对象的类型，它们是等价的
print(type(a))        # <class 'int'>
print(a.__class__)    # <class 'int'>

# Class和Type均是指类型（Type），Class通常用于普通用户使用class自定义的类型。Type通常指Python的解释器CPython内置的类型
# CPython提供内置方法type()而没有定义class()，因为它们本质是一样的，只是不同的语境产生的不同说法

# Python中的对象之间存在两种关系：
# 1）父子关系或继承关系（Subclass-Superclass或Object-Oriented），如“猫”类继承自“哺乳动物”类，我们说猫是一种哺乳动物。对象的__bases__属性记录这种关系，可以使用issubclass()判断
# 2）类型实例关系（Type-Instance），如“米老鼠是一只老鼠”，这里的米老鼠不再是抽象的类型，而是实实在在的一只老鼠。对象的__class__属性记录这种关系，可以使用isinstance()判断
# Python把对象分为两类：类型对象（Type）和非类型对象（Non-type）
"""
int、type、list等均是类型对象，可以被继承，也可以被实例化
1、[1]等均是非类型对象，它们不可再被继承和实例化，对象间可以根据所属类型进行各类操作，比如算数运算
"""
# object和type是CPython解释器内建对象，它们的地位非常特殊，是Python语言的顶层元素：
# 1）object是所有其他对象的基类，object自身没有基类，它的数据类型被定义为type
# 2）type继承了object，所有类型对象都是它的实例，包括它自身。判断一个对象是否为类型对象，就看它是否是type的实例
# isinstance()内置方法本质是在判断对象的数据类型，它会向基类回溯，直至回溯到object
print(isinstance(object, type))      # True
print(isinstance(type, object))      # True
print(isinstance(type, type))        # True
print(isinstance(object, object))    # True
print(object.__class__)              # <class 'type'>
print(type.__class__)                # <class 'type'>
# Python中还定义了一些常量，比如True、False。其中有两个常量None和NotImplemented比较特殊，通过type()可以获取它们的类型为NoneType和NotImplementedType，这两个类型不对外开放，即普通用户无法继承它们，它们只存在于CPython解释器中
print(type(None))                    # <class 'NoneType'>
print(type(NotImplemented))          # <class 'NotImplementedType'>

# 2、类的常用特殊方法

# 1）__str__
# __str__方法用于str()函数转换中，默认使用print()方法打印一个对象时，就是对它的调用，我们可以重写这个函数还实现自定义类向字符串的转换

# 2）__repr__
# repr()函数调用对象中的__repr__()方法，返回一个Python表达式，通常可以在eval()中运行它

# 3）attr方法
# Python在object基类中提供了3个与属性操作相关的方法：
# __delattr__：用于del语句，删除类或者对象的某个属性
# __setattr__：用于动态绑定属性
# __getattribute__：在获取类属性时调用，无论属性是否存在

# 4）attr内置方法
# Python提供了三个内置属性方法getattr()、setattr()和hasattr()，分别用于获取、设置和判定对象的属性
# 既然我们已经可以通过对象名直接访问它们，为何还要使用这些函数呢？通过它们我们可以对任意一个我们不熟悉的对象进行尝试性访问，而不会导致程序出错

# getattr()方法最大的用途在于如果对象没有相应属性，可以不报错AttributeError，可以为它指定一个默认值

# 5）__init__
# 类的构造方法

# 6）__new__
# 控制创建类的实例

# 7）__call__
# __call__具有非常特殊的功能，可以将一个对象名函数化。实现了__call__()函数的类，其实例就是可调用的（Callable）。可以像使用一个函数一样调用它
# 装饰器类就是基于__call__()方法来实现的。__call__()只能通过位置参数来传递可变参数，不支持关键字参数，除非函数明确定义形参
# 可以使用callable()方法来判断一个对象是否可被调用，也即对象能否使用()括号的方法调用

# 参考文档：https://pythonhowto.readthedocs.io/zh-cn/latest/object.html#id15

# 示例：
class Stu(object):

    # 构造方法
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # toString()方法
    def __str__(self):
        return f"Stu({self.name}, {self.age})"

    # attr方法
    def __getattribute__(self, item):
        print(f"getattribute: {item}")
        # 调用object的__getattribute__()
        return super().__getattribute__(item)
    # attr方法
    def __setattr__(self, key, value):
        print(f"setattr: {key}")
        # 调用object的__setattr__()
        super().__setattr__(key, value)

    # 实例创建方法
    def __new__(cls, *args, **kwargs):
        print(cls)
        return super().__new__(cls)

    # 实例函数化方法
    def __call__(self, *args, **kwargs):
        print(*args)
        return "call方法"


# 1）__str__()、__init__()、__new__()
stu = Stu('Tom', 18)
print(stu)
'''
<class '__main__.Stu'>
setattr: name
setattr: age
getattribute: name
getattribute: age
Stu(Tom, 18)
'''

# 2）__repr__()
print(repr("3 + 2"))         # '3 + 2'
print(eval(repr("3 + 2")))   # 3 + 2

# 3）attr方法
# 调用类对象的__getattribute__()
print(stu.name)
'''
getattribute: name
Tom
'''
# 调用类对象的__setattr__()
stu.age = 19
'''
setattr: age
'''

# 4）attr内置方法
print(hasattr(stu, "name"))
'''
getattribute: name
True
'''
# setattr()方法可以给对象添加属性
setattr(stu, "addr", "US")     # setattr: addr
# 如果对象没有相应属性，如果不想程序报错AttributeError，可以为它指定一个默认值
print(getattr(stu, "addr", "default"))
'''
getattribute: addr
US
'''

# 5）__call__()
print(callable(stu))    # True
print(stu('arg0', 'arg1'))
'''
arg0 arg1
call方法
'''


# 8）内置方法与对应操作
# 算术运算、比较运算、赋值运算、位运算、逻辑运算、成员运算、身份运算、其他
# 参考文档：https://pythonhowto.readthedocs.io/zh-cn/latest/object.html#id19

# 3、Python多重继承
# 继承是面向对象编程的一大特征，继承可以使得子类具有父类的属性和方法，并可对属性和方法进行扩展。Python中继承的最大特点是支持多重继承，也即一个类可以同时继承多个类
# 我们可以在新类中使用父类定义的方法，也可以定义同名方法，覆盖父类的方法，还可以在自定义的方法中使用super()调用父类的同名方法
# 那么如果从多个类继承，多个类中又实现了同名的方法，如何确定它们的继承顺序呢？
class A(object):
    def f0(self):
        print('A f0')

    def f1(self):
        print('A f1')

class B(object):
    def f0(self):
        print('B f0')

    def f1(self):
        print('B f1')

class C(A, B):
    def f0(self):
        print('C f0')

class D(B, A):
    def f1(self):
        print('D f1')

# __mro__属性用于记录类继承的关系，返回一个元组类型
print(C.__mro__)    # (<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class 'object'>)
c = C()
c.f0()              # C f0
print(D.__mro__)    # (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.A'>, <class 'object'>)
d = D()
d.f1()              # D f1
# 这种继承顺序被称为方法解释顺序（MRO，Method Resolution Order）。Python2.3版本后采用C3线性序列算法来计算MRO
# 类之间的继承关系可以用有向无环图（DAG，Directed Acyclic Graph）来描述，每个顶点代表一个类，顶点之间的有向边代表类之间的继承关系。C3算法对所有顶点进行线性排序

# 参考文档：https://pythonhowto.readthedocs.io/zh-cn/latest/object.html#id29

