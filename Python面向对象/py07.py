
# Python类的高级特性

"""
1、Python类属性与实例属性
在Python中，类属性是属于类本身的属性，所有实例共享该属性。实例属性是属于每个实例的属性，每个实例都有自己独立的副本
"""
class Person:
    # 类属性：所有Person类的实例共享该属性
    species = "Human"

    def __init__(self, name):
        # 实例属性：每个Person类的实例都有自己独立的name属性
        self.name = name

# 访问类属性：
print(Person.species)       # Human

# 创建实例
person1 = Person("Alice")
person2 = Person("Bob")
# 访问实例属性：
print(person1.name)         # Alice
print(person2.name)         # Bob

# 所有Person类的实例共享类属性：
print(person1.species)
print(person2.species)

'''
2、Python实例方法、类方法、静态方法
在Python类中，可以定义类方法和静态方法
类方法：是在类级别上操作的方法，可以通过类本身或类的实例调用。类方法使用@classmethod装饰器来定义，并且第一个参数是类本身（通常命名为cls）
静态方法：是与类相关的方法，但不需要访问类或实例的属性。静态方法使用@staticmethod装饰器来定义，没有额外的参数
'''
class MathUtils:
    # 类属性
    tmp = 10
    # 构造方法
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # 示例方法
    def sub(self, x, y):
        return x-y
    # 类方法
    @classmethod
    def add(cls, x, y):
        return x+y
    # 静态方法
    @staticmethod
    def multiply(x, y):
        return x*y

# 实例方法：方法内部既可以获取构造函数定义的变量，也可以获取类的属性值，只能通过实例调用。self是类的实例
o = MathUtils(3, 2)
print(o.sub(3, 2))                # 实例调用实例方法
print(o.y)                        # 实例访问构造函数属性
print(o.tmp)                      # 实例访问类属性
# 类方法：通过装饰器@calssmethod进行修饰。方法内部不能获取构造函数定义的变量，可以获取类的属性。调用方式：类名.类方法名或实例化调用。cls是类而不是实例
print(MathUtils.add(7, 5))        # 类访问类方法
print(o.add(7, 5))                # 实例调用类方法
print(MathUtils.tmp)              # 类访问类属性
# 静态方法：通过装饰器@staticmethod进行修饰。方法内部不能获取构造函数定义的变量，也不可以获取类的属性。调用方式：类名.静态方法名或实例化调用
print(MathUtils.multiply(6, 4))   # 类访问静态方法

# Python实例方法、类方法、静态方法区别：
'''
实例方法：既可以获取构造函数定义的变量，也可以获取类的属性值，只能通过实例调用。self是类的实例
类方法：通过装饰器@calssmethod进行修饰。不能获取构造函数定义的变量，可以获取类的属性。调用方式：类名.类方法名或实例化调用。cls是类而不
静态方法：通过装饰器@staticmethod进行修饰。不能获取构造函数定义的变量，也不可以获取类的属性。调用方式：类名.静态方法名或实例化调用
'''


'''
3、Python的特殊方法和运算符重载
Python特殊方法是以双下划线开头和结尾的方法，用于定义类的特定行为。通过定义特殊方法，可以实现运算符重载和自定义类的行为
'''
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 定义Vector类的__add__()特殊方法，实现向量的相加功能
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    # 定义Vector类的__str__()特殊方法，自定义向量对象的字符串表示
    def __str__(self):
        return f"({self.x}, {self.y})"

# 创建2个向量对象：
v1 = Vector(2, 3)
v2 = Vector(4, 5)

# 使用运算符重载进行向量相加
v3 = v1 + v2
print(v3)          # (6, 8)

'''
4、反射
反射机制是指在程序的运行状态中，动态获取对象的属性和方法
方法汇总：
1）hasattr(object,属性)：按字符串判断对象有无该属性，返回值为bool类型
2）getattr(object, 属性, 返回值)：当对象object不存在该属性时，返回自定义返回值，存在时返回对应值
3）setattr(object, 属性, 值)：修改或添加对象属性，有属性则修改，没有则添加
4）delattr(object, 属性)：删除对象的该属性
'''
# getattr()与getattribute()区别：

# Python中所有类默认继承了Object类，Object提供了很多原始的内建属性和方法
# 1）getattr(obj,attr,default)：Python内置的一个函数，可用来获取对象的属性和方法
# 2）getattribute(self,attr)：__getattribute__()是Python属性访问拦截器，当类的属性被实例访问时，优先调用__getattribute__()方法，无论属性存不存在
# 当__getattribute__()找不到对象属性时，才会调用__getattr__()方法；若直接使用类名.属性调用类属性时，则不会调用__getattribute__()方法

# Python自省：

# 在计算机编程领域里，自省是指程序运行时判断对象的类型的能力。Python中比较常见的自省函数有：
# type()、dir()、getattr()、hasattr()、isinstance()等，通过这些函数，我们能够在程序运行时获知对象的类型，判断对象的属性等

