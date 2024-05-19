
# 面向对象
"""
1）面向对象的基本思想：
面向对象编程（OOP）是一种编程范式，将数据和操作数据的方法封装在一起，以对象的形式呈现。它的基本思想是将现实
世界中的事物抽象成类（Class），通过实例化类来创建对象（Object），并通过对象之间的交互来完成程序的设计与开发
2）面向对象的基本概念：
类（Class）：类是对象的抽象。它是一种自定义数据类型，定义了一类对象的属性和行为。类是对象的模板，描述了对象应有的属性和行为
对象（Object）：对象是类的实例。它是类的具体表现，具有类定义的属性和行为。对象可以通过类来创建，每个对象都是类的一个实例
属性（Attribute）：属性是对象的数据，描述了对象的特征和状态。在类中，属性可以是变量或数据成员
方法（Method）：方法是对象的行为，定义了对象的操作和功能。在类中，方法是与类相关联的函数
3）面向对象的特点/特征：
封装（Encapsulation）：封装是指将数据和操作数据的方法封装在一起，隐藏了对象的内部实现细节，对外部只暴露必要的接口。通过封装，对象的使用者可以使用对象的方法来操作数据，而不需要关心内部的具体实现
继承（Inheritance）：继承是指一个类可以继承另一个类的属性和方法。被继承的类称为父类或基类，继承的类称为子类或派生类。子类可以继承父类的属性和方法，并且可以在此基础上添加新的属性和方法
多态（Polymorphism）：多态是指不同类型的对象可以通过相同的接口来进行操作。同一种方法可以根据不同对象的类型表现出不同的行为。多态提高了代码的灵活性和可扩展性
通过封装、继承和多态的机制，面向对象编程可以提高代码的可读性、可维护性和可扩展性，使代码更具有结构化和模块化的特点
"""

# 类与对象
'''
1、类的定义与实例化
类中的标识符：__开头：类的私有成员；__开头和结尾：特殊专用方法（如__init__()代表类的构造函数）
'''
class Emp:
    # 变量访问：Emp.empCount
    empCount = 0

    # 类的构造方法，初始化类的实例将调用该方法
    # self代表类的实例，self在定义方法时必须包含，调用时不必传入
    def __init__(self,name,age):
        self.name = name
        self.age = age
        Emp.empCount += 1

    def displayCount(self):
        print("Total Emps is %d" % Emp.empCount)

    def displayEmp(self):
        print("Name: ", self.name, "; Age: ", self.age)

# 创建实例对象
emp = Emp("Tom", 20)

# 访问成员
print(emp.name)
emp.displayCount()
emp.displayEmp()

# Python内置类属性
print("类名："+Emp.__name__)           # Emp
print("类所在模块："+Emp.__module__)    # __main__
# print("类的文档字符串："+Emp.__doc__)

class Person:
    # Python构造方法：在类定义中，可以定义特殊的方法__init__()作为构造方法（Constructor）
    # 用于在创建对象时初始化对象的属性；构造方法在实例化对象时自动调用
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print("Person", self.name, "Created")

    # Python析构方法：在类定义中，可以定义特殊的方法__del__()作为析构方法（Destructor）
    # 用于在对象被销毁时执行一些清理工作。析构方法在对象被垃圾回收时自动调用
    def __del__(self):
        print("Person", self.name, "Destroyed")

    # self关键字：在类的方法中，第一个参数通常被命名为self，它表示对象自身；通过self可以访问对象的属性和调用对象的方法
    def greet(self):
        print("Hello！My name is", self.name, "and I am", self.age, "years old.")

# 实例化一个对象：
person = Person("Alice", 18)
# 访问对象的属性
print(person.name)  # Alice
# 调用对象的方法
person.greet()      # Hello！My name is Alice and I am 18 years old.
# 销毁对象
del person          # Person Alice Destroyed

# 封装

# Python封装之property
# 封装是指私有化一个类所有属性，提供公共的get和set方法以供访问，但是Python不推荐，Python有property
# property：是一个装饰器，被装饰的方法是属性方法，既是属性也是方法

class Stu:

    def __init__(self, name: str, age: int):
        self.__name = name
        self.__age = age

    # get()方法
    @property
    def name(self):
        return self.__name

    # set()方法
    @name.setter
    def name(self, value):
        self.__name = value

    # get()方法
    def get_age(self):
        return self.__age

    # set()方法
    def set_age(self, age):
        self.__age = age

    def __str__(self):
        return f"Stu({self.__name},{self.__age})"

s1 = Stu('Tom', 18)
print(s1.name)         # Tom
print(s1.get_age())    # 18

s1.name = 'Tim'
s1.set_age(19)
print(s1)              # Stu(Tim,19)

# 继承与多态

'''
2、Python类的继承
类的继承：Python允许继承多个类
语法：class 子类/派生类(父类/基类):
在Python中，一个类可以通过继承来获得另一个类的属性和方法。被继承的类称为父类或基类，继承的类称为子类或派生类
子类可以继承父类的属性和方法，并且可以在此基础上添加新的属性和方法，或者重写父类的方法
'''
class Animal:
    # __开头的属性或方法表示私有成员(不能被继承)；_开头的属性或方法表示protected成员
    __addr = ""

    def eat(self):
        print("The animal is eating.")

    def sound(self):
        print("The animal is sounding.")

class Dog(Animal):
    def bark(self):
        print("The dog is barking.")

    # 方法重写：
    def sound(self):
        print("旺旺旺！")

dog = Dog()
# 继承
dog.eat()        # The animal is eating.
dog.bark()       # The dog is barking.
# 方法重写
dog.sound()      # 旺旺旺！
# 调用父类成员：super().父类属性或方法

# 抽象类、抽象基类

'''
3、Python类的多态
'''
# Animal类是一个抽象类，定义了make_sound()方法。Dog类和Cat类分别继承了Animal类，并实现了自己的make_sound()方法
class Animal:
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        print("The dog barks.")

class Cat(Animal):
    def make_sound(self):
        print("The cat meows.")

# make_sound()函数接受一个Animal类型的参数，并调用该参数的make_sound()方法
# 通过使用多态，当传递Dog对象或Cat对象给make_sound()函数时，它们将分别输出The dog barks.和The cat meows.
def make_sound(animal):
    animal.make_sound()

animals = [Dog(), Cat()]

for animal in animals:
    make_sound(animal)

# 多态的使用：
# Python内置函数：isinstance(obj, cla)和issubclass(sub_cla,sup_cla)
# 1）isinstance(obj, cla)：obj：实例/对象  cla：类名、基本类型，也可以是由它们组成的元组
# 如果obj是cla的实例或子类的实例则返回True，否则返回False
s1 = "abc"
print(isinstance(s1, str))          # True
print(isinstance(s1, (str, int)))   # True
print(isinstance(s1, int))          # False
# 2）issubclass(sub_cla,sup_cla)：如果sub_cla是sup_cla子类 返回True，否则返回False
print(issubclass(Dog, Animal))      # True

'''
4、Python基类与抽象基类
'''
# object是所有类的基类（Base Class，也被称为超类（Super Class）或父类），如果一个类在定义中没有明确定义继承的基类，那么默认就会继承object
class Stu:
    pass
# 等价于
class Stu(object):
    pass
# 打印类的继承关系
# __mro__属性用于记录类继承的关系，返回一个元组类型
print(Stu.__mro__)        # (<class '__main__.Stu'>, <class 'object'>)
from abc import ABC, abstractmethod

# Python提供了abc模块，用于定义抽象基类（Abstract Base Classes）。抽象基类是一种包含抽象方法的类，不能被直接实例化，只能被继承
# Animal类是一个抽象基类，通过在make_sound()方法上使用@abstractmethod装饰器来定义抽象方法
class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

# Dog类继承了Animal类，并实现了make_sound()方法。只有当子类实现了父类的所有抽象方法时，才能被实例化
class Dog(Animal):
    def make_sound(self):
        print("The dog barks.")

dog = Dog()
dog.make_sound()      # The dog barks.

'''
5、Python枚举类
枚举是一组绑定到唯一常数值的符号名称，具备可迭代性和可比较性
'''
from enum import Enum, unique

# 创建枚举类
# 枚举名称name必须唯一，枚举值value可重复，@unique装饰器用于设置枚举值唯一
# @unique
class HttpStatus(Enum):
    OK = 200
    NOT_FOUND = 404
    REDIRECT = 302

# enum自带name和value属性
print(HttpStatus.REDIRECT)
print(HttpStatus.OK.name)
print(HttpStatus.NOT_FOUND.value)

# 枚举迭代
# 枚举支持顺序迭代和遍历
for status in HttpStatus:
    print(f"{status.name} : {status.value}")

'''
6、Python元类
'''

# 所有对象都是实例化或者调用类而得到的，Python中一切都是对象，通过class关键字定义的类本质也是对象
# 对象又是通过调用类得到的，因此通过class关键字定义的类肯定也是调用了一个类得到的，这个类就是元类。type就是Python内置的元类

