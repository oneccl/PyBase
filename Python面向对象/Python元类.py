

# Python元类
# 所有对象都是实例化或者调用类而得到的，Python中一切都是对象，通过class关键字定义的类本质也是对象，对象又是通过调用类得到的，因此通过class关键字定义的类肯定也是调用了一个类得到的，这个类就是元类。type就是Python内置的元类

# 参考文章：
# https://www.cnblogs.com/JetpropelledSnake/p/9094103.html
# https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python

# 1、类也是对象

# 在理解元类之前，你需要先掌握Python中的类。Python中类的概念借鉴于Smalltalk语言，这显得有些奇特。在大多数编程语言中，类就是一组用来描述如何生成一个对象的代码段。在Python中这一点仍然成立：
class ObjectCreator(object):
    pass

obj = ObjectCreator()
print(obj)               # <__main__.ObjectCreator object at 0x0000021098A4AFB0>

# 但是，Python中的类还远不止如此。类同样也是一种对象。是的，没错，就是对象。只要你使用关键字class，Python解释器在执行的时候就会创建一个对象
# 例如，上面class代码段，将在内存中创建对象ObjectCreator。这个对象(类)自身拥有创建对象(类实例)的能力，而这就是它为什么是类也是对象的原因
# 但是，它本质上仍然是一个对象，于是乎你可以对它做如下操作：
'''
1）你可以将它赋值给一个变量
2）你可以拷贝它
3）你可以为它增加属性
4）你可以将它作为函数参数进行传递
'''
# 下面是示例：
# 你可以打印一个类，因为它就是一个对象
print(ObjectCreator)    # <class '__main__.ObjectCreator'>
# 你可以将类作为参数传给函数
def echo(o):
    print(o)

echo(ObjectCreator)     # <class '__main__.ObjectCreator'>
# 你可以为类增加属性
ObjectCreator.field = 'value'
print(hasattr(ObjectCreator, 'field'))    # True
print(ObjectCreator.field)                # value
# 你可以将类复制给一个变量
var = ObjectCreator
print(var())    # <__main__.ObjectCreator object at 0x0000026AF00EABF0>

# 动态地创建类
# 因为类也是对象，你可以在运行时动态的创建它们，就像其他任何对象一样。首先，你可以在函数中创建类，使用class关键字即可
def choose_class(name):
    match name:
        case 'stu':
            class Stu(object):
                pass
            return Stu
        case 'emp':
            class Emp(object):
                pass
            return Emp

stu = choose_class('stu')
# 返回类，而不是类的实例
print(stu)      # <class '__main__.choose_class.<locals>.Stu'>
# 可以通过这个类创建类的实例（类对象）
print(stu())    # <__main__.choose_class.<locals>.Stu object at 0x000001D7E6D9AAA0>

# 但这还不够动态，因为你仍然需要自己编写整个类的代码。由于类也是对象，所以它们应该也是通过什么东西来生成的才对。当你使用class关键字时，Python解释器自动创建这个对象。但就和Python中的大多数事情一样，Python仍然提供给你手动处理的方法
# 还记得内建函数type()吗？这个古老但强大的函数能够让你知道一个对象的类型是什么，就像这样：
print(type(0))                  # <class 'int'>
print(type('0'))                # <class 'str'>
print(type(ObjectCreator))      # <class 'type'>
print(type(ObjectCreator()))    # <class '__main__.ObjectCreator'>

# 这里，type有一种完全不同的能力，它也能动态的创建类。type可以接受一个类的描述作为参数，然后返回一个类
# 我知道，根据传入参数的不同，同一个函数拥有两种完全不同的用法是一件很愚蠢的事情，但这在Python中是为了保持向后兼容性
# type可以像这样工作：
'''
type(name, bases, attrs)
name：类的名称  bases：父类，用于继承，元组类型，可为空  attrs：包含属性名称和属性值的字典
'''
# 比如下面的代码：
class MyShinyClass(object):
    pass

# 可以手动像这样创建：
MyShinyClass = type('MyShinyClass', (), {})
# 返回类对象
print(MyShinyClass)      # <class '__main__.MyShinyClass'>
# 创建该类的实例
print(MyShinyClass())    # <__main__.MyShinyClass object at 0x0000018E2F0FAAA0>
# 你会发现我们使用MyShinyClass作为类名，并且也可以把它当做一个变量来作为类的引用。类和变量是不同的，这里没有任何理由把事情弄的复杂
# type接受一个字典来为类定义属性，因此：
class Foo(object):
    flag = True

# 可以翻译为：
Foo = type('Foo', (), {'flag': True})
# 并且可以将Foo当成一个普通的类一样使用：
print(Foo)         # <class '__main__.Foo'>
print(Foo.flag)    # True
foo = Foo()
print(foo)         # <__main__.Foo object at 0x00000203B4A4AA70>
print(foo.flag)    # True

# 当然，你可以向这个类继承：
class FooChild(Foo):
    pass

# 就可以写成：
FooChild = type('FooChild', (Foo,), {})
print(FooChild)         # <class '__main__.FooChild'>
# flag属性是由继承而来的
print(FooChild.flag)    # True

# 最终你会希望为你的类增加方法。只需要定义一个有着恰当签名的函数并将其作为属性赋值就可以了：
def echo_flag(self):
    print(self.flag)

FooChild = type('FooChild', (Foo,), {'echo_flag': echo_flag})
print(hasattr(Foo, 'echo_flag'))         # False
print(hasattr(FooChild, 'echo_flag'))    # True
child = FooChild()
child.echo_flag()                        # True
# 你可以看到，在Python中，类也是对象，你可以动态的创建类。这就是当你使用关键字class时Python在幕后做的事情，而这就是通过元类来实现的

# 2、什么是元类
# 元类就是用来创建类的东西。你创建类就是为了创建类的实例对象，不是吗？但是我们已经知道Python中的类也是对象。好吧，元类就是用来创建这些类（对象）的，元类就是类的类，你可以这样理解：
'''
MyClass = MetaClass()
MyObject = MyClass()
'''
# 你已经看到了type可以让你像这样做：
# MyClass = type('MyClass', (), {})

# 这是因为函数type实际上是一个元类。type就是Python在背后用来创建所有类的元类。现在你想知道那为什么type会全部采用小写形式而不是Type呢？好吧，我猜这是为了和str保持一致性，str是用来创建字符串对象的类，而int是用来创建整数对象的类
# type就是创建类对象的类。你可以通过检查__class__属性来看到这一点。Python中所有的东西，注意，我是指所有的东西——都是对象。这包括整数、字符串、函数以及类。它们全部都是对象，而且它们都是从一个类创建而来
age = 18
print(age.__class__)       # <class 'int'>
name = 'Tom'
print(name.__class__)      # <class 'str'>
def method(): pass
print(method.__class__)    # <class 'function'>
class Bar(object): pass
bar = Bar()
print(bar.__class__)       # <class '__main__.Bar'>

# 那么，对于任何一个__class__的__class__属性又是什么呢？
print(age.__class__.__class__)       # <class 'type'>
print(name.__class__.__class__)      # <class 'type'>
print(method.__class__.__class__)    # <class 'type'>
print(bar.__class__.__class__)       # <class 'type'>

# 因此，元类就是创建类这种对象的东西。如果你喜欢的话，可以把元类称为类工厂（不要和工厂类搞混了），type就是Python的内建元类，当然了，你也可以创建自己的元类

# 3、__metaclass__属性
# 你可以在写一个类的时候为其添加__metaclass__属性
# class Foo(object):
#     __metaclass__ = something

# 如果你这么做了，Python就会用元类来创建类Foo。小心点，这里面有些技巧。你首先写下class Foo(object)，但是类对象Foo还没有在内存中创建
# Python会在类的定义中寻找__metaclass__属性，如果找到了，Python就会用它来创建类Foo，如果没有找到，就会用内建的type来创建这个类
class Foo(Bar):
    pass
# 当你写该代码时，Python做了如下的操作：
# Foo中有__metaclass__这个属性吗？如果是，Python会在内存中通过__metaclass__创建一个名字为Foo的类对象。如果Python没有找到__metaclass__，它会继续在Bar（父类）中寻找__metaclass__属性，并尝试做和前面同样的操作。如果Python在任何父类中都找不到__metaclass__，它就会在模块层次中去寻找__metaclass__，并尝试做同样的操作，如果还是找不到__metaclass__，Python就会用内置的type来创建这个类对象

# 现在的问题就是，你可以在__metaclass__中放置些什么代码呢？答案就是：可以创建一个类的东西。那么什么可以用来创建一个类呢？type或任何使用到type或子类化type的东西都可以

# Python3中的元类
# 在Python3中，设置元类的语法已经更改：
# class Foo(object, metaclass=something):
#     pass
# 即不再使用metaclass属性，而是在基类列表中使用__metaclass__关键字参数。然而，元类的行为基本保持不变

# 4、自定义元类
# 元类的主要目的就是为了当创建类时能够自动地改变类。通常，你会为API做这样的事情，你希望可以创建符合当前上下文的类。假想一个很愚蠢的例子，你决定在你的模块里所有的类的属性都应该是大写形式
# 有好几种方法可以办到，但其中一种就是通过在模块级别设定__metaclass__。采用这种方法，这个模块中的所有类都会通过这个元类来创建，我们只需要告诉元类把所有的属性都改成大写形式就万事大吉了
# 幸运的是，__metaclass__实际上可以被任意调用，它并不需要是一个正式的类。所以，我们这里就先以一个简单的函数作为例子开始

# 元类会自动将你通常传给type的参数作为自己的参数传入
def upper_attr(future_class_name, future_class_parents, future_class_attrs):
    """返回一个类对象，将属性全部转为大写形式"""
    # 选择所有不以'__'开头的属性（私有属性），将它们转为大写形式
    uppercase_attrs = {
        attr if attr.startswith("__") else attr.upper(): v
        for attr, v in future_class_attrs.items()
    }
    # 通过type来做类对象的创建
    return type(future_class_name, future_class_parents, uppercase_attrs)

# 这会作用到这个模块中的所有类
__metaclass__ = upper_attr

# 需要注意的是，全局__metaclass__将不能与object一起工作
class Foo():
    # 我们也可以只在这里定义__metaclass__，这样就只会作用于这个类中
    # __metaclass__ = upper_attr
    bar = 'bip'

print(hasattr(Foo, 'bar'))    # True
print(hasattr(Foo, 'BAR'))    # False
foo = Foo()
print(foo.bar)                # bip
# 此处存在问题，结果与预期相反，原因未知，有人知道什么原因吗

# 现在，让我们做完全相同的事情，但对元类使用一个真正的类：
# 请记住，type实际上是一个类，就像str和int一样，所以，你可以从type继承
class UpperAttrMetaClass(type):
    # __new__是在__init__之前被调用的特殊方法，是用来创建对象并返回的方法
    # 而__init__只是用来将传入的参数初始化给对象，你很少用到__new__，除非你希望能够控制对象的创建
    # 这里，创建的对象是类，我们希望能够自定义它，所以我们这里改写__new__
    # 如果你希望的话，你也可以在__init__中做些事情，还有一些高级的用法会涉及到改写__call__特殊方法，但是我们这里不用
    def __new__(upperattr_metaclass, future_class_name, future_class_parents, future_class_attrs):
        uppercase_attrs = {
            attr if attr.startswith("__") else attr.upper(): v
            for attr, v in future_class_attrs.items()
        }
        return type(upperattr_metaclass, future_class_name, future_class_parents, uppercase_attrs)

# 但是，这种方式其实不是OOP。我们直接调用了type，而且我们没有改写父类的__new__方法。现在让我们这样去处理：
class UpperAttrMetaClass(type):
    def __new__(upperattr_metaclass, future_class_name, future_class_parents, future_class_attrs):
        uppercase_attrs = {
            attr if attr.startswith("__") else attr.upper(): v
            for attr, v in future_class_attrs.items()
        }
        return type.__new__(upperattr_metaclass, future_class_name, future_class_parents, uppercase_attrs)

# 你可能已经注意到了有个额外的参数upperattr_metaclass，这并没有什么特别的。类方法的第一个参数总是表示当前的实例，就像在普通的类方法中的self参数一样
# 当然了，为了清晰起见，这里的名字我起的比较长。但是就像self一样，所有的参数都有它们的传统名称。因此，在真实的产品代码中一个元类应该是像这样的：
class UpperAttrMetaclass(type):
    def __new__(cls, clsname, bases, attrs):
        uppercase_attrs = {
            attr if attr.startswith("__") else attr.upper(): v
            for attr, v in attrs.items()
        }
        return type.__new__(cls, clsname, bases, uppercase_attrs)

# 如果使用super方法的话，我们还可以使它变得更清晰一些，这会简化继承（你可以拥有元类，从元类继承，从type继承）
class UpperAttrMetaclass(type):
    def __new__(cls, clsname, bases, attrs):
        uppercase_attrs = {
            attr if attr.startswith("__") else attr.upper(): v
            for attr, v in attrs.items()
        }
        return super(UpperAttrMetaclass, cls).__new__(cls, clsname, bases, uppercase_attrs)

class Foo(object, metaclass=UpperAttrMetaclass):
    bar = 'bip'

print(hasattr(Foo, 'bar'))    # False
print(hasattr(Foo, 'BAR'))    # True
foo = Foo()
print(foo.BAR)                # bip

# 在Python3中，如果你使用关键字参数进行调用，如下所示：
# class Foo(object, metaclass=MyMetaclass, kwargs=default):
#     pass
# 它在元类中可转换为：
# class MyMetaclass(type):
#     def __new__(cls, clsname, bases, dct, kwargs=default):
#         pass

# 就是这样，除此之外，关于元类真的没有别的可说的了。使用到元类的代码比较复杂，这背后的原因倒并不是因为元类本身，而是因为你通常会使用元类去做一些晦涩的事情，依赖于自省，控制继承等
# 确实，用元类来搞些“黑暗魔法”是特别有用的，因而会搞出些复杂的东西来。但就元类本身而言，它们其实是很简单的：
# 1）拦截类的创建
# 2）修改类
# 3）返回修改之后的类

# 5、为什么要用metaclass类而不是函数?
# 由于__metaclass__可以接受任何可调用的对象，那为何还要使用类呢，因为很显然使用类会更加复杂啊！这样做有以下几个原因：
'''
1）意图会更加清晰。当你读到UpperAttrMetaclass(type)时，你知道接下来要发生什么
2）你可以使用OOP编程。元类可以从元类中继承而来，改写父类的方法。元类甚至还可以使用元类
3）你可以把代码组织的更好。当你使用元类的时候肯定不会是像我上面举的这种简单场景，通常都是针对比较复杂的问题。将多个方法归总到一个类中会很有帮助，也会使得代码更容易阅读
4）你可以使用__new__、__init__以及__call__这样的特殊方法。它们能帮你处理不同的任务。就算通常你可以把所有的东西都在__new__里处理掉，有些人还是觉得用__init__更舒服些
5）哇哦，这东西的名字是metaclass，肯定非善类，我要小心！
'''

# 6、究竟为什么要使用元类？
# 现在回到我们的主题上来，究竟是为什么你会去使用这样一种容易出错且晦涩的特性？好吧，一般来说，你根本就用不上它
# Python界的领袖Tim Peters说：
# 元类就是深度的魔法，99%的用户应该根本不必为此操心。如果你想搞清楚究竟是否需要用到元类，那么你就不需要它。那些实际用到元类的人都非常清楚地知道他们需要做什么，而且根本不需要解释为什么要用元类

# 元类的主要用途是创建API。一个典型的例子是Django ORM。它允许你像这样定义：
# class Person(models.Model):
#     name = models.CharField(max_length=30)
#     age = models.IntegerField()

# 但是，如果你这样做：
# person = Person(name='Tom', age='18')
# print(person.age)

# 这并不会返回一个IntegerField对象，而是会返回一个int，甚至可以直接从数据库中取出数据。这是有可能的，因为models.Model定义了__metaclass__， 并且使用了一些魔法能够将你刚刚定义的简单的Person类转变成对数据库的一个复杂hook。Django框架将这些看起来很复杂的东西通过暴露出一个简单的使用元类的API将其化简，通过这个API重新创建代码，在背后完成真正的工作

# 7、结语
# 首先，你知道了类其实是能够创建出类实例的对象。好吧，事实上，类本身也是实例，当然，它们是元类的实例
# Python中的一切都是对象，它们要么是类的实例，要么是元类的实例，除了type
# type实际上是它自己的元类，在纯Python环境中这可不是你能够做到的，这是通过在实现层面做一些手段实现的
# 其次，元类是很复杂的。对于非常简单的类，你可能不希望通过使用元类来对类做修改。你可以通过其他两种技术来修改类：
# monkey patching（猴子打补丁）
# class decorators（类装饰器）
# 当你需要动态修改类时，99%的时间里你最好使用上面这两种技术。当然了，其实在99%的时间里你根本就不需要动态修改类


