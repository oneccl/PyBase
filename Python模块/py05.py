
# 常用Python标准库（模块）

# 1、sys
import sys
"""
sys模块提供了与Python解释器和系统交互的功能。它包含了一些对解释器运行时环境的访问和操作
以及与系统交互的函数和变量；例如获取命令行参数，退出Python程序，获取输入输出相关内容
"""
# 常用功能：
'''
1）访问命令行参数（从程序外部获取参数）：sys.argv
2）获取Python解释器的版本信息：sys.version
3）设置递归调用的最大深度：sys.setrecursionlimit()
4）退出程序并返回指定的退出码：sys.exit()
'''
# 1）sys.argv
'''
执行Python脚本示例：E:\..\..demo.py  args1  args2  ...
            sys.argv[0]      sys.argv[1]  sys.argv[1] ...
'''
print(sys.argv[0])           # 获取程序本身路径
print(sys.argv[1])           # 从外部接收第一个参数
# 2）sys.path：动态的改变Python解释器搜索路径
'''
获取指定模块搜索路径的字符串列表(可以使用append()向里面添加第三方库或自定义模块路径)
'''
print(sys.path)              # 内置标准库模块列表
sys.path.append("文件路径")
# 3）sys.version
print(sys.version)           # 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)]
# 4）sys.exit(n)
'''
sys.exit(0)：程序正常退出；其他为异常退出，若需要中途退出时使用
'''
# 5）sys.platform：返回操作系统平台名称
print(sys.platform)          # win32
# 6）sys.setrecursionlimit()
print(sys.encursionlimit())  # 最大递归层数：1000
# 7）sys.stdin.readline()：标准输入
res = sys.stdin.readline()   # 获取控制台输入(会将输入全部获取，包括末尾的\n；input不会获取\n)
print(res)
# 8）sys.stdout.write("x")：标准输出
sys.stdout.write("SYS")      # 控制台输出：SYS

# 2、os与shutil
import os
"""
Python中的os和shutil是用于处理文件和目录操作的标准库
os模块提供了与操作系统交互的功能。它允许你访问文件系统、执行系统命令、管理进程和环境变量等
shutil库是os库的扩展，提供了更高级的文件和目录操作功能，如复制、移动、剪切、删除、压缩解压等
shutil模块对压缩包的处理是调用ZipFile和TarFile两个模块来进行的
glob模块是Python标准库模块之一，主要用来查找符合特定规则的目录和文件，返回到一个结果列表
"""
# os常用功能：
'''
1）获取当前工作目录：os.getcwd()
2）切换当前工作目录：os.chdir()
3）列出目录中的文件和子目录：os.listdir()
4）创建目录：os.mkdir()
5）执行系统命令：os.system()
6）获取环境变量的值：os.getenv()
'''
# 1）获取当前目录
# print(os.getcwd())
# 2）获取当前目录(或给定目录)下列表(文件名(带后缀)和目录名，不会递归)
# print(os.listdir())
# os.listdir('path')
# 3）获取路径的上一级路径
# print(os.path.dirname(os.getcwd()))
# 4）获取路径的最后一级的目录名或文件名(带后缀)
# print(os.path.basename(os.getcwd()))
# 5）获取文件名及其后缀组成的元组
# print(os.path.splitext('E:/A/a.txt'))       # ('E:/A/a', '.txt')
# 6）获取当前文件/目录的绝对路径
# print(os.path.abspath(os.getcwd()))         # E:/A/a
# 7）获取当前目录的绝对路径并加上给定str组成的新路径字符串
# print(os.path.abspath('./append'))          # E:/A/a/append
# 8）路径拼接
# print(os.path.join(os.getcwd(), 'append'))  # E:/A/a/append
# 9）判断给定路径是否存在
# os.path.exists('path')
# 10）判断给定路径是否为目录（文件夹）
# os.path.isdir('path')
# 11）判断给定路径是否为文件
# os.path.isfile('path')
# 12）获取指定路径的目录名或文件名(带后缀)
# print(os.path.split(os.getcwd()))           # ('E:/A/a', 'a')
# 13）获取文件大小
# os.path.getsize('文件')
# 14）文件重命名
# os.rename('原文件名', '新文件名')
# 15）创建文件
# Win系统不支持
# os.mknod('a.txt')
# Win下新建文件
# f = open('a.txt', 'w', encoding='utf-8')
# f.close()
# 16）删除文件
# os.remove('目标文件')
# 17）创建文件夹（目录）
# os.mkdir('目录')
# os.makedirs('递归目录')
# 18）删除文件夹（文件夹必须为空）
# os.rmdir('目录')
# 19）生成路径中每一级路径的迭代器
'''
a/
 ├── b/
 ├── c/
 │   └── c.txt
 └── a.txt
'''
# docs = os.walk('../../a')
# for i in docs:
#     print(i)
'''
('../../a', ['b', 'c'], ['a.txt'])
('../../a/b', [], [])
('../../a/c', [], ['c.txt'])
每一个元素都是一个三元组，第一个为路径，第二个为文件夹，第三个为文件
'''
# 20）其他
# os.sep                            # 获取操作系统特定的路径分隔符，Win为"\\"，Linux为"/"
# os.linesep                        # 获取当前平台使用的行终止符，Win为"\t\n"，Linux为"\n"
# os.pathsep                        # 获取用于分割文件路径的字符串，Win为";"，Linux为":"
# os.system("bash command")         # 运行当前系统shell命令，直接显示，返回值该命令的退出状态码，常见的退出状态码：0：命令成功执行；1：一般错误（非特定错误）；2：错误的shell命令语法；126：命令不可执行；127：未找到命令
# os.popen("bash command").read()   # 运行当前系统shell命令，获取执行结果
# 获取系统环境变量值：
os.getenv('key')                  # 返回指定名称的环境变量值，如果没有找到则返回None，不会报错
os.environ.get('key', 'default')  # 返回指定名称的环境变量值，如果没有找到则可以返回一个默认值，不指定默认值报错

# 案例：批量修改文件名
# path = 'path'
# file_list = os.listdir(path)
# p = re.compile(r'\..*')
# for count, file in enumerate(file_list):
#     old_name = path + file_list[count]
#     # 获取后缀
#     form = p.findall(old_name)[0]
#     new_name = path + str(count) + form
#     os.rename(old_name, new_name)

# 补充：os.path模块isfile()和isdir()的正确用法
# 在Python中，如果一个路径实际上不存在，那么无法直接通过os.path模块的isfile()或isdir()函数来判断这个路径是文件路径还是文件夹路径，此时两个函数都会返回False
# 已经存在的绝对路径对os.path模块的isfile()和isdir()函数才会真正起作用
# 实际不存在的路径如何判断文件还是目录：
def isfile(path: str):
    base = os.path.basename(path)
    suffix = os.path.splitext(base)[1]
    return True if suffix != '' and '.' in suffix else False

import shutil
# shutil常用功能：
'''
1）复制
复制文件到目录：shutil.copy(src, dst)
复制文件内容到目标文件（dst不存在则创建，存在则覆盖）：shutil.copyfile(src, dst)
复制文件夹：shutil.copytree(src, dst)
2）移动/剪切（可改名）：shutil.move(src, dst)
3）删除
递归删除文件夹：shutil.rmtree(dir)
4）压缩、解压
创建压缩文件（zip）：zipfile.write(file)
读取压缩包文件：zipfile.namelist(path)
解压压缩包单个文件：zipfile.extract(file)
解压到当前目录：zipfile.extractall(path)
'''
import glob
# glob库的4个通配符与3个函数：
'''
*   匹配0个或多个字符
**  递归匹配所有文件、目录
？  代匹配一个字符
[]  匹配指定范围内的字符，如[0-9]匹配数字、[a-z]匹配小写字母
'''
'''
glob()：返回所有符合匹配条件的文件路径列表
iglob()：返回所有符合匹配条件的文件路径迭代器
escape()：用于转义4个通配特殊字符
'''

# 3、collections
import collections
"""
collections模块提供了一些额外的数据类型，用于扩展Python内置的数据类型
"""
# 常用数据类型：
'''
1）namedtuple()：创建带有名称的元组
2）Counter：计数可哈希对象的出现次数
3）deque：双端队列，支持高效的添加和删除操作
4）defaultdict：带有默认值的字典
5）OrderedDict：有序字典，按照插入顺序保持键的顺序
'''
'''
1）namedtuple()函数是一个工厂函数，它返回一个子类，这个子类继承自tuple类，且拥有名字(第一个参数)
这个子类的实例就像一个普通的元组，但提供了方便的属性访问。namedtuple是一种定义小型和不可变的数据类的简单方法
'''
from collections import namedtuple
# 创建一个namedtuple类型User，并包含name和age两个属性
User = namedtuple('User', ['name', 'age'])
# 创建一个User对象
user = User(name='Tom', age=18)
print(user.name + " : " + str(user.age))   # Tom : 18
'''
2）deque（双向队列）是来自collections模块的容器，它提供了从左端和右端高效、快速地添加和删除元素的功能
'''
from collections import deque
# 创建一个deque
d = deque(['a', 'b', 'c'])
# 从右端添加元素
d.append('d')          # deque(['a', 'b', 'c', 'd'])
# 从左端添加元素
d.appendleft('e')      # deque(['e', 'a', 'b', 'c', 'd'])
# 从右端删除元素
d.pop()                # 返回'd'（deque(['e', 'a', 'b', 'c'])）
# 从左端删除元素
d.popleft()            # 返回'e'（deque(['a', 'b', 'c'])）
'''
3）Counter类是一个简单的计数器，例如它可以用来统计字符的个数；Counter对象有一个方法most_common(n)
该方法返回计数最多的n个元素的列表，每个元素是一个元组，元组的第一个元素是元素，第二个元素是元素的计数
'''
from collections import Counter
c = Counter('hello')   # 从一个可迭代对象创建
print(c)               # Counter({'l': 2, 'o': 1, 'h': 1, 'e': 1})
print(c.most_common(2))
'''
4）defaultdict是dict的一个子类，它接受一个工厂函数作为默认值，当查找的键不存在时，可以实例化一个值作为默认值
'''
from collections import defaultdict
# 使用列表(list)作为default_factory，当键不存在时，返回一个空列表
dd = defaultdict(list)
# 添加一个键值对
dd['dogs'].append('Rufus')
dd['dogs'].append('Kathrin')
dd['dogs'].append('Sniffles')
print(dd['dogs'])      # ['Rufus', 'Kathrin', 'Sniffles']
'''
5）OrderedDict是dict的一个子类，它记住了元素插入的顺序。在Python3.7之前，普通的dict并不保证键值对的顺序，而OrderedDict则按照插入的顺序排列元素
从Python3.7开始，dict也会保持插入顺序，但是OrderedDict仍然有它的特性，如重新排列字典的顺序等
'''
from collections import OrderedDict
d = OrderedDict()
d['first'] = 1
d['second'] = 2
d['third'] = 3
for key in d:
    print(key, d[key])
'''
first 1
second 2
third 3
'''

# 4、日期和时间：datetime、date、time
# 常用的类和函数：
'''
1）datetime类：表示日期和时间的对象，可以进行日期和时间的计算和比较
2）date类：表示日期的对象，可以获取日期的年、月、日等信息
3）time类：表示时间的对象，可以获取时间的小时、分钟、秒等信息
4）timedelta类：表示时间间隔的对象，可以进行时间的加减操作
5）strftime()函数：将日期和时间格式化为字符串
6）strptime()函数：将字符串解析为日期和时间对象
'''
# datetime
from datetime import datetime
"""
datetime模块提供了处理日期和时间的功能。它包含了一些类和函数，用于创建、操作和格式化日期和时间
"""
# 获取当前日期和时间（datetime）
current_datetime = datetime.now()
print(current_datetime)       # 2023-07-22 18:56:03.927568

# date
from datetime import date
# 获取当前日期
current_date = date.today()
print(current_date)           # 2023-07-22
# 创建一个表示生日的日期对象
birthday = date(1990, 2, 1)
# 计算年龄（返回天数）
age = date.today() - birthday
print(age)                    # 12224 days

# time
import time
print(time.time())            # 获取当前时间戳
time.sleep(0.5)               # 睡眠0.5秒（单位s）

# 格式化pd.date_range()生成的日期序列
import pandas as pd
dts = [datetime.strftime(dt, '%Y%m%d') for dt in pd.date_range('20210601', '20211201', freq='MS')]
print(dts)

# 格式化：
# 格式化日期为字符串类型：strftime()
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(datetime.today().strftime('%Y%m%d'))
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# 字符串转换为datetime类型：strptime()：
# 格式：%Y完整年份 %y去掉世纪的年份（00 - 99） %m月份（01 - 12） %M分钟数（00 - 59） %H小时（24小时制，00 - 23） %S秒（01 - 61）
datetime.strptime('2023-07-01 10:30:50', '%Y-%m-%d %H:%M:%S')
# 字符串转换为date类型：
# datetime.datetime(2023, 7, 1)

# ※ 其它常用用法：
from dateutil.relativedelta import relativedelta
# 获取当月1号
print(datetime.now().strftime('%Y%m01'))
# 获取上个月1号
print(datetime.strftime(datetime.now() - relativedelta(months=1), "%Y%m01"))
# 获取上n个月1号
# print(datetime.strftime(datetime.now() - relativedelta(months=n), "%Y%m01"))

import calendar
# 获取当月、上个月最后一天
cur_year = int(datetime.now().strftime("%Y"))
cur_month = int(datetime.now().strftime("%m"))
print("当前月份：", cur_month)
day_end = calendar.monthrange(cur_year, cur_month-1)[1]
print(f"{cur_year}0{cur_month-1}{day_end}")

# 获取上n个月最后一天
# end = calendar.monthrange(cur_year, cur_month-n)[1]
# print(f"{cur_year}0{cur_month-1}{day_end}")

# 5、re
import re
"""
re模块为高级字符串处理提供了正则表达式工具。对于复杂的匹配和处理，正则表达式提供了简洁、优化的解决方案
"""
# 1）re.search(pattern,string,flags=0)：返回字符串中搜索匹配正则表达式的match对象(第一个位置的值)
s = 'abc20de200'
print(re.search('[0-9]+', s).group(0))        # 20
# 2）re.match(pattern,string,flags=0)：返回目标字符串开始位置匹配正则表达式的match对象，未匹配成功返回None
print(re.match('\w', s).group(0))             # a
# 3）re.findall(pattern,string,flags=0)：以列表格式返回全部匹配到的所有字符串
print(re.findall('[0-9]+', s))                # ['20', '200']
# 4）re.split(pattern, string, maxsplit=0, flags=0)：将一个字符串按正则表达式匹配结果进行分割，返回一个列表；maxsplit参数表示最多进行分割次数
print(re.split('[0-9]+', s))                  # ['abc', 'de', '']
print(re.split('[0-9]+', s, maxsplit=1))      # ['abc', 'de200']
# 5）re.finditer(pattern,string,flags=0)：返回一个匹配结果的迭代器，每个迭代元素都是match对象
print(re.finditer('[0-9]+', s))
# 6）re.sub(pattern,repl,string,count=0,flags=0)：替换被正则表达式匹配到的字符串，返回替换后的字符串
print(re.sub('[0-9]+', '替换', s))             # abc替换de替换
# 补充：以上每个方法可换成如下示例（flag：匹配模式）
'''
pattern = re.compile(pattern, flags=0)
print(pattern.search(string))
'''

# 6、math
import math
"""
math数学模块为浮点运算提供了对底层C函数库的访问
"""
# 1）数学常量
print(math.pi)                  # 圆周率π
print(math.e)                   # 自然常数e
print(math.inf)                 # 无穷大（浮点型）
print(math.nan)                 # 非数值
# 2）通用函数
print(math.ceil(5.2))           # 向上取整
print(math.floor(5.8))          # 向下取整
print(math.fabs(-3.5))          # 绝对值
print(math.fmod(5, 2))          # x%y余数（浮点型）
print(math.gcd(16, 20))         # 求最大公约数
print(math.isinf(math.inf))     # 若x是无穷大，则返回True，否则返回False
print(math.isfinite(math.e))    # 若x是有限数值，则返回True，否则返回False
print(math.isnan(math.nan))     # 若x是NaN，则返回True，否则返回False
print(math.modf(3.1415))        # 返回x的小数和整数部分，以元组形式显示（都是浮点型）
print(math.factorial(5))        # 返回x的阶乘
# 3）幂函数与对数
print(math.pow(2, 3))           # x的y次幂
print(math.sqrt(9))             # x的平方根（开方）
print(math.exp(2))              # e的x次幂
print(math.log(2))              # x的对数
# 4）三角函数
print(math.sin(30))             # x弧度的正弦值
print(math.cos(30))             # x弧度的余弦值
print(math.tan(30))             # x弧度的正切值
print(math.asin(0.5))           # 以弧度为单位返回x的反正弦值
print(math.acos(0.5))           # 以弧度为单位返回x的反余弦值
print(math.atan(0.5))           # 以弧度为单位返回x的反正切值
# 弧度与角度转换
math.degrees(0.5)               # 弧度转角度
math.radians(30)                # 角度转弧度

# 7、random
"""
random提供了生成随机数的工具
"""
# 1）random随机数
import random

print(random.random())          # (0, 1)随机浮点数
random.randint(1, 10)           # [1, 10]随机整数
random.uniform(1, 10)           # [1, 10]均匀分布随机数（浮点型）
random.gauss(5, 1)              # 生成正态分布随机数，均值5，标准差1
random.expovariate(0.2)         # 生成指数分布随机数，均值5
random.shuffle([1, 3, 5, 7])    # 将序列中的元素顺序打乱
random.choice([1, 3, 5, 7])     # 从序列中随机选取一个元素
random.randrange(1, 10, 2)      # [1, 10]间隔为2的随机整数
random.sample([1, 3, 5, 7], 2)  # 从序列中随机抽取n个数据（列表类型返回）

# 2）numpy随机数
import numpy as np

np.random.randint(1, 10, [2, 3])      # [1, 10]随机整数，2行*3列
np.random.uniform(1, 10, [2, 3])      # [1, 10]均匀分布随机数，2行*3列
np.random.normal(5, 1, [2, 3])        # 生成正态分布随机数，均值5，标准差1，2行*3列
np.random.exponential(0.2, [2, 3])    # 生成指数分布随机数，均值5，2行*3列
np.random.poisson(5, [2, 3])          # 生成泊松分布随机数，均值5，2行*3列

# 8、json
import json

# A、json.dumps() 对象->Json（序列化）
# json.dumps()会默认将中文转化成ASCII编码格式，一般需要手动设置ensure_ascii=False
# 1）将Python对象转化为Json对象
data = {"k1": "v1", "k2": "v2"}
print(json.dumps(data))    # {"k1": "v1", "k2": "v2"}

# B、json.loads() Json->对象（反序列化）
# 2）访问Json中的k对应的值
json_data = '''{"k1": "v1", "k2": "v2"}'''
data = json.loads(json_data)
print(data['k2'])          # v2
# 3）访问Json中的嵌套k对应的值
json_data = '''
{
    "k1": "v1",
    "k2": ["v2", "v3"],
    "k3": {"k4": "V5", "k5": ["v6", "v7", "v8"]}
}
'''
data = json.loads(json_data)
print(data["k3"]["k5"])    # ['v6', 'v7', 'v8']

# A、json.dumps() 对象->Json（序列化）
# # json.dumps()会默认将中文转化成ASCII编码格式，一般需要手动设置ensure_ascii=False
# 4）将类对象转化为Json
from json import JSONEncoder

class Stu:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class StuEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

stu = Stu('Tom', 18)
print(json.dumps(stu, indent=4, cls=StuEncoder))
'''
{
    "name": "Tom",
    "age": 18
}
'''

# B、json.loads() Json->对象（反序列化）
# 5）将Json转化为类对象
def stuDecoder(obj):
    return Stu(obj['name'], obj['age'])

stu = json.loads('{"name": "Tom", "age": 18}', object_hook=stuDecoder)
print(stu)
'''
<__main__.Stu object at 0x000001A44F034A00>
'''

# 6）获取Json中指定k的所有值
json_data = '''
[
   {
      "id": 1,
      "color":["red", "green"]
   },
   {
      "id": 2,
      "color": "yellow"
   }
]
'''
data = []
try:
    data = json.loads(json_data)
except Exception as e:
    print(e)

ls = [item.get('id') for item in data]
print(ls)
'''
[1, 2]
'''

# 补充：
'''
json.load(file)：读取文件中的json字符串，转换成对象
json.dump(obj,file)：将对象转换成json字符串，并写入到文件中
'''

# 9、logging
# logging模块是Python内置的标准模块，主要用于输出运行日志
import logging

# 日志事件级别（级别从小到大）
'''
DEBUG、INFO、WARNING(默认)、ERROR
'''
# 配置日志级别和格式
'''
logging.basicConfig(level,filename,filemode,format,datefmt)
 - level：日志输出级别，只会追踪该级别及以上的事件
 - filename：日志输出的文件名
 - filemode：文件写入模式，默认是追加，w覆盖
 - format：日志输出的格式
 - datefmt：日期格式，如%Y-%m-%d %H:%M:%S
'''
# 基本使用：
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s Line: %(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.debug("This is DEBUG！")
logging.info("This is INFO！")
logging.warning("This is WARNING！")
logging.error("This is ERROR！")
'''
2023-09-09 16:45:21 root INFO Line: 17 This is INFO！
2023-09-09 16:45:21 root WARNING Line: 18 This is WARNING！
2023-09-09 16:45:21 root ERROR Line: 19 This is ERROR！
'''
# 将logger的级别改为ERROR，结果：
'''
2023-09-09 16:46:24 root ERROR Line: 19 This is ERROR！
'''

# 高级用法
# logging模块四大组件
'''
日志器（Logger）：暴露函数给应用程序，基于日志记录器和过滤器级别决定哪些日志有效
处理器（Handler）：将logger创建的日志记录发送到合适的目的输出
过滤器（Filter）：提供了更细粒度的控制工具来决定输出哪些日志记录
格式器（Formatter）：决定日志记录的最终输出格式
'''
# 日志器（Logger）
# 创建、实例化
# logger_name：日志记录的名称，对应配置文件和打印日志格式中的%(name)s，如果不指定则返回root对象
# logger = logging.getLogger('logger_name')
# 设置日志级别：只有日志级别大于等于INFO的日志才会输出
# logger.setLevel(logging.INFO)
# 添加处理器
# logger.addHandler(handler_name)
# 删除处理器
# logger.removeHandler(handler_name)

# 处理器（Handler）
# Handler常用处理器类型：StreamHandler、FileHandler、NullHandler
# 创建
# handler = logging.StreamHandler(stream=None)
# 设置日志级别
# handler.setLevel(logging.WARN)
# 设置格式化器
# handler.setFormatter(formatter_name)
# 添加过滤器（可添加多个）
# handler.addFilter(filter_name)
# 删除过滤器
# handler.removeFilter(filter_name)

# 过滤器（Filter）
# Handler和Logger可以使用Filter来完成比级别更复杂的过滤
# 创建：name=''表示所有事件都接受
# filter = logging.Filter(name='')

# 格式器（Formatter）
# 使用Formatter对象可以设置日志信息内容最后的输出规则、结构
# 创建: fmt默认使用%(message)s datefmt默认使用asctime，格式：%Y-%m-%d %H:%M:%S
# formatter = logging.Formatter(fmt=None, datefmt=None)
'''
%(asctime)s	          日志的时间
%(levelname)s	      日志级别名称
%(name)s              日志记录的名称
%(filename)s	      当前执行程序名称
%(funcName)s	      日志的当前函数
%(lineno)d	          日志的当前行号
%(message)s	          日志信息
'''

# 配置日志
# 1）代码显示配置
logger = logging.getLogger('example')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
ch.setFormatter(fmt)
logger.addHandler(ch)

# 2）使用conf配置文件
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('example')

# logging.conf内容
'''
[loggers]
keys=root,example

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[logger_example]
level=DEBUG
handlers=consoleHandler
qualname=example
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format='%(asctime)s %(name)s %(levelname)s %(message)s'
datefmt='%Y-%m-%d %H:%M:%S'
'''

# 10、pickle模块
# pickle（二进制协议）模块主要用于Python对象的序列化和反序列化
# pickle序列化优缺点
'''
优点：接口简单，各平台通用，支持数据类型广泛，扩展性强
缺点：不保证操作的原子性，存在安全问题，不同语言间不兼容，可使用json模块序列化
'''
# JSON是一种文本序列化格式（输出Unicode文本）；pickle是二进制序列化格式
import pickle

# 1）序列化
# pickle.dump(obj, file)：将序列化后的对象obj以二进制形式写入文件file中
# pickle.dumps(obj)：直接返回一个序列化的bytes对象

# 2）反序列化
# pickle.load(file)：将序列化的对象从文件file中读取出来
# pickle.loads(bytes_obj)：直接从bytes对象中读取序列化的对象


# 11、CSV模块

# CSV（Comma-Separated Values）是一种以逗号分隔值的文件类型，CSV以纯文本的形式存储表格数据(数字和文本)
# CSV本质是字符序列，文件的每一行代表一条数据，每条记录包含由逗号分隔的一个或多个属性值

# CSV与Excel的主要区别
"""
CSV                Excel
纯文本文件          二进制文件
消耗内存小          消耗内存较多
"""
# 使用注意：
'''
1）读写默认使用逗号作为分隔符(delimiter)、使用双引号作为引用符(quotechar)
2）数据写入时默认None转换为空字符串、浮点型使用repr()转化为字符串、非字符串数据使用str()转换为字符串存储
3）如果newline=''没有指定，嵌入引用字段内换行符将不会被正确地解释
'''

# 1、基本使用
import csv

# 1）写入：writer(csvfile)
# 若csvfile是文件对象，则必须指定newline=''，否则结果每写一行会空一行
with open(r'C:\Users\cc\Desktop\test.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # 写入一条数据：writerow()
    writer.writerow(['id', 'name', 'age'])
    # 批量写入：writerows()
    data = [
        ('100', 'Tom', 18),
        ('101', 'Jerry', 17)
    ]
    writer.writerows(data)

# 2）读取：reader(csvfile)
with open(r'C:\Users\cc\Desktop\test.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    res = [row for row in reader]
    print(res)

# 获取CSV文件表头及其索引
# 获取CSV文件表头及其索引
with open(r'C:\Users\cc\Desktop\test.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # 读取文件第一行表头（跳过首行）
    head = next(reader)
    # 字段名
    for index, col in enumerate(head):
        print(index, col)
    # 数据
    for row in reader:
        id, name, age = row
        print(row)


# 2、字典格式数据读写

# 1）写入
with open(r'C:\Users\cc\Desktop\test_d.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # 定义表头
    fields = ['id', 'name', 'age']
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    # 写入一条数据
    writer.writerow({'id': '102', 'name': 'Bob', 'age': 20})
    # 批量写入
    data = [
        {'id': '103', 'name': 'A', 'age': 19},
        {'id': '104', 'name': 'B', 'age': 21}
    ]
    writer.writerows(data)

# 2）读取
with open(r'C:\Users\cc\Desktop\test_d.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    res = [row for row in reader]
    print(res)

# 3、CSV追加写入数据
# 使用mode='a'：追加模式
with open(r'C:\Users\cc\Desktop\test.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    data = [
        ('102', 'Bob', 20)
    ]
    writer.writerows(data)


