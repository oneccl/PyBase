
# 文件I/O、Python异常处理、文件压缩、StringIO与BytesIO

# A、文件I/O

# 1、文件访问：
# open('文件路径',mode='操作模式',encoding='编码方式')
"""
只读：r
只写：w
只追加：a
可读可追加：a+
读写：r+/w+
以二进制格式打开一个文件只读：rb
以二进制格式打开一个文件只写：wb
以二进制格式打开一个文件读写：rb+/wb+
"""
# Python内置的open()函数用于打开一个文件(创建文件)，创建一个file对象
# file = open("foo.txt", 'r+')
file = open("foo.txt", 'w')
# file对象的属性
print("文件名："+file.name)
# print("文件是否已关闭："+file.closed)
# print("文件访问模式："+file.mode)

# 2、文件写入
# file.write("www.site.com!\nVery good site!\n")
file.flush()

# 3、文件读取
# file.read([以字节计数，若未传入，则读取全部])
# res = file.read()
# print(res)
# 读取一行
# line = file.readline()
# 读取所有行，返回列表
# lines = file.readlines()

# 4、文件关闭
file.close()
del file      # 回收应用程序级的变量

# OS文件/目录方法
# 重命名和删除文件：需要导入os模块
import os
os.rename("foo.txt", "new_foo.txt")
os.remove("new_foo.txt")

print(os.path.exists('path'))     # 文件是否存在
print(os.path.isdir('path'))      # 是否是文件夹/目录
print(os.path.isfile('path'))     # 是否是文件
print(os.path.abspath('path'))    # 返回绝对路径

os.mkdir('path')                  # 创建目录
os.makedirs('p1'/'p2'/'...')      # 创建多级目录
os.rmdir('path')                  # 删除目录
os.remove('path')                 # 删除文件
os.rename('src', 'dest')          # 重命名文件或目录

print(os.path.getsize('path'))    # 获取文件大小（字节）
print(os.listdir('path'))         # 获取指定目录下全部

# with上下文管理
# with open() as写法：with会自动执行file.close()
"""
with open('path', 'mode', encoding='utf-8') as file:
    contents = file.read()        # 读取
    print(contents)
    file.write(contents)          # 写入
"""

# B、Python异常处理

'''
异常处理：当Python脚本发生异常时我们需要捕获处理它，否则程序会终止执行
异常捕获：try...except...else语句
try:
   语句
except 异常名:
   try中发生异常进行处理
else:
   没有异常发生时执行的代码
'''
try:
    fh = open("textfile.txt", "w")
    fh.write("测试异常")
except IOError:
    print("没有找到文件或读取文件失败")
else:
    print("文件写入成功")
    fh.close()

# try...except...finally语句：无论是否发生异常都将执行finally代码
try:
    fh = open("textfile.txt", "w")
    fh.write("测试异常")
except IOError:
    print("没有找到文件或读取文件失败")
finally:
    fh.close()


# 自定义异常：
# 继承基类Exception或子类RuntimeError等
class BaseError(Exception):
    # __init__构造方法用于接收一些参数来设置异常信息，例如错误码、错误消息等
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg

    # 重写__str__方法用于返回异常的描述信息，相当于Java的toString()方法
    def __str__(self):
        return f"{self.code}: {self.msg}"

# 案例eg1：
try:
    # raise语句用于手动触发异常；变量e用于创建类的实例
    raise BaseError(400, 'Raise is used to test Exception')
except BaseError as e:
    print(e)         # 400: Raise is used to test Exception

# 案例eg2：主动抛出异常（主动抛出的异常同样也会导致程序的终止）
# if not x 用法：用于判断x是否等于None、False、""、0、空列表[]、空字典{}、空元祖()
ls_t = []
try:
    if not ls_t:
        raise Exception("The List is Empty.")
except Exception as e:
    print(e)        # The List is Empty.

# 使用traceback获取异常堆栈信息
import traceback

# print_exc()函数用于打印异常堆栈的详细信息
traceback.print_exc()
# format_exc()函数用于返回异常堆栈信息的字符串
print(traceback.format_exc())

# C、文件压缩、解压缩
import os
import zipfile

SRC_DIR_PATH = r'..\..\目录名'
DST_DIR_PATH = r'..\..\目录名.zip'

# 1、压缩文件夹（目录）为.zip格式
zip = zipfile.ZipFile(DST_DIR_PATH, "w", zipfile.ZIP_DEFLATED)
for dir_path, dir_names, file_names in os.walk(SRC_DIR_PATH):
    # print(dir_path)         # 源文件夹（目录）路径，子文件夹（子目录）会递归遍历
    # print(dir_names)        # 子文件夹（子目录），以list形式显示
    # print(file_names)       # 文件夹（目录）下的所有文件，以list形式显示，子文件夹（子目录）会递归遍历
    # 去掉 ..\..\目录名 层级路径
    fpath = dir_path.replace(SRC_DIR_PATH, '')
    for filename in file_names:
        zip.write(os.path.join(dir_path, filename), os.path.join(fpath, filename))
zip.close()

# 2、解压缩.zip格式文件夹（目录）到当前目录
zip_ref = zipfile.ZipFile(DST_DIR_PATH)
os.mkdir(DST_DIR_PATH.replace(".zip", ""))
# 解压zip文件到指定文件夹
zip_ref.extractall(DST_DIR_PATH.replace(".zip", ""))
zip_ref.close()

# D、StringIO与BytesIO

# 内存与硬盘：内存读写速度快，硬盘读写速度慢
# StringIO与BytesIO作用：在内存中虚拟一个文件，该虚拟文件允许你像操作硬盘中的文件一样操作数据，并不会降低读写速度

# 1）StringIO：写入字符串
from io import StringIO

# 写入数据到内存
io = StringIO('data')    # 方式1
io.write('data')         # 方式2
# 从内存读取数据
print(io.read())         # 方式1
print(io.getvalue())     # 方式2

# 按行输出写入内存的数据
for line in io.readlines():
    # 去掉每行末尾\n
    print(line.strip('\n'))

# 2）BytesIO：写入字节流
from io import BytesIO
# 操作与StringIO类似

