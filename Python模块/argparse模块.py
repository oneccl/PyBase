
# argparse模块

# argparse是用于解析命令行参数和选项的Python标准模块，取代了已弃用的optparse模块。argparse用于解析命令行参数
# 很多时候，我们需要使用解析命令行参数的程序
# 通过argparse模块，可以轻松编写用户友好的命令行界面。argparse可以解析sys.argv()中定义的参数
# argparse模块还会自动生成帮助和使用消息，并在用户为程序提供无效参数时发出错误

# argparse模块使用三部曲：
# 1）使用ArgumentParser创建一个解析器
# parser = argparse.ArgumentParser()
# 2）使用add_argument()添加将要关注的命令行参数和选项到对象中，每个add_argument()方法对应一个要关注的参数或选项，参数可以是可选的、必需的或定位的
# parser.add_argument()
# 3）调用parse_args()方法进行解析，解析成功后，即可使用
# parser.parse_args()

# 基本使用：
import argparse

# 1）argparse可选参数

# description：这些参数在调用parser.print_help()时或由于参数不正确而运行程序时会打印这个描述信息
parser = argparse.ArgumentParser(description='you should add those parameter')
# 添加一个具有两个选项的参数：-o（short）和--output（long）。这个参数是可选参数
# action：如果设置为store_true，则action会将参数存储为True。 help选项提供参数帮助
parser.add_argument('-o', '--output', action='store_true', help="shows output")
# 参数由parse_args()解析。解析的参数（long）作为对象属性存在
args = parser.parse_args()
# 访问对象的属性
if args.output:
    print("There are optional args output")

# 此时，我们可以使用-o或--output运行程序：
"""
$ optional_args.py -o
There are optional args output
$ optional_args.py --output
There are optional args output
"""
# 我们也可以向程序显示帮助：
'''
$ optional_args.py --help
usage: optional_args.py [-h] [-o]

optional arguments:
    -h, --help    show this help message and exit
    -o, --output  shows output
'''

# 2）argparse必需参数

# 使用required选项需要一个必须参数

parser = argparse.ArgumentParser()
# 例如，必须指定一个name选项，否则失败
parser.add_argument('--name', required=True)
args = parser.parse_args()
print(f'Hello {args.name}')
# 此时，我们必须使用--name运行程序：
'''
$ required_args.py --name Alice
Hello Alice

$ required_args.py
usage: required_args.py [-h] --name NAME
required_args.py: error: the following arguments are required: --name
'''

# 3）argparse位置参数

parser = argparse.ArgumentParser()
# 本示例需要两个位置参数：name和age，创建位置参数时不带破折号前缀字符
parser.add_argument('name')
parser.add_argument('age')

args = parser.parse_args()

print(f'{args.name} is {args.age} years old')
# 创建位置参数时不带破折号前缀字符：
'''
$ positional_args.py Tom 18
Tom is 18 years old
'''

# 4）argparse目标

# add_argument()的dest选项为参数指定名称。如果未给出，则从选项中推断出来

import datetime

parser = argparse.ArgumentParser()

parser.add_argument('-n', dest='now', action='store_true', help="shows now")

args = parser.parse_args()

if args.now:
    print(datetime.datetime.now())

# 程序会将now名称赋予-n参数：
'''
$ dest.py -n
2023-11-28 15:20:40.406571
'''

# 5）argparse类型

# type参数确定参数类型

import random

parser = argparse.ArgumentParser()
# -n选项是必需参数，且需要整数值
parser.add_argument('-n', type=int, required=True, help="define the number of random integers")

args = parser.parse_args()

for i in range(args.n):
    print(random.randint(0, 100))

# 程序显示0到100的n个随机整数：
'''
$ rand_int.py -n 3
92
61
16
'''

# 6）argparse默认

# 如果未指定包含default的选项，则使用指定默认值
import math

parser = argparse.ArgumentParser()

# 指定一个基础值，-b选项是必需参数，且需要整数值
parser.add_argument('-b', type=int, required=True, help="defines the base value")
# 指定一个指数值，-e选项是可选参数，如果未指定，则默认取2
parser.add_argument('-e', type=int, default=2, help="defines the exponent value")

args = parser.parse_args()

print(math.pow(args.b, args.e))

# 计算指数，如果未给出指数值，则使用默认值2
'''
$ power.py -b 3
9
$ power.py -b 3 -e 3
27
'''



