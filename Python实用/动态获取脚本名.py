"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/11/30
Time: 21:32
Description:
"""
# 如何动态获取调用者当前执行的Python脚本名？

# sys提供了一个函数sys._getframe()用于查看函数被什么函数调用及被第几行调用以及被调用函数所在的文件
# sys._getframe(depth)从调用堆栈返回一个框架对象。 如果给定了可选整数depth，则返回在堆栈顶部以下调用多次的框架对象。depth的默认值为0，返回调用堆栈顶部的帧。如果参数比调用堆栈更深，则会引发ValueError异常
# sys._getframe()的常用使用：
'''
sys._getframe(0)                       # 被调用模块层
sys._getframe(1)                       # 调用者模块层
sys._getframe().f_back                 # 调用者模块层，和sys._getframe(1)相同
sys._getframe(0).f_code.co_filename    # 被调用模块当前文件名，也可以通过__file__获得
sys._getframe(1).f_code.co_filename    # 调用者模块当前文件名，和sys._getframe().f_back.f_code.co_filename相同
sys._getframe(0).f_code.co_name        # 被调用模块层当前函数名
sys._getframe(1).f_code.co_name        # 调用者模块层调用所在函数名，如果没有，则返回<module>
sys._getframe(0).f_lineno              # 被调用模块层被调用函数的行号
sys._getframe(1).f_lineno              # 调用者模块层调用该函数所在的行号
'''
# sys提供的sys.argv[]可用于访问命令行参数。sys.argv[0]可用于获取正在执行的Python脚本本身

# 工具API模块：module.py
import os
import sys

def do():
    # 方式1：
    print(os.path.basename(sys._getframe(1).f_code.co_filename))
    print(os.path.splitext(os.path.basename(sys._getframe(1).f_code.co_filename))[0])
    # 方式2：
    print(os.path.basename(sys.argv[0]))
    print(os.path.splitext(os.path.basename(sys.argv[0]))[0])

# 调用者模块：client.py
# from module import do

do()
'''
client.py
'''



