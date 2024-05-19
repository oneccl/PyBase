"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/11/20
Time: 21:01
Description:
"""

# Python文本翻译库

# translate非标准库是Python中可以实现对多种语言进行互相翻译的库，translate可以将原始文本翻译成我们需要的目标语言
# en：英语
# zh/zh-CN：简体中文
# ru：俄语
# ko：韩语

# 安装：pip install translate

from translate import Translator

def translate(text: str, sl='en', dl='zh'):
    translator = Translator(from_lang=sl, to_lang=dl)
    return translator.translate(text)


line = "Apache Flink is a framework and distributed processing engine for stateful computations over unbounded and bounded data streams. Flink has been designed to run in all common cluster environments, perform computations at in-memory speed and at any scale."
print(translate(line))
'''
Apache Flink是一个框架和分布式处理引擎，用于在无界和有界数据流上进行有状态计算。Flink旨在在所有常见集群环境中运行，以内存速度和任何规模执行计算。
'''




