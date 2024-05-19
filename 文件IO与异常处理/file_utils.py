
# Python常用文件IO操作

# 准备：Test目录：
"""
Test/
    ├── Test1/
    │        ├── d.txt
    │        └── e.xlsx
    ├── a.xlsx
    ├── b.txt
    └── c.xlsx
"""

import os

# 1、递归遍历指定目录下的指定类型文件
import string

def get_files(path, suffix, recursion=False):
    try:
        # 判断目录是否存在
        if not os.path.exists(path) or os.path.isfile(path):
            raise IOError("目录不存在或不是目录！")
        # 获取所有文件及目录名（带后缀）
        list_file = os.listdir(path)
        # 拼接路径
        files = [os.path.join(path, file) for file in list_file]
        # 筛选指定文件
        res_files = [file for file in files if os.path.isfile(file) and file.endswith(suffix)]
        if recursion:
            # 筛选目录
            dirs = [dir for dir in files if os.path.isdir(dir)]
            for dir in dirs:
                res_files += get_files(dir, suffix, True)
        return res_files
    except IOError as e:
        print(e)


# print(get_files(r'C:\Users\cc\Desktop\Test', '.xlsx', True))


# glob模块是Python标准库模块之一，主要用来查找符合特定规则的目录和文件，返回到一个结果列表
# 4个通配符：
'''
*   匹配0个或多个字符
**  递归匹配所有文件、目录
？  代匹配一个字符
[]  匹配指定范围内的字符，如[0-9]匹配数字、[a-z]匹配小写字母
'''
# 3个函数：
'''
glob()：返回所有符合匹配条件的文件路径列表
iglob()：返回所有符合匹配条件的文件路径迭代器
escape()：用于转义4个通配特殊字符
'''
import glob

# 2、递归遍历指定目录下的指定类型文件或目录
def get_status(path, pattern, recursive=False):
    try:
        # 判断目录是否存在
        if not os.path.exists(path) or os.path.isfile(path):
            raise IOError("目录不存在或不是目录！")
        if recursive:
            path = os.path.join(path, '**')
            return glob.glob(os.path.join(path, pattern), recursive=True)
        else:
            return glob.glob(os.path.join(path, pattern))
    except IOError as e:
        print(e)


# print(get_status(r'C:\Users\cc\Desktop\Test', '*.xlsx', True))
# print(get_status(r'C:\Users\cc\Desktop\Test', 'Test*'))


# Python中的os和shutil是用于处理文件和目录操作的标准库
# shutil库是os库的扩展，提供了更高级的文件和目录操作功能，如复制、移动、剪切、删除、压缩解压等
# shutil模块对压缩包的处理是调用ZipFile和TarFile两个模块来进行的

import shutil
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

# 3、复制文件/目录到指定目录下
def copy_status(status, path):
    try:
        # 判断路径是否存在
        if not os.path.exists(status):
            raise IOError("文件或目录不存在！")
        # 目标路径不存在时创建
        if not os.path.exists(path):
            os.makedirs(path)
        # 获取路径中的文件名
        filename = os.path.basename(status)
        if os.path.isfile(status):
            shutil.copy(status, os.path.join(path, filename))
        if os.path.isdir(status):
            shutil.copytree(status, os.path.join(path, filename))
    except IOError as e:
        print(e)


# copy_status('C:\\Users\\cc\\Desktop\\Test\\c.xlsx', 'C:\\Users\\cc\\Desktop\\TestX')
# copy_status('C:\\Users\\cc\\Desktop\\Test', 'C:\\Users\\cc\\Desktop\\TestX')

# 4、文件/目录剪切
def cut_status(status, path):
    try:
        # 判断路径是否存在
        if not os.path.exists(status):
            raise IOError("文件或目录不存在！")
        # 目标路径不存在时创建
        if not os.path.exists(path):
            os.makedirs(path)
        # 获取路径中的文件名
        filename = os.path.basename(status)
        # 移动
        shutil.move(status, os.path.join(path, filename))
        # 删除原来文件或目录
        if os.path.isfile(status):
            os.remove(status)
        if os.path.isdir(status):
            shutil.rmtree(status)
    except IOError as e:
        print(e)


# cut_status(r'C:\Users\cc\Desktop\TestX\c.xlsx', 'C:\\Users\\cc\\Desktop\\TestY')
# cut_status(r'C:\Users\cc\Desktop\TestX\Test', 'C:\\Users\\cc\\Desktop\\TestY')

import zipfile
# 5、压缩目录
def zip_dir(path):
    try:
        # 判断目录是否存在
        if not os.path.exists(path):
            raise IOError("目录不存在！")
        # 是否是目录
        if not os.path.isdir(path):
            raise IOError("类型不是目录！")
        with zipfile.ZipFile(path + '.zip', 'w') as zip:
            # dir_path: 源文件夹路径 dir_names: 子文件夹 file_names: 文件夹下的所有文件
            for dir_path, dir_names, file_names in os.walk(path):
                # 去掉层级路径
                fpath = dir_path.replace(path, '')
                for filename in file_names:
                    zip.write(os.path.join(dir_path, filename), os.path.join(fpath, filename))
    except IOError as e:
        print(e)

# zip_dir(r'C:\Users\cc\Desktop\Test')

# 6、解压缩文件
def unzip_dir(path):
    try:
        # 判断文件是否存在
        if not os.path.exists(path):
            raise IOError("文件不存在！")
        # 是否是.zip文件
        if os.path.splitext(path)[1] != '.zip':
            raise IOError("类型不是ZIP！")
        with zipfile.ZipFile(path, 'r') as unzip:
            os.makedirs(path.replace(".zip", ""))
            # 解压到当前文件夹
            unzip.extractall(path.replace(".zip", ""))
    except IOError as e:
        print(e)


# unzip_dir(r'C:\Users\cc\Desktop\Test.zip')

# 7、将指定目录下的指定文件纵向(垂直)合并
import pandas as pd

# 本方法仅支持Excel和CSV，返回合并后的DataFrame
def concat_files(path, pattern, recursive=False):
    try:
        # 判断目录是否存在
        if not os.path.exists(path) or os.path.isfile(path):
            raise IOError("目录不存在或不是目录！")
        if recursive:
            path = os.path.join(path, '**')
            files = glob.glob(os.path.join(path, pattern), recursive=True)
            return table2df(files, pattern)
        else:
            files = glob.glob(os.path.join(path, pattern))
            return table2df(files, pattern)
    except IOError as e:
        print(e)

def table2df(files, pattern):
    try:
        if len(files) == 0:
            raise IOError("文件不存在！")
        df_list = []
        suffix = os.path.splitext(pattern)[1]
        if suffix == '':
            raise IOError("Pattern格式错误！")
        elif suffix == '.xlsx':
            for file in files:
                df_list.append(pd.read_excel(file))
            return pd.concat(df_list, axis=0, ignore_index=True)
        elif suffix == '.csv':
            for file in files:
                df_list.append(pd.read_csv(file))
            return pd.concat(df_list, axis=0, ignore_index=True)
        else:
            raise IOError("仅支持Excel和CSV！")
    except IOError as e:
        print(e)
        return pd.DataFrame()


# print(concat_files(r'C:\Users\cc\Desktop\Test', '*.xlsx').to_string())
# print(concat_files(r'C:\Users\cc\Desktop\Test', '*.xlsx', True).to_string())

# 8、提取文本文件中匹配正则的字符串列表
# 常用正则表达式
'''
[0-9]+            数字
[A-Za-z]+         英文字母
[A-Za-z0-9]+      数字和英文字母
[A-Za-z0-9_]+     字母数字下划线
[\u4e00-\u9fa5]   中文
\d                数字[0-9]
\D                非数字
\w                [0－9a-zA-Z_]
\W                除[0－9a-zA-Z_]外的字符
\s                空白字符（如空格、Tab、换页符等）
\S                非空白字符
'''
import re

def text_parser(path, pattern):
    with open(path, 'r+', encoding='utf-8') as file:
        text = file.read()
        return re.findall(pattern, text)

# print(text_parser(r'C:\Users\cc\Desktop\Test\b.txt', '[A-Za-z]+'))

# 9、词频统计

# jieba模块

# jieba是Python的一个第三方中文分词函数库
# 常用方法有：
'''
jieba.lcut(s)：精确模式
jieba.lcut(s,cut_all=True)：全模式
jieba.lcut_for_search(s)：搜索引擎模式
'''
# 安装：pip install jieba

import jieba
import zhon.hanzi

punc = zhon.hanzi.punctuation     # 需要去除的中文标点符号
# print(punc)
punc_e = string.punctuation       # 需要去除的英文标点符号
# print(punc_e)

# 1）中文词频统计：输出TopN，返回List[Tuple]类型
def word_count(path, n):
    with open(path, 'r+', encoding='utf-8') as file:
        text = file.read()
        # 分词
        words = jieba.lcut(text)
        # 去除中文标点符号
        words = [word for word in words if word not in punc and len(word) > 1]
        # 词频统计
        wcs = {}
        for w in words:
            wcs[w] = wcs.get(w, 0) + 1
        # 排序
        res = sorted(wcs.items(), key=lambda d: d[1], reverse=True)
        return res[:n]

print(word_count(r'C:\Users\cc\Desktop\Test\b.txt', 5))

# 2）英文词频统计
def word_count(path, n):
    with open(path, 'r+', encoding='utf-8') as file:
        text = file.read()
        words = [word for word in re.split('\\s+', re.sub('\\W', ' ', text)) if len(word) > 1]
        wcs = {}
        for w in words:
            wcs[w] = wcs.get(w, 0) + 1
        res = sorted(wcs.items(), key=lambda d: d[1], reverse=True)
        return res[:n]

print(word_count(r'C:\Users\cc\Desktop\Test\b.txt', 5))

# jieba词性标注
import jieba.posseg as pseg

text = "This is a Test."
words_pair = [(word, tag) for word, tag in pseg.lcut(text) if len(word.strip()) > 0]
print(words_pair)
'''
[('This', 'eng'), ('is', 'eng'), ('a', 'x'), ('Test', 'eng'), ('.', 'm')]
'''


# NLTK模块

# NLTK是Python内置的自然语言处理模块，可以处理文本语言的各种任务，如分词、词性标注、情感分析等
# NLTK只支持英文分词，不支持中文分词
# 官方文档：https://www.nltk.org/data.html

import nltk
import zhon as zhon

# 下载内置的语料库
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')

# 分词
text = "This is a Test."
words = nltk.word_tokenize(text)
print(words)
'''
['This', 'is', 'a', 'Test', '.']
'''

# 词性标注
tags = nltk.pos_tag(words)
print(tags)
'''
[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('Test', 'NN'), ('.', '.')]
'''

# 去除停用词（不会去除标点符号）
# 在NLP任务中，我们可能希望移除一些对分析贡献不大的词，这些词被称为停用词
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
res_words = [w for w in words if w.lower() not in stop_words]
print(res_words)
'''
['Test', '.']
'''


# SpaCy模块

# NLTK是一个功能强大的NLP库，但SpaCy的性能更高，且更容易集成到生产环境中
# SpaCy是Python最流行、最强大的自然语言处理库，特别适用于大规模的文本处理任务
# 官网：https://spacy.io/
# 安装：pip install spacy
# 安装英文模型：python -m spacy download en_core_web_sm
# 安装中文模型：python -m spacy download zh_core_web_sm

import spacy

# 加载英文模型
nlp = spacy.load("en_core_web_sm")
# 分词
doc = nlp(text)
words = [token.text for token in doc]
print(words)
'''
['This', 'is', 'a', 'Test', '.']
'''
# 词性标注，可用于识别文本中名词、动词等
tags = [token.pos_ for token in doc]
print(tags)
'''
['PRON', 'AUX', 'DET', 'NOUN', 'PUNCT']
'''
# 命名实体识别，可用于识别文本中的人名、地名等
labels = [ent.label_ for ent in doc.ents]
print(labels)

text_cn = "SpaCy是一个用于高级自然语言处理的Python库！SpaCy的设计目标是高性能、易于使用和可扩展性。"

# 加载模型，并排除掉不需要的Components
nlp = spacy.load("zh_core_web_sm", exclude=("tagger", "parser", "senter", "attribute_ruler", "ner"))
# 分词
doc = nlp(text_cn)
words = [token.text for token in doc]
print(words)
'''
['SpaCy', '是', '一个', '用于', '高级', '自然', '语言', '处理', '的', 'Python库', '！', 'SpaCy', '的', '设计', '目标', '是', '高性能', '、', '易于', '使用', '和', '可', '扩展性', '。']
'''

# 去除标点符号
punc = zhon.hanzi.punctuation
punc_words = [word for word in words if word not in punc]
print(punc_words)
'''
['SpaCy', '是', '一个', '用于', '高级', '自然', '语言', '处理', '的', 'Python库', 'SpaCy', '的', '设计', '目标', '是', '高性能', '易于', '使用', '和', '可', '扩展性']
'''

# 去除停用词（会去除标点符号）
filter_words = [w for w in words if not nlp.vocab[w].is_stop]
print(filter_words)
'''
['SpaCy', '用于', '高级', '自然', '语言', 'Python库', 'SpaCy', '设计', '目标', '高性能', '易于', '扩展性']
'''

# 词频统计
from collections import Counter

wcs = Counter(filter_words)
items = list(wcs.items())
print(items)
'''
[('SpaCy', 2), ('用于', 1), ('高级', 1), ('自然', 1), ('语言', 1), ('Python库', 1), ('设计', 1), ('目标', 1), ('高性能', 1), ('易于', 1), ('扩展性', 1)]
'''

# 排序
wcs_sort = sorted(items, key=lambda t: t[1], reverse=True)
print(wcs_sort)

# 10、Excel-Sheet表转CSV

import warnings

def excel2csv(excel, sheet):
    # 警告解决：UserWarning: Workbook contains no default style, apply openpyxl's default
    warnings.simplefilter('ignore')
    with pd.ExcelFile(excel) as xlsx:
        data = xlsx.parse(sheet, header=0)
    print(f"数据条数: {len(data)}")
    csv = f"{os.path.splitext(excel)[0]}_{sheet}.csv"
    data.to_csv(csv, index=False, encoding="utf-8")
    return data

import numpy as np

# 11、TXT文本文件格式化
# read_table()：将具有格式的txt文本文件格式化为Excel/CSV输出，经过验证，其中需要注意的字符串如下：
# 可以自动转为空值（NaN）的字符串：None、NA、nan、NaN、null、NULL、N/A、<NA>、''
# 不可自动转为空值（NaN）的字符串：na、Na、none、np.nan、Null、\N
# 需要转换的字符串：true、false
# txt格式化为Excel或CSV
def export_txt_format(txt: str, fmt='excel', sep='\t', dtype=None, **kwargs):
    base = os.path.splitext(txt)[0]
    # 部分软件导出txt数据空值显示为：\N
    data = pd.read_table(filepath_or_buffer=txt, sep=sep, dtype=dtype, **kwargs)
    # 所有可能的空值替换
    data.replace(['\\N', 'Null', 'none', 'Na', 'na'], np.NaN, inplace=True)
    # true、false替换
    data.replace(['true', 'false'], ['True', 'False'], inplace=True)
    if fmt == 'excel':
        data.to_excel(f'{base}.xlsx', index=False, engine='xlsxwriter')
    elif fmt == 'csv':
        data.to_csv(f'{base}.csv', index=False, encoding='utf-8')
    else:
        raise Exception("Only supports excel and csv.")
    return data

# 注意：Excel超长数字字符串类型若不使用dtype指定字符串类型，则会以科学计数法表示，结果丢失精度；而CSV不指定类型不会出现该情况



