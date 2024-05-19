
# 源码打包上传

# 1、为什么要打包源码
"""
如果你想让你的Python代码，通过pip install方式供所有人下载，那就需要将代码上传到PyPi上，这样才能让所有人使用
"""
# 2、前提条件
'''
1）有一个PyPi官网账号；注册地址: https://pypi.org/account/register/
2）更新pip版本到最新：py -m pip install --upgrade pip
3）通过pip安装twine：要使用twine来上传代码
4）安装编译工具：pip install --upgrade build
'''
# 3、操作步骤
# 1）创建项目结构
# 1.1）创建本地目录结构
'''
root/
└── src/
    └── module/
        ├── __init__.py
        └── dlt.py
'''
# 以上除了src和__init__.py固定外其他都可自定义；目录结构需保持一致
# 其中__init__.py用于将目录作为包导入，默认可为空；dlt.py是包中的一个模块，用于提供功能供下载者调用

# 1.2）创建上传所需的文件
'''
root/
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.py
└── src/
    └── module/
        ├── __init__.py
        └── dlt.py
'''
# pyproject.toml：告诉构建工具构建项目所需的内容
'''
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"
'''
# README.md：包的描述信息
# setup.py：setuptools的构建脚本；它告诉setuptools你的包（例如名称和版本）以及要包含的代码文件
'''
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="包的分发名称",
    version="0.0.1",  # 包版本
    author="Author",  # 标识包的作者
    author_email="author@example.com",
    description="example package",  # 简短的包摘要
    long_description=long_description,  # 包的详细说明
    long_description_content_type="text/markdown",  # 描述使用什么类型的标记
    # url="",  # 项目主页URL，可不写
    # project_urls={},  # 显示的任意数量的链接，通常是文档、问题跟踪器等
    package_dir={"": "src"},  # src目录被指定为根包
    packages=setuptools.find_packages(where="src"),  # 包含在分发包中的所有 Python导入包的列表
    python_requires=">=3.6"   # 给出项目支持的Python版本
)
'''
# setup.cfg：配置文件，推荐使用setup.py配置
# LICENSE：许可文件
'''
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
# 2）编译打包
# 在pyproject.toml文件同级目录打开命令行工具，执行：python -m build
# 打包完成后，会生成dist文件和打包文件

# 3）源码上传
# 执行检查：twine check dist/*
# 检查是否存在问题，若提示存在问题，先解决；若无问题，执行命令上传：twine upload dist/*

# 4）验证是否可安装
'''
a、访问上传成功的地址，是否存在刚才上传的包
b、使用pip install xxx验证是否可安装
'''
# 注意：如果使用的镜像不是官网，例如国内使用最多的清华镜像，可能需要等5分钟以上才能安装，镜像同步需要时间

