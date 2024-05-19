# Python模块、包管理与发布
# 虚拟环境、项目结构

# 1、Python模块与导入
"""
模块是一种组织和重用代码的方式，就是一个Python文件，以.py结尾；模块可以包含函数、变量和类等，可以被其他程序或模块导入和使用
"""
# 模块的使用：
# 1）导入/引入模块
# 导入整个模块：import 模块名 [as 模块别名]
# 导入模块函数或变量：from 模块名 import 函数名/变量名(多个使用,分割) [as 模块别名]
# 2）调用模块函数：模块名.函数名
import Python函数.py03 as mymodule
print(mymodule.greet('Alice'))

from Python函数.py03 import greet
print(greet('Bob'))

import math as m
print(m.pow(2, 3))

# 2、Python中的__name__属性
'''
在Python中，每个模块都有一个特殊的__name__属性，用于标识模块的名称
当模块直接被执行时，__name__属性的值为：__main__
当模块被导入时，__name__属性的值为：模块实际名称
通过使用__name__属性，可以判断模块是被直接执行还是被导入，并在不同情况下执行相应的代码
'''
def greet(name):
    print("Hi, " + name + "！")

# 在模块文件中直接执行的代码（当本模块被导入时，__name__属性的值为模块实际名称py04，所以greet("Alice")不会被执行）
if __name__ == '__main__':
    greet("Alice")

# 3、Python的模块搜索路径
'''
在导入模块时，Python会按照特定的搜索路径来查找模块文件
Python模块搜索路径的常见位置有：当前目录、标准库目录、第三方库目录
可以使用sys模块的path属性查看Python的模块搜索路径
'''
import sys
# sys.path是一个包含模块搜索路径的列表，可以打印出来查看Python的模块搜索路径
print(sys.path)

# Python包管理与发布

# 4、Python包（Package）
'''
包（Package）是一种用于组织和管理模块的方式；包是一个包含多个模块的目录，目录中必须包含一个名为__init__.py的文件
1）包结构示例：day02包名，包含两个模块py03.py和py04.py
day01/
    __init__.py
    py01.py
    py02.py
2）导入包中的模块
import 包名.模块名 as 别名
3）导入包中的模块的函数或变量
from 包名.模块名 import 函数名/变量名
'''

# 5、Python包管理工具pip
'''
pip是Python的包管理工具，用于安装、升级和卸载Python包
常用pip命令（命令输入执行位置：编辑器Terminal终端）：
1）安装包：pip install package_name
2）升级包：pip install --upgrade package_name
3）卸载包：pip uninstall package_name
4）查看已安装的包：pip list
'''

# 6、Python包的发布与分发
'''
要发布和分发自己的Python包，可以使用Python的包管理工具setuptools和twine
发布和分发Python包的一般步骤：
1）创建包的目录结构，并确保包含__init__.py文件
2）在包的根目录下创建setup.py文件，用于描述包的元数据和依赖项
3）使用setuptools进行包的构建和打包，将项目打包成一个wheel包（一种Python二进制包格式），其中sdist表示
源码，bdist表示二进制码：python setup.py sdist bdist_wheel
4）注册一个PyPI账户（https://pypi.org/）并进行身份验证
5）使用twine上传包到PyPI（分发）：twine upload dist/*
6）其他人就可以通过pip来安装和使用发布的包：pip install dist/包名.whl
'''

# Python虚拟环境管理
'''
Python虚拟环境是一种隔离和管理Python包和依赖项的方式。它允许你在同一台计算机上创建多个独立的Python环境
每个环境可以有自己独立的Python版本和安装的包，而不会相互干扰。通过使用Python虚拟环境，可以在不同项目之间
隔离依赖项，确保每个项目都使用特定版本的包(避免依赖冲突)，并且可以轻松共享项目的环境配置
'''
# Python项目
'''
1）新建Python项目时，可以选择"本地编译器"和"虚拟环境编译器"；选择New environment using->选择虚拟环境(Virtualenv)
此时，项目目录结构中会多出venv目录
venv环境下，使用pip安装的包都会安装到venv这个环境下，系统Python环境不受任何影响；Python 虚拟环境主要是为不同Python
项目创建一个隔离的环境，每个项目都可以拥有独立的依赖包环境，而项目间的依赖包互不影响，从而提高项目的可维护性和稳定性
2）通用项目目录结构：
myproject/
    ├── .idea/                   创建项目时自动生成的配置目录
    │   ├── inspectionProfiles/  包含项目的代码审查配置信息，例如代码检查器的设置、代码风格检查等
    │   ├── runConfigurations/   包含项目的运行配置信息，例如运行Python脚本、调试程序等
    │   ├── workspace.xml        包含项目的整体配置信息，例如项目的 SDK、Python解释器、代码风格等
    │   ├── modules.xml          定义项目的模块(多个模块)，以及模块之间的依赖关系
    │   ├── misc.xml             包含项目的其他配置信息，例如代码审查工具、代码模板等
    │   └── vcs.xml              包含与版本控制相关的配置信息(使用版本控制工具如Git)
    ├── venv/                    整个应用程序的虚拟环境目录
    │   ├── python.exe           Python解释器
    │   └── site-packages        用于存放虚拟环境中安装的第三方库
    ├── models/                  工程源码
    │   ├── sub_model/
    │   └── ...
    ├── config.py                配置文件
    ├── requirements.txt         存放依赖的外部Python包列表
    ├── setup.py                 安装、部署、打包脚本
    ├── main.py                  程序入口
    └── README.md                项目说明文档
2.1）.idea目录：创建项目时自动生成，用于存储当前项目相关的一些配置、状态（如版本信息，历史记录等）
.idea目录是否可删除：可删除，删除后不能再使用Pycharm进行回溯和复原
删除方法：关闭Pycharm后在本地项目目录将.idea目录删除，重新打开工程后就会变成工程的初始页面
2.2）venv/目录：项目的虚拟环境目录，用于存储Python虚拟环境(如第三方库)，venv文件夹中通常
包含一个Python解释器和一个独立的site-packages目录，用于存放虚拟环境中安装的第三方库
2.3）requirements.txt：如何在虚拟环境快速、高效的安装所需插件和第三方依赖？
步骤1：在项目根目录下新建requirements.txt文件，在文件中写好对应的插件和第三方依赖，格式如（指定版本：包名 == 版本）
numpy
pandas
requests
...
步骤2：在Pycharm终端执行命令统一安装（为提高安装每个插件的成功几率，可配置指定镜像源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
'''

# 7、Python虚拟环境工具venv的使用
'''
Python3.3及以上版本内置了一个名为venv的模块，用于创建和管理Python虚拟环境
使用venv工具创建和激活Python虚拟环境步骤:
1）创建虚拟环境
python3 -m venv myenv(虚拟环境名)
2）激活虚拟环境 (Windows)
myenv\Scripts\activate.bat
3）激活虚拟环境 (Mac/Linux)
source myenv/bin/activate
'''
# ※ 激活虚拟环境报错(打开终端Windows PowerShell后报错)及解决：
'''
1）报错内容：
无法加载文件 F:\work\venv\Scripts\activate.ps1，因为在此系统上禁止运行脚本。有关详细信息，请参阅https:/go.microsoft.com/fwlink/?LinkID=135170中的about_Execution_Policies。
2）报错原因：
Win10默认PowerShell的执行策略是不载入任何配置文件，不运行任何脚本(Restricted)，它不允许执行任何脚本，包括虚拟环境的激活脚本
这一点可以通过命令查看：在Pycharm终端执行：get-executionpolicy，返回结果：Restricted
3）问题解决：修改PowerShell的执行策略
步骤1：方式1(推荐)：打开Pycharm终端(选择Windows PowerShell)；方式2：电脑搜索PowerShell，以管理员身份运行Windows PowerShell应用
步骤2：执行以下命令修改PowerShell的执行策略：
方式1(推荐)：Pycharm终端：Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
方式2：Windows PowerShell命令框：Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
步骤3：验证：PowerShell执行策略已设置为RemoteSigned，它允许执行本地脚本和来自受信任发布者的远程脚本
注意事项：此操作仅修改了当前用户的Windows PowerShell执行策略
'''

# 8、Python虚拟环境工具conda的使用
'''
conda是一个跨平台的包管理和环境管理系统，可用于创建、安装和切换虚拟环境，还可安装包、管理环境和导出环境配置等
使用conda工具创建和激活Python虚拟环境步骤:
1）创建虚拟环境
conda create --name myenv
2）激活虚拟环境
conda activate myenv
3）退出虚拟环境
conda deactivate
'''

# 9、Python解释器的选择
'''
Python有多个解释器可供选择，每个解释器都有自己的特点和用途，可以根据项目需求选择适合的Python解释器来运行和调试代码
1）CPython：CPython是官方的Python解释器，它是用C语言实现的，并且是最常用的解释器。大多数Python代码都可以在CPython上运行
2）Jython：Jython是一个使用Java虚拟机（JVM）作为运行环境的Python解释器。它允许你在Java平台上运行Python代码
3）IronPython：IronPython是一个使用.NET Framework作为运行环境的Python解释器。它允许你在.NET平台上运行Python代码
4）PyPy：PyPy是一个采用即时编译技术的Python解释器，它可以提供更好的性能和内存利用率
'''

