
# JupyterLab

"""
JupyterLab是Jupyter Notebook的全面升级，是一个集文本编辑器、终端以及各种个性化组件于一体的全能IDE
JupyterLab支持更多数据格式的预览与修改，除了代码文件（.py、.cpp、.java等），还包括CSV、JSON、Markdown、PDF、PPT等
JupyterLab是一个加强版的资源管理器和交互模式下的Python，能让我们可视化地进行一些数据操作
JupyterLab的执行文件被称作notebook，它的后缀是.ipynb
"""

"""
JupyterLab是Jupyter主打的最新数据科学生产工具，JupyterLab包含了Jupyter Notebook的所有功能
JupyterLab作为一种基于Web的集成开发环境，可以使用它编写notebook、操作终端、Markdown文本、打开交互模式、查看CSV文件及图片等功能
"""
# 特点：
'''
交互模式：Python交互式模式可以直接输入代码，然后执行，并立刻得到结果，因此Python交互模式主要是为了调试Python代码用的
内核支持的文档：使你可以在可以在Jupyter内核中运行的任何文本文件（Markdown，Python，R等）中启用代码
模块化界面：可以在同一个窗口同时打开多个Notebook或文件（HTML、TXT、Markdown等），且都以标签的形式展示，更像是一个IDE
镜像Notebook输出：让你可以轻易地创建仪表板
同一文档多视图：使你能够实时同步编辑文档并查看结果
支持多种数据格式：可以查看并处理多种数据格式，也能进行丰富的可视化输出或Markdown形式输出
云服务：使用Jupyter Lab连接Google Drive等服务，可以极大地提升生产力
'''

# 1）JupyterLab安装
# cmd命令框执行：pip install jupyterlab

# 2）启动
# 开启JupyterLab工作页面：jupyter-lab 或 jupyter lab
# 默认在本地的8888端口启动，若本地已经有一个Jupyter正在运行，再启动就会运行在8889端口
# 运行完该命令，会在系统默认的浏览器打开Jupyter网页：http://localhost:8888/lab

# 通过修改jupyter_lab_config.py文件可以自定义JupyterLab启动端口与文件保存目录
# 在cmd命令行执行jupyter lab --generate-config命令，可查看jupyter_lab_config.py配置文件路径

# 3）切换语言中文
# 安装中文语言包：
# cmd下执行：pip install jupyterlab-language-pack-zh-CN
# 重启JupyterLab，通过Settings->Language更改语言

# 4）Cell及使用
# 选择新建一个notebook，编辑内容：print('Hello World！')，点击上面三角运行
# 在notebook里，一个基本的代码块被称作一个cell，一个cell理论上可以有无数行代码
# 每一个cell有两种模式：命令模式（蓝色条）和编辑模式（绿色条）
# 在命令模式下，按enter或鼠标单击代码框可以进入编辑模式；在编辑模式下，按esc或鼠标单击代码框左侧区域即可进入命令模式

# 5）安装插件
# 搜索需要安装的插件(前提：安装NodeJS)，点击插件下的install即可

# 6）退出JupyterLab
# 注意：直接关闭网页是无法退出的，因为是通过控制台启动的JupyterLab
# 方式1：File->Shut Down->确认退出
# 方式2：cmd控制台，按2次ctrl+c

