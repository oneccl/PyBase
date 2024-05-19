
# Flask框架简介与环境搭建

"""
1、Flask
"""
'''
Flask是一个轻量级的Python Web框架，它被设计成简单易用且灵活的工具，用于构建Web应用程序和API
Flask提供了基本的功能和组件，同时也支持扩展，可以根据项目的需求进行定制。Flask框架的特点：
1）简单易用：Flask的API简洁明了，易于上手和学习。它提供了构建Web应用所需的基本功能，但又没有过多的复杂性和冗余
2）灵活性：Flask允许开发者自由选择需要的扩展和库，以构建符合项目需求的应用。它没有严格的约束和规范，你可以根据自己的喜好和需求进行定制
3）微框架：Flask被称为微框架，因为它的核心库只提供了最基本的功能。这使得Flask非常轻量级，适用于小型和中小型的Web应用
4）Jinja2模板引擎：Flask使用Jinja2作为模板引擎，可以方便地进行页面渲染和数据展示。Jinja2提供了丰富的模板语法和功能，使得前端开发更加便捷
5）Werkzeug工具库：Flask基于Werkzeug工具库构建，它提供了一系列底层的Web工具，包括HTTP请求处理、URL路由、会话管理等
'''

'''
2、Flask环境安装、配置及搭建
搭建Flask开发环境：
1）安装Python：https://www.python.org
2）安装虚拟环境（可选）：为了隔离不同项目的依赖包，建议使用虚拟环境。可以使用Python内置的venv模块创建虚拟环境，或者使用第三方工具如virtualenv等
3）安装Flask：打开命令行终端，并执行命令：pip install flask
4）创建Flask应用：新建一个Python脚本文件（例如app.py），并在文件中导入Flask模块（如下）：
'''
from flask import Flask

app = Flask(__name__)

# 定义一个路由处理函数，当访问根路径/时，会返回字符串'Hello, Flask！'
@app.route('/')
def hello():
    return "Hello, Flask！"

# app.run()用于启动应用
if __name__ == '__main__':
    app.run()

'''
5）运行应用：保存并执行Python脚本，以启动Flask应用；在命令行终端中，进入脚本所在的目录，并执行命令：
python py11.py
如果一切正常，将看到类似以下的输出：
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
这表示Flask应用已经成功运行在本地的http://127.0.0.1:5000/地址上
6）浏览器访问：http://127.0.0.1:5000/ 返回Hello, Flask！
'''

'''
3、Flask项目结构
在开发一个大型的Flask应用时，良好的项目结构可以帮助我们组织和管理代码。下面是一个典型的Flask项目结构：
myapp/
    ├── app/
    │   ├── __init__.py
    │   ├── routes.py
    │   ├── models.py
    │   ├── templates/
    │   │   └── index.html
    │   └── static/
    │       └── style.css
    ├── config.py
    ├── requirements.txt
    ├── run.py/main.py
    └── README.md
1）app/目录：存放应用的代码和资源
__init__.py文件初始化应用，并将路由、模型等组件注册到应用中；routes.py文件定义应用的路由和视图函数；
models.py文件定义应用的数据模型；templates/目录存放HTML模板文件；static/目录存放静态文件，如CSS、JavaScript等
2）config.py：存放应用的配置信息，如数据库连接、密钥等
3）requirements.txt：列出了应用所需的依赖包及其版本
4）run.py：启动应用的入口脚本
5）README.md：项目的说明文档，通常使用Markdown格式编写
'''
'''
补充：Flask RESTful API项目结构
myapp/
    ├── app/
    │   ├── __init__.py
    │   ├── models.py
    │   ├── resources/
    │   │   ├── __init__.py
    │   │   ├── user.py
    │   │   ├── post.py
    │   │   └── ...
    │   ├── routes.py
    │   └── utils.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_user.py
    │   ├── test_post.py
    │   └── ...
    ├── config.py
    ├── requirements.txt
    └── run.py/main.py
1）resources/目录：存放RESTful API的资源文件，每个资源对应一个文件
2）utils.py：存放一些辅助函数或工具函数
3）tests/目录：存放测试代码
'''

'''
4、Flask应用的运行与调试
在开发过程中，我们经常需要运行和调试Flask应用。下面是一些常用的方法：
1）运行应用：执行启动脚本或使用命令行运行应用。例如，执行python run.py或flask run命令
2）调试模式：在开发阶段，启用调试模式可以帮助我们捕获和查看错误信息。在应用中设置app.debug = True，或使用FLASK_ENV=development环境变量
3）自动重载：在调试模式下，Flask应用会自动检测代码变化并进行重载，这样，在修改代码后，无需手动重启应用
4）调试器：如果应用发生错误，Flask会在浏览器中显示调试器页面，以帮助我们定位问题。调试器提供了异常追踪、变量查看等功能
5）日志：使用Flask的日志系统，可以记录应用的运行日志。通过设置日志级别和输出目标，我们可以控制日志的详细程度和输出位置
'''

