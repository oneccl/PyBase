
# Flask路由与视图函数

"""
1、Flask路由机制
在Flask中，路由（Route）指的是将URL和对应的视图函数绑定在一起的机制。通过路由，我们可以定义不同
的URL路径，以及访问这些路径时执行的相应处理逻辑。Flask的路由机制是基于装饰器（Decorator）实现的
"""
from flask import Flask

app = Flask(__name__)

# 使用@app.route()装饰器定义两个路由

# @app.route('/')表示将根路径/和index()视图函数绑定在一起，当用户访问根路径时，会执行index()函数
@app.route('/')
def index():
    return "Hello, Flask！"

# @app.route('/about')表示将路径/about和about()视图函数绑定在一起，当用户访问/about时，会执行about()函数
@app.route('/about')
def about():
    return "About Page！"

if __name__ == '__main__':
    app.run()

'''
2、Flask视图函数基础
在Flask中，视图函数（View Function）是处理请求的核心部分。每个路由都对应一个视图
函数，当用户访问对应的URL路径时，Flask会调用相应的视图函数来处理请求，并返回响应
'''
"""
上述代码中，index()和about()都是视图函数，当用户访问根路径/时，Flask会调用index()并返回'Hello, Flask!'
视图函数可以根据实际需求，接受不同类型的请求参数，如URL参数、查询参数、表单数据等。Flask提供了多种方式来获取这些参数，并在视图函数中进行处理
"""

"""
3、Flask的URL构造与反转
在Flask中，URL构造和反转是指根据视图函数的名称和参数，生成对应的URL路径。Flask提供了便捷的方法来实现URL构造和反转
"""
from flask import Flask,url_for

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, Flask!'

@app.route('/user/<username>')
def user_profile(username):
    return f'Hello, {username}!'

if __name__ == '__main__':
    with app.test_request_context():
        # URL构造：url_for()函数用于生成对应视图函数的URL路径，可以通过指定视图函数的名称和参数来构造URL
        # url_for('index')生成了根路径'/'
        print(url_for('index'))    # '/'
        # url_for('user_profile', username='Alice')生成了路径'/user/Alice'
        print(url_for('user_profile', username='Alice'))   # 'user/Alice'


'''
4、Flask路由的高级用法
除了基本的路由定义外，Flask还提供了许多高级用法，用于更灵活和精确地定义路由
'''
"""
1）路由参数：可以在路由路径中定义参数，并在视图函数中接收和使用这些参数
例如，@app.route('/user/<username>')定义了一个带有参数的路由，username参数可以作为视图函数的参数进行处理
2）HTTP方法：通过指定请求方法，可以限制路由仅响应特定的HTTP方法
例如，@app.route('/', methods=['GET', 'POST'])表示该路由只会响应GET和POST请求
3）URL前缀：可以给一组相关的路由添加URL前缀，以便对它们进行分组和管理
例如，@app.route('/api/user/')定义了一个URL前缀为/api/user/的路由，后续的路由都可以在该前缀下进行定义
4）重定向：可以使用重定向函数redirect()将请求重定向到其他路由或URL
例如，return redirect(url_for('index'))将请求重定向到index()视图函数
5）错误处理：可以通过装饰器@app.errorhandler()来定义处理特定错误的视图函数
例如，@app.errorhandler(404)定义了一个处理404错误的视图函数
以上是一些Flask路由的高级用法示例，你可以根据实际需求来使用这些功能，并根据项目的复杂性进行更灵活的路由定义
"""
