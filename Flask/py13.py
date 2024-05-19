
# Flask表单处理与数据库操作

# Flask表单处理

"""
1、Flask表单处理
在Web应用中，表单是用户与应用进行交互的重要方式。Flask提供了丰富的功能来处理表单数据，包括接收、验证和处理用户提交的数据
"""
from flask import Flask, render_template, request

app = Flask(__name__)

# 定义一个路由/signup，允许GET和POST请求
@app.route('signup', methods=['GET', 'POST'])
def signup():
    # 当用户访问该路由时，如果是POST请求，则通过request.form获取表单数据，并进行处理
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # 处理表单数据...
        return 'Signup Successful!'
    # 如果是GET请求，则渲染一个包含表单的HTML模板signup.html
    return render_template('signup.html')

if __name__ == '__main__':
    app.run()

'''
2、Flask表单验证与CSRF保护
在处理用户提交的表单数据时，安全性是一个重要的考虑因素。Flask提供了表单验证和CSRF保护的功能，以确保数据的完整性和安全性
'''
"""
1）表单验证：Flask-WTF扩展提供了丰富的表单验证功能。通过定义表单类并添加验证器，可以对用户输入的数据进行验证，如字段是否为空、长度限制、数据格式等
2）CSRF保护：Flask自带的CSRF保护功能可以防止跨站请求伪造攻击。通过在表单中添加CSRF令牌，并在提交表单时验证令牌的有效性，可以确保请求来自应用的合法来源
"""
# 在Flask中使用Flask-WTF进行表单验证和CSRF保护
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    # 定义一个登录表单LoginForm，并为字段添加验证器
    form = LoginForm()
    # 在登录视图函数中，创建一个表单实例，并使用form.validate_on_submit()方法来验证表单数据
    # 如果验证成功，我们可以获取表单字段的数据进行处理
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        # 验证表单数据...
        return 'Login Successful!'
    return render_template('login.html', form=form)

if __name__ == '__main__':
    app.run()


# Flask数据库操作

'''
3、Flask与SQLAlchemy集成
在许多Web应用中，数据库是必不可少的组成部分。Flask与SQLAlchemy集成，可以方便地进行数据库操作，包括创建表、插入数据、查询数据等
'''
# 在Flask中使用SQLAlchemy进行数据库操作：
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# 配置SQLAlchemy的数据库URI，指定SQLite数据库的位置
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)

# 定义一个数据库模型User，该模型映射到数据库中的一张表
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)

# 在list_users()视图函数中，使用User.query.all()查询所有用户，并返回结果
@app.route('/users')
def list_users():
    users = User.query.all()
    return 'Users: ' + ', '.join([user.username for user in users])

if __name__ == '__main__':
    app.run()

'''
4、Flask数据库操作与Migrate
在开发过程中，数据库的结构和数据可能会发生变化。Flask-Migrate扩展提供了数据库迁移的功能，可以方便地管理数据库模式的变更
'''
# 使用Flask-Migrate进行数据库迁移：
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)
# 创建一个数据库迁移对象migrate，并将其与应用和数据库关联起来
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)

if __name__ == '__main__':
    app.run()

# 通过使用命令行工具，可以执行数据库迁移命令来创建或更新数据库的结构

