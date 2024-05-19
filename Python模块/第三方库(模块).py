"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/9/9
Time: 16:11
Description:
"""

# 1、smtplib/zmail 发送邮件

# 1）smtplib库
# 1.1）登录SMTP服务器
'''
smtplib.SMTP(host,port,timeout)：默认端口25，数据是明文传输，没有加密
smtplib.SMTP_SSL(host,port,timeout)：SSL(Secure Socket Layer)，默认端口465
 - host：smtp服务器主机名
 - port：smtp服务的端口，可省略
若在创建SMTP邮件对象时传递了这两个参数，在初始化时会自动调用connect方法连接服务器
'''
# 1.2）发送邮件
'''
smtp.sendmail(from_addr, [to_addrs, cc_addrs], msg)
 - from_addr：发件人邮箱
 - to_addrs：收件人邮箱，多个Email地址，用逗号隔开
 - cc_addrs：抄送列表，多个Email地址，用逗号隔开
 - msg：邮件内容
'''
# 1.3）解析转换
'''
1）parseaddr：将纯Email地址进行编码得到name和Email地址的字符串
to_addrs = 'name<addr>'
name, addr = parseaddr(to_addrs)
2）formataddr：和parseaddr相反（name需要使用Header函数进行编码）
to_addrs = formataddr((Header(name,'utf-8').encode(), addr))
'''
# 常用模块
import smtplib
from email.header import Header                      # 用来对Email主题进行编码
from email.mime.text import MIMEText                 # 用来构造文本
from email.mime.image import MIMEImage               # 用来构造图片
from email.mime.multipart import MIMEMultipart       # 用来构造多个对象，如文本、图片、文件等
from email.mime.application import MIMEApplication   # 多种附件类型使用
from email.utils import formataddr

from_addr = 'xxx@qq.com'             # 发件人邮箱账号
password = 'xxx'                     # 发件人邮箱密码(授权码)：获取步骤：登录QQ邮箱->帮助中心->客户端设置->问题2
to_addrs = 'xxx@qq.com'              # 收件人邮箱账号，多个Email地址，用逗号隔开
cc_addrs = 'xxx@qq.com'              # 抄送列表，多个Email地址，用逗号隔开
smtp_server = 'smtp.qq.com'          # SMTP服务器地址，smtp.163.com(163邮箱)、smtp.qq.com(qq邮箱)

# 文本邮件
def mail(subject, message):
    # 创建邮件内容对象，参数对应内容、类型(plain文本、html超文本、邮件对象.attach(附件))、编码
    msg = MIMEText(message, 'plain', 'utf-8')
    # 邮件主题，将邮件标题转换成标准Email格式
    msg['Subject'] = Header(subject, 'utf-8').encode()
    # 发件人邮箱昵称、发件人邮箱账号
    msg['From'] = formataddr((Header('name', 'utf-8').encode(), from_addr))
    # 收件人邮箱昵称、收件人邮箱账号（多个Email地址，用逗号隔开）
    msg['To'] = formataddr((Header('name', 'utf-8').encode(), to_addrs))
    # 抄送列表（多个Email地址，用逗号隔开）
    msg['Cc'] = formataddr((Header('name', 'utf-8').encode(), cc_addrs))

    try:
        # 通过SMTP_SSL登录SMTP服务器，创建SMTP邮件对象
        server = smtplib.SMTP_SSL(smtp_server, 465)
        # 使用授权码登录邮箱
        server.login(from_addr, password)
        # 发送邮件：as_string()会将整个Email内容转换为字符串
        server.sendmail(from_addr, [to_addrs, cc_addrs], msg.as_string())

        # 关闭连接
        server.quit()
        print('发送成功！')
    except Exception as e:
        print(f'发送失败: {e}')

# 图片附件邮件
def mail(subject, message):
    # 构建MIMEMultipart对象，可以往里面添加文本、图片、文件等
    msg = MIMEMultipart()
    # 邮件主题，将邮件标题转换成标准Email格式
    msg['Subject'] = Header(subject, 'utf-8').encode()
    # 发件人邮箱昵称、发件人邮箱账号
    msg['From'] = formataddr((Header('name', 'utf-8').encode(), from_addr))
    # 收件人邮箱昵称、收件人邮箱账号（多个Email地址，用逗号隔开）
    msg['To'] = formataddr((Header('name', 'utf-8').encode(), to_addrs))
    # 邮件正文内容
    msg.attach(MIMEText(message, 'plain', 'utf-8'))

    try:
        # 设置图片信息
        image = MIMEImage(open(r'../../x.png', 'rb').read())
        image.add_header('Content-Disposition', 'attachment', filename="x.png")
        msg.attach(image)

        # 通过SMTP_SSL登录SMTP服务器，创建SMTP邮件对象
        server = smtplib.SMTP_SSL(smtp_server, 465)
        # 使用授权码登录邮箱
        server.login(from_addr, password)
        # 发送邮件：as_string()会将整个Email内容转换为字符串
        server.sendmail(from_addr, to_addrs, msg.as_string())

        # 关闭连接
        server.quit()
        print('发送成功！')
    except Exception as e:
        print(f'发送失败: {e}')

# 正文带图片的邮件
'''
1）将MIMEMultipart中的MIMEText类型改为html
2）正文使用HTML如<p><img src="cid:0"></p>，cid:n会将附件作为图片引用，多个图片依次编号
'''

# 文件附件邮件
def mail(subject, message):
    # 构建MIMEMultipart对象，可以往里面添加文本、图片、文件等
    msg = MIMEMultipart()
    # 邮件主题，将邮件标题转换成标准Email格式
    msg['Subject'] = Header(subject, 'utf-8').encode()
    # 发件人邮箱昵称、发件人邮箱账号
    msg['From'] = formataddr((Header('name', 'utf-8').encode(), from_addr))
    # 收件人邮箱昵称、收件人邮箱账号（多个Email地址，用逗号隔开）
    msg['To'] = formataddr((Header('name', 'utf-8').encode(), to_addrs))
    # 邮件正文内容
    msg.attach(MIMEText(message, 'plain', 'utf-8'))

    try:
        # 设置文件信息
        file = MIMEText(open(r'../../x.xlsx', 'rb').read(), 'base64', 'utf-8')
        file.add_header('Content-Disposition', 'attachment', filename="x.xlsx")
        msg.attach(file)

        # 通过SMTP_SSL登录SMTP服务器，创建SMTP邮件对象
        server = smtplib.SMTP_SSL(smtp_server, 465)
        # 使用授权码登录邮箱
        server.login(from_addr, password)
        # 发送邮件：as_string()会将整个Email内容转换为字符串
        server.sendmail(from_addr, to_addrs, msg.as_string())

        # 关闭连接
        server.quit()
        print('发送成功！')
    except Exception as e:
        print(f'发送失败: {e}')

# 包含多种附件类型的邮件
def mail(subject, message):
    # 构建MIMEMultipart对象，可以往里面添加文本、图片、文件等
    msg = MIMEMultipart()
    # 邮件主题，将邮件标题转换成标准Email格式
    msg['Subject'] = Header(subject, 'utf-8').encode()
    # 发件人邮箱昵称、发件人邮箱账号
    msg['From'] = formataddr((Header('name', 'utf-8').encode(), from_addr))
    # 收件人邮箱昵称、收件人邮箱账号（多个Email地址，用逗号隔开）
    msg['To'] = formataddr((Header('name', 'utf-8').encode(), to_addrs))
    # 邮件正文内容
    msg.attach(MIMEText(message, 'plain', 'utf-8'))

    try:
        # 附件
        # png类型附件（图片）
        part = MIMEApplication(open(r'../../x.png', 'rb').read())
        # Content-Disposition是MIME协议的扩展，表示如何显示附件
        # inline：将文件内容直接显示在页面
        # attachment：弹出对话框让用户下载
        part.add_header('Content-Disposition', 'attachment', filename="x.png")
        msg.attach(part)

        # xlsx类型附件（文件）
        part = MIMEApplication(open(r'../../x.xlsx', 'rb').read())
        part.add_header('Content-Disposition', 'attachment', filename="x.xlsx")
        msg.attach(part)

        # 通过SMTP_SSL登录SMTP服务器，创建SMTP邮件对象
        server = smtplib.SMTP_SSL(smtp_server, 465)
        # 使用授权码登录邮箱
        server.login(from_addr, password)
        # 发送邮件：as_string()会将整个Email内容转换为字符串
        server.sendmail(from_addr, to_addrs, msg.as_string())

        # 关闭连接
        server.quit()
        print('发送成功！')
    except Exception as e:
        print(f'发送失败: {e}')


# 2）zmail库（只支持python3）
# 优点：
'''
1）自动查找服务器地址及其端口
2）自动使用合适的协议登录
3）自动将python字典转换为MIME对象（带附件）
4）自动添加邮件标题和本地名称，以避免服务器拒绝您的邮件
5）轻松自定义邮件标题
6）支持HTML作为邮件内容
'''
import zmail

from_addr = 'xxx@qq.com'
password = 'xxx'
to_addrs = 'xxx@qq.com'
cc_addrs = 'xxx@qq.com'
smtp_server = 'smtp.qq.com'
file = '../../文件名.suffix'

def mail(subject, message):
    mail_info = {
        'subject': subject,
        'content_text': message,
        # 当content_text和content_html同时出现时，只显示content_html
        # 'content_html': 'message：邮件正文带图片',
        # 附件，str或list类型
        'attachments': file,
    }

    try:
        server = zmail.server(username=from_addr, password=password, smtp_host=smtp_server, smtp_port=465)
        server.send_mail(
            recipients=to_addrs,
            mail=mail_info,
            # 抄送列表，str或list类型
            cc=cc_addrs
        )
        print('发送成功！')
    except Exception as e:
        print(f'发送失败: {e}')

# 2、Yaml模块

# YAML不是一种标记语言，而是一种易读的序列化语言
# Yaml通常被用作配置文件，后缀是.yaml或.yml；主要用于数据存储与传输
# Python的PyYAML模块是Python的YAML解析器和生成器

# 安装：pip install pyyaml

import yaml
# Python处理yaml文件的方法
from yaml import load, dump
# LibYAML的解析器和生成器：CParser和CEmitter类
from yaml import CLoader, CDumper
# YAML加载器类型
'''
BaseLoader：仅加载最基本的YAML
SafeLoader：安全地加载YAML语言的子集，用于加载不受信任的输入（safe_load）
FullLoader（默认）：加载完整YAML语言，避免任意代码执行（full_load）
UnsafeLoader：也称为Loader向后兼容性，原始的Loader代码，不受信任的数据可能通过这种方式执行其他有危害的代码
'''

# YAML支持的数据类型（对象：键值对的集合、数组、纯量（Scalars））
# YAML的基本语法（大小写敏感、使用缩进表示层级关系、字符串不使用引号包裹、对象通过":"表示、列表通过"-"表示、注释通过"#"表示）

# Python处理Yaml文件：
"""
load(doc/stream, Loader)：用于将YAML文档反序列化转化成Python对象
load_all(doc/stream, Loader)：用于解析多个YAML文档，返回包含所有反序列化后的YAML文档的生成器对象
"""
'''
dump(data, stream, Dumper)：用于将Python对象转换成一个YAML文档，若无其他参数，直接返回生成的YAML文档
dump_all(data, stream, Dumper)：用于将Python对象转换成多个YAML文档
'''
# 1）基本使用：load()、dump()
doc = """
a: 1
b:
  c: 2
  d: 3
"""
print(yaml.load(doc, Loader=CLoader))     # {'a': 1, 'b': {'c': 2, 'd': 3}}
print(yaml.dump(yaml.load(doc, Loader=CLoader)))
'''
a: 1
b:
  c: 2
  d: 3
'''

with open('test.yaml', 'r', encoding='utf-8') as f:
    print(yaml.load(f, Loader=CLoader))   # {'a': 1, 'b': {'c': 2, 'd': 3}}

with open('test_write.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(doc, f, Dumper=CDumper)     # {'a': 1, 'b': {'c': 2, 'd': 3}}

# 2）load_all()、dump_all()
doc = """
name: Tom
age: 18
---
name: Jerry
age: 17
"""
datas = yaml.load_all(doc, Loader=CLoader)
for data in datas:
    print(data)
'''
{'name': 'Tom', 'age': 18}
{'name': 'Jerry', 'age': 17}
'''
# with open('test_write.yaml', 'w', encoding='utf-8') as f:
#     yaml.dump_all(doc, f, Dumper=CDumper)

# 3）复杂YAML文档解析
doc = """
- A: 1
  a: 2
- B: 3
  b: 4
"""
datas = yaml.load(doc, Loader=CLoader)
print(datas)
'''
[{'A': 1, 'a': 2}, {'B': 3, 'b': 4}]
'''

doc = """
name:
 - A1
 - B2
 - C3
age: 20
addrs:
  addr1: M
  addr2: N
"""
datas = yaml.load(doc, Loader=CLoader)
print(datas)
'''
{'name': ['A1', 'B2', 'C3'], 'age': 20, 'addrs': {'addr1': 'M', 'addr2': 'N'}}
'''
# with open('test_write.yaml', 'w', encoding='utf-8') as f:
#     yaml.dump(doc, f, Dumper=CDumper)



