# Python自动化-企业微信机器人

# 1、发送文本
import requests
import json

# 1）发送文本消息
# mentioned_list、mentioned_mobile_list：需要艾特哪些成员，user_id列表、手机号列表，艾特所有人：@all
def send_text(webhook, content, mentioned_list=None, mentioned_mobile_list=None):
    header = {
        "Content-Type": "application/json",
        "Charset": "UTF-8"
    }
    data = {
        "msgtype": "text",
        "text": {
            "content": content,
            "mentioned_list": mentioned_list,
            "mentioned_mobile_list": mentioned_mobile_list
        }
    }
    data = json.dumps(data)
    info = requests.post(url=webhook, data=data, headers=header)


# 2）发送Markdown消息
content = """
### **提醒！实时新增用户反馈**：<font color="warning">**123例**</font>\n
#### **请相关同事注意，及时跟进！**\n
[点击链接](https://work.weixin.qq.com/api/doc) \n
> 类型：<font color="info">用户反馈</font> \n
> 普通用户反馈：<font color="warning">117例</font> \n
> VIP用户反馈：<font color="warning">6例</font>
"""

'''目前只支持3种内置颜色
<font color="info">绿色</font>
<font color="comment">灰色</font>
<font color="warning">橙色</font>
'''

def send_md(webhook, content):
    header = {
        "Content-Type": "application/json",
        "Charset": "UTF-8"
    }
    data = {
        "msgtype": "markdown",
        "markdown": {
            "content": content
        }
    }
    data = json.dumps(data)
    info = requests.post(url=webhook, data=data, headers=header)


# 2、发送图文
# 一个图文消息支持1到8条图文

articles = [
    {
        "title": "xxx",        # 标题，不超过128个字节，超过会自动截断
        "description": "xxx",  # 描述，不超过512个字节，超过会自动截断
        "url": "url",          # 点击后跳转的链接
        "picurl": "img-url"    # 图文消息中的图片链接，支持JPG、PNG格式，较好的效果为大图1068*455，小图150*150
    }
]

def send_img_txt(webhook, articles):
    header = {
        "Content-Type": "application/json",
        "Charset": "UTF-8"
    }
    data = {
        "msgtype": "news",
        "news": {
            "articles": articles
        }
    }
    data = json.dumps(data)
    info = requests.post(url=webhook, data=data, headers=header)

# 3、发送文件
# 企业微信支持推送文件，首先将文件上传至企业微信指定的地址，然后返回media_id；文件应小于20M，且media_id有效时间为三天

def send_file(webhook, file):
    # 获取media_id
    key = webhook.split('key=')[1]
    id_url = f'https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type=file'
    files = {'file': open(file, 'rb')}
    res = requests.post(url=id_url, files=files)
    media_id = res.json()['media_id']
    # 发送文件
    header = {
        "Content-Type": "application/json",
        "Charset": "UTF-8"
    }
    data = {
        "msgtype": "file",
        "file": {
            "media_id": media_id
        }
    }
    # info = requests.post(url=webhook, json=data, headers=header) 或
    data = json.dumps(data)
    info = requests.post(url=webhook, data=data, headers=header)

key = 'robot_id'
webhook = rf'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}'
file = r'文件路径'
send_file(webhook, file)

