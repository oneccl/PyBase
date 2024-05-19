
# PyQuery库

# PyQuery库是一个非常强大灵活的HTML网页解析库，PyQuery是Python仿照JQuery的严格实现，语法几乎与JQuery相同

# 官方文档：https://pythonhosted.org/pyquery/

# 安装：pip install pyquery

# PyQuery可以将HTML字符串、HTML文件初始化为对象，也可以将请求的响应初始化为对象
# 如果字符串不是HTML格式，PyQuery会自动加上段落标签将字符串内容包装成HTML内容

from pyquery import PyQuery as pq

# 初始化字符串
doc = pq('html_str')

# 初始化HTML文件
html = pq(filename='html_path')

# 初始化请求响应
resp = pq(url='url')

# CSS选择器
print(doc('#id'))
print(doc('.class'))

# 查找
print(doc.find('#id'))

# 提取标签文本
print(doc('.class').text())
# 提取指定标签下的所有文本(包括子标签)
print(doc('标签名').text())
# 提取指定标签下的所有文本(包括子标签)，排除某些标签
print(doc('标签名').remove('排除标签名').text())


