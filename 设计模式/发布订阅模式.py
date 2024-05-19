"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2023/9/21
Time: 23:07
Description:
"""

# Python发布订阅模式实现松耦合

# broadcast-service是一个轻量级的Python发布订阅者框架，且支持同步、异步、多主题订阅等不同场景下的模式建立
# 通过broadcast-service，只需要引入一个单例类，就可以十分轻松地构建起一个发布订阅者模式，几乎没有代码侵入性
# broadcast-service默认支持异步回调的方式，避免了线程阻塞的发生
# 主要特性：
'''
1）支持异步、同步、多主题订阅等不同的应用场景
2）提供lambda、回调函数、装饰器等不同的语法编写模式
3）支持同时订阅多个主题回调、同时向多个主题发布信息
'''
# 安装：pip install broadcast-service

# 1、基本使用：
from broadcast_service import broadcast_service
# 1）发布
# 不带参数：publish(topics, msgs)
# 带参数：broadcast(topics, msgs, params)
def publish(topics, msgs, *args):
    if not args:
        broadcast_service.publish(topics, msgs)
    else:
        broadcast_service.broadcast(topics, msgs, args)
# 2）订阅
# subscribe(topics, callback) 或 listen(topics, callback)
def subscribe(topics, func):
    broadcast_service.subscribe(topics, func)
    # broadcast_service.listen(topics, func)

# 回调处理
def callback_handle(msgs):
    print(f"Received: {msgs}  正在处理...")

if __name__ == '__main__':
    # 订阅主题（监听）
    subscribe('topic1', callback_handle)
    # 发布主题（广播）
    publish(['topic1', 'topic2'], 'Message')
'''
Received: Message  正在处理...
'''

# 2、使用装饰器
# 订阅主题（监听）及回调处理
@broadcast_service.on_listen('topicX')
def callback_subscribe_handle(msgs):
    print(f"Received: {msgs}  正在处理...")

if __name__ == '__main__':
    # 发布主题（广播）
    publish(['topicX', 'topicY'], 'MessageX')
'''
Received: MessageX  正在处理...
'''

# 3、发布Topic带参数
# 订阅主题（监听）及回调处理
@broadcast_service.on_listen('topicP')
def callback_subscribe_handle(msgs, args):
    print(f"Params: {args}")
    print(f"Received: {msgs}  正在处理...")

if __name__ == '__main__':
    # 发布主题（广播）
    publish('topicP', 'MessageP', 'id', 'info')
'''
Params: ('id', 'info')
Received: MessageP  正在处理...
'''

