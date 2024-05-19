
# Python网络编程

"""
1、网络编程基础概念
网络编程涉及到计算机之间的数据传输和通信，而网络是由多台计算机和设备通过通信链路连接在一起的
1）协议（Protocol）：网络通信中使用的规则和约定。常见的网络协议有TCP、UDP、HTTP等
2）IP地址（IP Address）：用于唯一标识网络中的计算机或设备的地址。IPv4和IPv6是目前广泛使用的IP地址版本
3）端口（Port）：用于标识网络应用程序的地址。一个IP地址可以有多个端口，每个端口对应一个特定的应用程序
4）套接字（Socket）：在网络编程中，套接字是用于实现网络通信的一种编程接口。通过套接字，可以进行数据的发送和接收
"""

'''
2、Python网络编程库介绍
在Python中，有许多网络编程库可供选择，用于简化网络编程的过程；常用的Python网络编程库：
1）socket：Python的内置模块，提供了对底层网络通信的支持。它允许你创建套接字对象，进行数据的发送和接收，并处理网络连接的建立和关闭
2）requests：一个简洁而强大的HTTP库，用于发送HTTP请求和处理响应。它可以用于构建基于HTTP协议的网络应用程序
3）urllib：另一个用于发送HTTP请求的库，它提供了一些更低级别的接口，可以对URL进行编码和解码，发送请求并处理响应
4）asyncio：一个用于编写异步代码的库，提供了在单线程中实现并发的能力。它可以用于编写高性能的网络应用程序
'''

'''
3、Python socket编程
socket模块提供了网络编程所需的套接字接口。使用socket，你可以创建套接字对象，进行数据的发送和接收，以及处理网络连接的建立和关闭
'''
import socket

# 创建套接字对象：指定地址族为AF_INET，表示使用IPv4地址；
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 建立与服务器的连接
s.connect(('example.com', 80))

# 发送数据到服务器
s.send(b'Hello, Server！')

# 接收服务器返回的数据
data = s.recv(1024)
print(data.decode())

# 关闭连接
s.close()

'''
4、网络编程应用案例（基于TCP的简单聊天室）
网络编程在实际应用中有着广泛的用途，例如开发基于客户端-服务器模型的网络应用程序，实现远程控制和数据交换等功能
'''
import socket
import threading

# 处理客户端连接
def handle_client(client_socket):
    # 循环不断接收客户端发送的消息，并向客户端发送响应
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        message = data.decode()
        print('Received message:', message)
        response = 'Server received your message: ' + message
        client_socket.send(response.encode())
    client_socket.close()

# 启动服务器
def start_server():
    # 创建一个套接字对象server_socket，绑定地址为localhost:8888
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 8888))
    # 监听客户端连接
    server_socket.listen(5)
    print('Server started. Listening on port 8888...')

    # 当有新的客户端连接时，创建一个新的线程来处理该连接，避免阻塞主线程
    while True:
        client_socket, addr = server_socket.accept()
        print('New connection:', addr)
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

start_server()

