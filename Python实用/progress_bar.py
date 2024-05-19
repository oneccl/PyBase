
# 转圈
import time

def circle():
    ls = ["|", "/", "—", "\\"]
    while True:
        for i in ls:
            # Python3默认print()换行参数end='\n'
            # \r：将光标的位置回退到本行的开头位置，后面print函数打印的内容会替换上一次print函数的打印内容
            print('\r正在加载中 %s ' % i, end='')
            time.sleep(0.25)

# circle()

# 动态表情
def emoji():
    list1 = ['(づ｡◕ᴗᴗ◕｡)づ', '(づ｡—ᴗᴗ—｡)づ']
    list2 = ['u~(@_@)~*', 'u~(—_—)~*']
    while True:
        for a, b in zip(list1, list2):
            print('\r %s %s ' % (a, b), end='')
            time.sleep(0.20)

# emoji()

# 进度条
# 普通进度条
def progress_bar():
    t = 50
    print("开始执行".center(t // 2, "-"))
    for i in range(t + 1):
        # print('\r进度: [%-50s] %.0f%%' % ('#' * i, i * 2), end='')
        # 带颜色的进度条 \033[32m开始 \033[0m结束
        print('\r进度: [\033[32m%-50s\033[0m] \033[32m%.0f%%\033[0m' % ('#' * i, i * 2), end='')
        time.sleep(0.1)
    print("\n" + "执行结束".center(t // 2, "-"))

# progress_bar()

# %ns 字符串前面填充n个空格
print("%10s" % 'A')
# %-ns 字符串后面填充n个空格
print("%-10s" % 'A')

# 5舍6入
print("%.0f" % 10.6)      # 11
print("%.0f" % 10.5)      # 10
# 字符串格式化添加%：%%
print("%.0f%%" % 10.5)    # 10%


# 带时间的进度条
def progress_bar():
    t = 50
    # n // 2：去除除完结果的小数部分
    print("开始执行".center(t // 2, "-"))
    # 返回性能计数器的值（单位：秒）
    start = time.perf_counter()
    for i in range(t + 1):
        f = "#" * i
        o = "." * (t - i)
        p = (i / t) * 100
        d = time.perf_counter() - start
        # :引导符号  ^居中对齐  3输出宽度  .nf输出精度
        # print("\r{:^3.0f}% [{}{}] {:^5} {:.2f}s".format(p, f, o, f"{i}/{t}", d), end="")
        # 带颜色的进度条 \033[32m开始 \033[0m结束
        print("\r{:^3.0f}% [\033[32m{}{}\033[0m] {:^5} \033[32m{:.2f}s\033[0m".format(p, f, o, f"{i}/{t}", d), end="")
        time.sleep(0.1)
    print("\n"+"执行结束".center(t // 2, "-"))

# progress_bar()


import math

# 自定义进度条
def progress_bar(bar_sum: int, bar_len=75):
    mid = bar_sum / bar_len
    print("开始执行".center(bar_len // 2, "-"))
    start = time.perf_counter()
    for i in range(bar_sum + 1):
        c = math.floor(i / mid)
        f = "#" * c
        o = "." * (bar_len - c)
        p = (i / bar_sum) * 100
        d = time.perf_counter() - start
        print("\r{:^3.0f}% [\033[32m{}{}\033[0m] {:^5} \033[32m{:.2f}s\033[0m".format(p, f, o, f"{i}/{bar_sum}", d), end="")
        time.sleep(0.01)
    print("\n" + "执行结束".center(bar_len // 2, "-"))

progress_bar(1024)

# while定时器
import time

count = 100
sta_time = 0

while count > 0:
    count -= 1
    time.sleep(0.1)
    sta_time += 0.1
    print(f"\r正在执行...{round(sta_time, 1)}s", end='')




