"""
Created with PyCharm.
Author: CC
E-mail: 203717588@qq.com
Date: 2024/1/2
Time: 21:42
Description:
"""

import matplotlib.pylab as plt
import numpy as np

# 定义二次函数
def f(x):
    return pow((x-5), 2) + 2

# 曲线：f(x)=x^2 -10x + 27
# 求导（切线斜率）：f(x)' = 2x - 10
# (1.5,14.25)  在x=1.5处的斜率：-7  y=-7x+b  b=24.75  y=-7x+24.75
# (2.5,8.25)   在x=2.5处的斜率：-5  y=-5x+b  b=20.75  y=-5x+20.75
# (3.5,4.25)

# 定义一次函数（切线）
def y1(x):
    return -7*x+24.75

def y2(x):
    return -5*x+20.75

# 设置画布大小
plt.figure(figure=(10, 5))

# 设置x轴的范围
x = np.linspace(0, 10, 100)

# 绘制二次曲线
plt.plot(x, f(x), color='red', linewidth=1)

# 绘制切线1
plt.plot(x, y1(x), color='green', ls='--', linewidth=1)
# 切点1
plt.scatter(1.5, 14.25, color="green")
# 标记虚线
plt.plot([0, 1.5], [14.25, 14.25], c='green', linestyle='--', linewidth=1)
plt.plot([1.5, 1.5], [0, 14.25], c='green', linestyle='--', linewidth=1)
# 对应x，y坐标标签
plt.text(1.4, -1.4, r'$\omega_1$')
plt.text(-1, 14, r'$L(\omega_1)$')
# 标注点A
plt.text(1.8, 14.4, 'A')

# 绘制切线2
plt.plot(x, y2(x), ls='--', linewidth=1)
# 切点2
plt.scatter(2.5, 8.25)
# 标记虚线
plt.plot([0, 2.5], [8.25, 8.25], c='#1f77b4', linestyle='--', linewidth=1)
plt.plot([2.5, 2.5], [0, 8.25], c='#1f77b4', linestyle='--', linewidth=1)
# 对应x，y坐标标签（Matplotlib支持LaTeX表达式）
plt.text(2.4, -1.4, r'$\omega_2$')
plt.text(-1, 8, r'$L(\omega_2)$')
# 标注点B
plt.text(2.8, 8.4, 'B')

# 再标注一个点C
plt.scatter(3.5, 4.25, color='blue')
plt.text(3.8, 4.4, 'C')

# x、y轴坐标限制（解决两个坐标轴0点不重合问题）
plt.xlim([0, 10])
plt.ylim([0, 25])

# 添加坐标轴标签
plt.xlabel(r'$\omega$', loc='right')
# rotation=0：设置y轴标签水平显示（默认90）
plt.ylabel(r'$L$', loc='top', rotation=0)
plt.show()


