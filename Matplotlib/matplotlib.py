
# Matplotlib数据可视化库

# Matplotlib可以用来绘制各种静态，动态，交互式的图表
# Matplotlib通常与NumPy和SciPy一起使用

# Pyplot是Matplotlib的子库，提供了和MATLAB类似的绘图API
# Pyplot是最常用的绘图模块，可以方便绘制2D图表
import matplotlib.pyplot as plt
import numpy as np

'''
常用pyplot函数：
plot()：用于绘制折线图
bar()：用于绘制垂直条形图和水平条形图
pie()：用于绘制饼图
scatter()：用于绘制散点图
hist()：用于绘制直方图
'''

# 1、Matplotlib折线图

# plot()方法：用于绘制折线图
# 语法：
'''
# 画单条线
plot(x, y, fmt, **kwargs)
# 画多条线
plot(x, y, fmt, x2, y2, fmt2, ..., **kwargs)
 - x, y：点或线的节点，x为x轴数据，y为y轴数据，数据可以是列表或数组
 - fmt：可选，定义基本格式（如颜色、标记和线条样式）
 - **kwargs：可选，用于二维平面图上，设置属性，如标签，线的宽度等
'''
# 正弦余弦图
x = np.arange(0, 4*np.pi, 0.1)     # arange(start,stop,step)
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y, x, z)
plt.show()

# 1）Matplotlib绘图标记
# ①、marker参数：使用plot()方法的marker参数来标记字符
x_points = np.array(range(6))
y_points = np.array([3, 8, 1, 10, 5, 7])
plt.plot(x_points, y_points, marker='o')
plt.show()
# ②、fmt参数：fmt参数定义了基本格式，如标记、线条样式和颜色
'''
fmt = '[marker][line][color]'
'''
x_points = np.array(range(6))
y_points = np.array([3, 8, 1, 10, 5, 7])
plt.plot(x_points, y_points, 'o:r')
plt.show()
'''
颜色字符：b蓝色，m品红，g绿色，y黄色，r红色，k黑色，w白色，c青绿，'#008000'RGB颜色符串(多条曲线不指定颜色时，会自动选择不同颜色)
线型参数：‐实线(solid)，‐‐破折线(dashed)，‐.点划线(dashdot)，:虚线(dotted)
标记字符：.点标记，,像素标记(极小点)，o实心圈标记，*五角星，v倒三角标记，^上三角标记，>右三角标记，<左三角标记等
'''
# ③、标记大小与颜色
'''
ms：定义标记的大小
mfc：定义标记内部的颜色
mec：定义标记边框的颜色
'''
y_points = np.array([6, 2, 13, 10])
plt.plot(y_points, marker='o', ms=15, mec='r', mfc='#4CAF50')
plt.show()
# annotate()方法：箭头注释
'''
plt.annotate('$文本内容$',xy=(标记点x,标记点y),xytext=(文本位置x,文本位置y)
'''

# 2）Matplotlib绘图线
# ①、自定义线的样式，包括线的类型、颜色和大小等
'''
- ls参数(linestyle)：线的类型
- c参数(color)：线的颜色
- lw参数(linewidth)：线的宽度
'''
y_points = np.array([6, 2, 13, 10])
plt.plot(y_points, ls='-.', c='SeaGreen', lw='2.0')
plt.show()

# ②、绘制多线条（图例）
'''
legend('图例名',loc,fontsize,shadow,labelspacing,handlelength)
 - '图例名'：若在plot()中添加label='图例名'，此处可省略
 - loc：图例位置：1或upper right、2或upper left、3或lower left、4或lower right
 - fontsize：字体大小，int或float类型
 - shadow: 图例边框是否添加阴影
 - labelspacing: 图例中条目间的距离
 - handlelength: 图例句柄的长度
'''
y1 = np.array([3, 7, 5, 9])
y2 = np.array([6, 2, 13, 10])
plt.plot(y1, label='y1')
plt.plot(y2, label='y2')
plt.legend(loc=1, labelspacing=1, handlelength=2, fontsize=14, shadow=True)
plt.show()

# 3）Matplotlib轴标签和标题

# ①、xlabel()、ylabel()方法：用于设置x轴、y轴标签
# ②、title()方法：用于设置标题
# 图形中文显示：Matplotlib默认不支持中文，解决：
# 方式1：
plt.rcParams["font.family"] = "SimHei"

x = np.arange(1, 11)
y = 2 * x + 5
plt.title("标题", fontproperties="SimHei")
# 方式2：fontproperties：设置中文字体；fontsize：设置字体大小；color：设置字体颜色
plt.xlabel("x轴", fontproperties="SimHei", fontsize=12)
plt.ylabel("y轴", fontproperties="SimHei", fontsize=12)
plt.plot(x, y)
plt.show()

# ③、标题与标签的定位：loc参数
'''
title(loc)：设置标题显示的位置: 'left', 'right', 和 'center'(默认)
xlabel(loc)：设置x轴显示的位置: 'left', 'right', 和 'center'(默认)
ylabel(loc)：设置y轴显示的位置: 'bottom', 'top', 和 'center'(默认)
'''

# 4）Matplotlib网格线与边框
# grid()方法：用于设置图表的网格线
'''
grid(b=None, axis='both')
b：布尔类型，是否显示网格线
axis：哪个方向的网格线，both(默认)、x轴或y轴
**kwargs：设置网格样式，如颜色color='r', 样式linestyle='-'和宽度linewidth=2
'''
# ax = plt.gca()：获取坐标轴信息
# ax.spines()：选择边框：top、right、bottom、left
# ax.set_color()：设置边框颜色，none表示不显示

# 5）Matplotlib画板（绘制多图）
# figure()方法：用于创建一个画板：
'''
fig = plt.figure(figsize, dpi)
 - figsize：画板大小（宽高），单位英寸
 - dpi：绘图对象的分辨率
 - facecolor：背景颜色
 - dgecolor：边框颜色
 - frameon：是否显示边框
'''
# 给画板添加标题：
'''
fig.suptitle("画板标题")
'''
# ①、绘制双轴图
# 创建图形对象，设置图形尺寸
fig = plt.figure(figsize=(10, 8), frameon=False)
# 添加子图区域：add_axes()的参数是一个序列，序列中4个数字分别对应图形的左侧，底部，宽度和高度
ax1 = fig.add_axes([0, 0, 1, 1])
# 数据
x = np.arange(1, 11)
# 绘制指数函数：y=exp(x)
ax1.plot(x, np.exp(x))
# 设置y轴标签名称
ax1.set_ylabel('exp')
# 添加双轴
ax2 = ax1.twinx()
# 绘制对数函数(红色实线圆点)：y=log(x)
ax2.plot(x, np.log(x), 'ro-')
# 设置y轴标签名称
ax2.set_ylabel('log')
# 添加图例
fig.legend(labels=('exp', 'log'), loc='upper left')
# 添加标题
plt.title('Double Axis')
plt.show()

# ②、subplot()方法：
'''
subplot(nrows, ncols, index, **kwargs)
- 将整个绘图区域分成nrows行和ncols列，从左到右、从上到下对每个子区域进行编号1-N，左上1、右下N，编号可通过index设置
'''
# 将图表绘制成1x2的图片区域
plt.rcParams["font.family"] = "SimHei"
# plot 1:
x_points = np.array([0, 6])
y_points = np.array([0, 100])
# 给定两个绘图区域
plt.subplot(1, 2, 1)
plt.plot(x_points, y_points)
plt.title("plot 1")
# plot 2:
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title("plot 2")

plt.suptitle("nrows*ncols为1x2的图片区域")
plt.show()

# ③、subplots()方法：
'''
subplots(nrows, ncols, sharex=False, sharey=False)
- sharex、sharey：设置x、y轴是否共享属性，可设置为none、all、row或col
'''
plt.rcParams["font.family"] = "SimHei"
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)
# 创建一个画板和两个子图，共享y轴
fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row')
# 在子图1上绘画
ax1.plot(x, y)
# 为子图1添加标题
ax1.set_title('ax1_title')
# 添加图例
ax1.legend('图1', loc=1, labelspacing=1, handlelength=2, fontsize=14, shadow=True)
# 在指定位置添加文本
ax1.text(1, 1, "text")
# 是否显示网格
ax1.grid(True)
# 在子图2上绘制散点图
ax2.scatter(x, y)
# 为子图2添加标题
ax2.set_title('ax2_title')
# 添加图例
ax2.legend('图2', loc=1, labelspacing=1, handlelength=2, fontsize=14, shadow=True)
# 添加画板标题
fig.suptitle("Sharing y axis")
plt.show()

# 2、Matplotlib柱形图
# bar()方法：用于绘制柱形图
'''
bar(x, y, height, width=0.8, align='center')
 - x：柱形图x轴数据，浮点型数组
 - height：柱形图的高度，浮点型数组
 - width：柱形图的宽度，浮点型数组
 - align：柱形图与x坐标的对齐方式
'''
x = np.array(["Bar-1", "Bar-2", "Bar-3", "Bar-4"])
y = np.array([12, 22, 6, 18])
plt.bar(x, y, width=0.4, color="#4CAF50")
plt.show()
# 垂直方向的柱状图
plt.barh(x, y, height=0.4)
plt.show()

# 3、Matplotlib饼图
# pie()方法：用于绘制饼图
'''
pie(x,labels,autopct,explode,colors)
 - x：绘制饼图的数据，浮点型数组或列表
 - explode：各个扇形间的间隔，默认0，数组
 - labels：各个扇形的标签，列表
 - colors：各个扇形的颜色，数组
 - autopct：设置各个扇形百分比显示格式：%d%%整数、%0.1f%%一位小数、%0.2f%%两位小数
 - radius：：设置饼图的半径，默认1
 - shadow：：设置饼图的阴影，布尔类型，默认False
 - startangle：设置饼图的起始角度，默认为从x轴正方向逆时针画起，如设定90则从y轴正方向画起
'''
# 数据
data = [15, 30, 45, 10]
# 饼图的标签
labels = ['A', 'B', 'C', 'D']
# 饼图的颜色
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
# 突出显示第二个扇形
explode = (0, 0.1, 0, 0)
# 绘制饼图
plt.pie(data, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
# 标题
plt.title("Pie Title")
# 显示图形
plt.show()

# 4、Matplotlib散点图
# scatter()方法：用于绘制散点图
'''
scatter(x, y, s, c, marker, cmap, alpha)
 - x，y：绘制散点图的数据点，长度相同的数组
 - s：点的大小，默认20，可以为数组
 - c：点的颜色，默认蓝色，也可为RGB或RGBA二维行数组
 - marker：点的样式，默认小圆圈o
 - cmap：设置颜色条
 - alpha：设置透明度，0-1之间，默认不透明
'''
# ①、绘制两组散点图
x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
plt.scatter(x, y, color='hotpink')

x = np.array([2, 2, 8, 1, 15, 8, 12, 9, 7, 3, 11, 4, 7, 14, 12])
y = np.array([100, 105, 84, 105, 90, 99, 90, 95, 94, 100, 79, 112, 91, 80, 85])
plt.scatter(x, y, color='#88c999')

plt.title("Scatter Title")
plt.show()

# ②、使用随机数绘制散点图
# 随机数生成器的种子
np.random.seed(19680801)

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
# 0-15 point
area = (30 * np.random.rand(N))**2
# 设置颜色及透明度
plt.scatter(x, y, s=area, c=colors, alpha=0.5)

plt.title("Scatter Title")
plt.show()

# savefig()方法：指定dpi(分辨率)保存绘图
# plt.savefig('img.jpg',dpi=600)

