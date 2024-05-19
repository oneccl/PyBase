
# pandas数据可视化

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 使用DataFrame对象的plot()方法可以方便的进行图表可视化
'''
plot()参数：
 - x：指定数据框的列作为x轴的值，若不指定，默认使用数据框的索引
 - y：指定数据框的列作为y轴的值，若不指定，默认使用数据框所有列
 - kind：图表类型：如line、bar, barh, pie, hist, scatter等，str类型
 - style：图表样式，list或dict类型
 - title：图表标题，str或list(子图标题)类型
 - alpha：图表透明度
 - legend：是否显示图例，默认显示
 - label：图例名称
 - grid：是否显示网格线，默认关闭
 - xticks、yticks：设置x轴、y轴刻度值，序列类型
 - xlimit、ylimit：设置轴界限，列表或元组（区间范围）类型
 - rot：旋转刻度标签的角度
 - use_index：是否使用数据框的索引用于x轴刻度标签，默认使用
 - logx、logy：是否在x轴或y轴上使用对数标尺
 - subplots：是否使用子图，将各个列单独绘制到subplot中
 - figsize：图表大小
 - layout：子图布局
 - sharex、sharey：是否共享x或y轴
 - sort_columns：以字母顺序绘制各列，默认原顺序
 - fontsize：刻度轴字体大小
 - position：条形图的对齐方式，取值范围[0,1]
 - table：是否图表下面显示df表
'''
# 1、折线图：df.plot()
df = pd.DataFrame(np.random.randn(10, 4), index=pd.date_range('1/1/2022', periods=10), columns=list('ABCD'))
ax = df.plot(title='XXXXX', rot=30)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
plt.show()

# 2、条形图/柱状图：df.plot.bar()
df = pd.DataFrame(np.random.rand(10, 4), columns=['A', 'B', 'C', 'D'])
df.plot.bar()
plt.show()

# 3、堆叠柱状图：df.plot.bar(stacked=True)
df = pd.DataFrame(np.random.rand(10, 4), columns=['A', 'B', 'C', 'D'])
df.plot.bar(stacked=True)
plt.show()

# 4、水平条形图/柱状图：df.plot.barh()
df = pd.DataFrame(np.random.rand(10, 4), columns=['A', 'B', 'C', 'D'])
df.plot.barh()
plt.show()

# 5、饼图：df.plot.pie()
df = pd.DataFrame(3 * np.random.rand(4), index=['A', 'B', 'C', 'D'], columns=['x'])
df.plot.pie(subplots=True)
plt.show()

# 6、散点图：df.plot.scatter()
df = pd.DataFrame(np.random.rand(50, 4), columns=['A', 'B', 'C', 'D'])
df.plot.scatter(x='A', y='B')
plt.show()

# 7、直方图：df.plot.hist()
df = pd.DataFrame({'A': np.random.randn(1000) + 1,
                   'B': np.random.randn(1000),
                   'C': np.random.randn(1000) - 1,
                   'D': np.random.randn(1000) - 2
                   }, columns=['A', 'B', 'C', 'D'])
# df.plot.hist(bins=20)        # 所有列绘制在一起
df.diff().hist(bins=20)      # 为每列绘制不同的直方图
plt.show()

# 8、箱体图：series.plot.box()、df.plot.box()、df.boxplot()
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box()
plt.show()

# 9、面积图：series.plot.area()、df.plot.area()
df = pd.DataFrame(np.random.rand(10, 4), columns=['A', 'B', 'C', 'D'])
df.plot.area()
plt.show()

# 10、绘制多图
df = pd.DataFrame(np.random.randn(10, 4), index=pd.date_range('1/1/2022', periods=10), columns=list('ABCD'))
df.plot(subplots=True, layout=(2, 2), figsize=(20, 10))
plt.show()

# 示例：
fig = plt.figure(figsize=(10, 3))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
df = pd.DataFrame(np.random.randn(10, 4).cumsum(0), index=np.arange(0, 100, 10), columns=['A', 'B', 'C', 'D'])
df['A'].plot(ax=ax1, style='ro-', rot=60, xlim=(0, 100), label='折线图', legend=True)
df['B'].plot(ax=ax2, color='b', kind='bar', label='柱形图', legend=True)
plt.show()

