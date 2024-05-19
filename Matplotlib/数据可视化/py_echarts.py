
# PyEcharts数据可视化

# ECharts是百度提供的基于JavaScript的开源可视化库，主要用于Web端数据可视化
# Echarts是通过JS实现的，PyEcharts则可以使用Python来调用里面的API

# PyEcharts特点：
"""
1）简洁的API设计，支持链式调用
2）丰富的图表，包括地图
3）支持主流Notebook环境，如JupyterLab
4）可集成Flask、Django等主流Web框架
"""
# PyEcharts官方网站：https://pyecharts.org/
# PyEcharts中文网站：https://pyecharts.org/#/zh-cn/

# 安装：pip install pyecharts

# 基本使用
from pyecharts.charts import Bar, Line, Pie, EffectScatter, Grid, WordCloud, Map
from pyecharts import options as opts
from pyecharts.globals import SymbolType
from pyecharts.faker import Faker

# 使用注意：render(path)在Python文件的同级目录下生成render.html文件，可以通过path参数指定HTML输出路径

x = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
data_china = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
data_russia = [1.6, 5.4, 9.3, 28.4, 22.7, 60.7, 162.6, 199.2, 56.7, 43.8, 3.0, 4.9]

# 1、柱状图
bar = Bar().\
    add_xaxis(x).\
    add_yaxis("China", data_china).\
    set_global_opts(title_opts=opts.TitleOpts(title="柱状图示例")).\
    render(r'C:\Users\cc\Desktop\bar.html')

# 多柱状图
bar_m = Bar().\
    add_xaxis(x).\
    add_yaxis("China", data_china).\
    add_yaxis("Russia", data_russia).\
    set_global_opts(title_opts=opts.TitleOpts(title="多柱状图示例")).\
    render(r'C:\Users\cc\Desktop\bar_m.html')

# 柱状图翻转
bar_t = Bar().\
    add_xaxis(x).\
    add_yaxis("China", data_china).\
    add_yaxis("Russia", data_russia).\
    reversal_axis().\
    set_series_opts(label_opts=opts.LabelOpts(position="right")).\
    set_global_opts(title_opts=opts.TitleOpts(title="柱状图翻转")).\
    render(r'C:\Users\cc\Desktop\bar_t.html')


# 2、折线图
line = Line().\
    add_xaxis(x).\
    add_yaxis("China", data_china).\
    set_global_opts(title_opts=opts.TitleOpts(title="折线图示例")).\
    render(r'C:\Users\cc\Desktop\line.html')

# 多折线图
line_m = Line().\
    add_xaxis(x).\
    add_yaxis("China", data_china).\
    add_yaxis("Russia", data_russia).\
    set_global_opts(title_opts=opts.TitleOpts(title="多折线图示例")).\
    render(r'C:\Users\cc\Desktop\line_m.html')

# 阶梯折线图
line_t = Line().\
    add_xaxis(x).\
    add_yaxis("China", data_china, is_step=True).\
    set_global_opts(title_opts=opts.TitleOpts(title="阶梯折线图")).\
    render(r'C:\Users\cc\Desktop\line_t.html')


# 3、饼图
pie = Pie().\
    add("", [list(z) for z in zip(x, data_china)]).\
    set_global_opts(title_opts=opts.TitleOpts(title="饼图示例")).\
    set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}")).\
    render(r'C:\Users\cc\Desktop\pie.html')

# 环状饼图
pie_c = Pie(init_opts=opts.InitOpts(width="600px", height="400px")).\
    add(
        series_name="环状饼图",
        data_pair=[list(z) for z in zip(x, data_china)],
        radius=["50%", "70%"],
        label_opts=opts.LabelOpts(is_show=False, position="center")
    ).\
    set_global_opts(legend_opts=opts.LegendOpts(pos_left="left", orient="vertical")).\
    set_series_opts(
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"),
        label_opts=opts.LabelOpts(formatter="{b}: {c}")
    ).\
    render(r'C:\Users\cc\Desktop\pie_c.html')


# 4、散点图
scatter = EffectScatter().\
    add_xaxis(x).\
    add_yaxis("China", data_china, symbol=SymbolType.ROUND_RECT).\
    set_global_opts(title_opts=opts.TitleOpts(title="散点图示例")).\
    render(r'C:\Users\cc\Desktop\scatter.html')


# 5、图表合并
# 例如：将柱状图和折线图放在一起
bar_m = Bar().\
    add_xaxis(x).\
    add_yaxis("China", data_china).\
    add_yaxis("Russia", data_russia).\
    set_global_opts(title_opts=opts.TitleOpts(title="多图合并"))

line_m = Line().\
    add_xaxis(x).\
    add_yaxis("蒸发量", [p + 50 for p in data_china])

bar_m.overlap(line_m)
grid = Grid()
grid.add(bar_m, opts.GridOpts(pos_left="5%", pos_right="5%"), is_control_axis_index=True)
grid.render(r'C:\Users\cc\Desktop\bar_line.html')


# 6、词云
# PyEcharts支持词云，更贴心的是中文也完全没有问题，不会出现乱码
# 例如词频统计结果：
data = [("生活资源", "999"), ("供热管理", "888"), ("供气质量", "777"), ("生活用水管理", "688"), ("一次供水问题", "588"),
        ("交通运输", "516"), ("城市交通", "515"), ("环境保护", "483"), ("房地产管理", "462"), ("城乡建设", "449"),
        ("社会保障与福利", "429"), ("社会保障", "407"), ("文体与教育管理", "406"), ("公共安全", "406"),
        ("公交运输管理", "386"), ("出租车运营管理", "385"), ("供热管理", "375"), ("市容环卫", "355"),
        ("自然资源管理", "355"), ("粉尘污染", "335"), ("噪声污染", "324"), ("土地资源管理", "304"),
        ("物业服务与管理", "304"), ("医疗卫生", "284"), ("粉煤灰污染", "284"), ("占道", "284"), ("供热发展", "254"),
        ("农村土地规划管理", "254"), ("生活噪音", "253"), ("供热单位影响", "253"), ("城市供电", "223"),
        ("房屋质量与安全", "223"), ("大气污染", "223"), ("房屋安全", "223"), ("文化活动", "223"), ("拆迁管理", "223"),
        ("公共设施", "223"), ("供气质量", "223"), ("供电管理", "223"), ("燃气管理", "152"), ("教育管理", "152"),
        ("医疗纠纷", "152"), ("执法监督", "152"), ("设备安全", "152"), ("政务建设", "152"), ("县区、开发区", "152"),
        ("宏观经济", "152"), ("教育管理", "112"), ("社会保障", "112"), ("生活用水管理", "112"),
        ("物业服务与管理", "112"), ("分类列表", "112"), ("农业生产", "112"), ("二次供水问题", "112"),
        ("城市公共设施", "92"), ("拆迁政策咨询", "92"), ("物业服务", "92"), ("物业管理", "92"),
        ("社会保障保险管理", "92"), ("低保管理", "92"), ("文娱市场管理", "72"), ("城市交通秩序管理", "72"),
        ("执法争议", "72"), ("商业烟尘污染", "72"), ("占道堆放", "71"), ("地上设施", "71"), ("水质", "71"),
        ("无水", "71"), ("供热单位影响", "71"), ("人行道管理", "71"), ("主网原因", "71"), ("集中供热", "71"),
        ("客运管理", "71"), ("国有公交（大巴）管理", "71"), ("工业粉尘污染", "71"), ("治安案件", "71"),
        ("压力容器安全", "71"), ("身份证管理", "71"), ("群众健身", "41"), ("工业排放污染", "41"),
        ("破坏森林资源", "41"), ("市场收费", "41"), ("生产资金", "41"), ("生产噪声", "41"), ("农村低保", "41"),
        ("劳动争议", "41"), ("劳动合同争议", "41"), ("劳动报酬与福利", "41"), ("医疗事故", "21"), ("停供", "21"),
        ("基础教育", "21"), ("职业教育", "21"), ("物业资质管理", "21"), ("拆迁补偿", "21"), ("设施维护", "21"),
        ("市场外溢", "11"), ("占道经营", "11"), ("树木管理", "11"), ("农村基础设施", "11"), ("无水", "11"),
        ("供气质量", "11"), ("停气", "11"), ("市政府工作部门（含部门管理机构、直属单位）", "11"), ("燃气管理", "11"),
        ("市容环卫", "11"), ("新闻传媒", "11"), ("人才招聘", "11"), ("市场环境", "11"), ("行政事业收费", "11"),
        ("食品安全与卫生", "11"), ("城市交通", "11"), ("房地产开发", "11"), ("房屋配套问题", "11"), ("物业服务", "11"),
        ("物业管理", "11"), ("占道", "11"), ("园林绿化", "11"), ("户籍管理及身份证", "11"), ("公交运输管理", "11"),
        ("公路（水路）交通", "11"), ("房屋与图纸不符", "11"), ("有线电视", "11"), ("社会治安", "11"), ("林业资源", "11"),
        ("其他行政事业收费", "11"), ("经营性收费", "11"), ("食品安全与卫生", "11"), ("体育活动", "11"),
        ("有线电视安装及调试维护", "11"), ("低保管理", "11"), ("劳动争议", "11"), ("社会福利及事务", "11"),
        ("一次供水问题", "11")]

# 内置词云图轮廓shape：circle、cardioid、diamond、triangle-forward、triangle、pentagon、star
# 可自定义图片轮廓：支持jpg、jpeg、png、ico格式，由mask_image参数指定路径（打开结果需要刷新）
wordcloud = WordCloud().\
    add(series_name="词云分析",
        data_pair=data,
        word_size_range=[6, 66],    # 字体大小范围
        mask_image=r'C:\Users\cc\Desktop\china_map.jpg',             # 自定义轮廓图
        textstyle_opts=opts.TextStyleOpts(font_family='cursive'),    # 字体格式
        word_gap=20,       # 单词间隔
        rotate_step=45,    # 旋转角度
        width='600',       # 宽
        height='512'       # 高
    ).\
    set_global_opts(
        title_opts=opts.TitleOpts(title="热点分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)),
        tooltip_opts=opts.TooltipOpts(is_show=True)
    ).\
    render(r'C:\Users\cc\Desktop\wordcloud.html')


# 7、地图
import random

# 中国地图
provinces = ['广东省', '湖北省', '上海市', '湖南省', '重庆市', '四川省', '新疆维吾尔自治区', '黑龙江省', '浙江省']
values = [random.randint(1, 1024) for p in provinces]

map = Map().\
    add("商家X", [z for z in zip(provinces, values)], "china").\
    set_global_opts(
        title_opts=opts.TitleOpts(title="地图示例"),
        visualmap_opts=opts.VisualMapOpts(max_=1024, is_piecewise=True)
    ).\
    render(r'C:\Users\cc\Desktop\map.html')

# 省地图
cities = ['西安市', '宝鸡市', '榆林市', '渭南市', '汉中市']
values = [random.randint(1, 1024) for c in cities]

map = Map().\
    add("商家Y", [z for z in zip(cities, values)], "陕西").\
    set_global_opts(
        title_opts=opts.TitleOpts(title="地图示例"),
        visualmap_opts=opts.VisualMapOpts(max_=1024, is_piecewise=True)
    ).\
    render(r'C:\Users\cc\Desktop\map_p.html')

