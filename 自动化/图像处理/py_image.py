
# Python图像处理

# PIL（Python Image Library）是Python提供的图像处理标准库，来满足开发者处理图像的各种功能，Python2.7后不再支持
# PIL支持的图像文件格式包括JPEG、PNG、GIF等，它提供了图像创建、图像显示、图像处理等功能
# Pillow是基于PIL模块Fork的一个派生分支，如今已经发展成为比PIL本身更具活力的图像处理库，Pillow支持python3
# Pillow库是一个非常强大的基础图像处理库，是计算机图像识别的基础，主要模块有Image模块、ImageChops通道操作模块）、ImageColor颜色转换模块、ImageDraw二维图形模块等
# Pillow参考学习网站：https://www.osgeo.cn/pillow/reference/

# 安装：pip install Pillow==9.5.0

# 1、基本概念

# 图像的基本概念：深度（depth）、通道（bands）、模式（mode）、坐标系（coordinate system）等
# 1）深度（depth）
# 图像中像素点占得Bit位数
# 二值图像：图像的像素点不是0就是1（图像不是黑色就是白色），图像像素点占的位数是1位，图像的深度是1，也称位图
# 灰度图像：图像的像素点位于0-255之间（0代表全黑，255代表全白），在0-255之间插入了2^8=255个等级的灰度，图像像素点占的位数是8位，图像的深度是8
# 2）通道（bands）
# 每张图像都是由一个或多个数据通道构成
# RGB是基本的三原色（红色、绿色和蓝色），如果用8位代表一种颜色，那么每种颜色的最大值是255，这样每个像素点的颜色值范围就是(0-255,0-255,0-255)，图像的通道是3，而灰度图像的通道是1
# 3）模式（mode）
# 图像实际上是像素数据的矩形图，图像的模式定义了图像中像素的类型和深度
# 常见的模式有：
"""
1：1位像素，表示黑和白，占8bit像素，在图像表示中称为位图
L：表示黑白之间的灰度，占8bit像素
P：8bit像素，使用调色版映射
RGB：真彩色，占用3x8位像素，其中R为红色，G为绿色，B为蓝色，三原色叠加形成的色彩变化，如三通道都为0则代表黑色，都为255则代表白色
RGBA：带透明蒙版的真彩色，其中的A为alpha透明度，占用4x8位像素
"""
# 4）坐标系（coordinate system）
# PIL中图像的坐标是从左上角开始，向右下角延伸，以二元组(x，y)的形式传递，x轴从左到右，y轴从上到下，即左上角的坐标为(0, 0)
# 因此矩形图像使用四元组表示，例如一个450x450像素的矩形图像可以表示为(0, 0, 450, 450)

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import requests

# 2、图像常用操作

# 1、打开图像
# Image.open(file,mode)：mode默认r
# 1）打开图像文件
img1 = Image.open(r'C:\Users\cc\Desktop\cat.png')
# 2）从文件流中打开图像
resp = requests.get(r'http://f.hiphotos.baidu.com/image/pic/item/b151f8198618367aa7f3cc7424738bd4b31ce525.jpg')
img2 = Image.open(BytesIO(resp.content))

# 展示图像
# img1.show()
# img2.show()

# 图像旋转(逆时针)
# img1.rotate(90).show()

# 2、创建图像
# Image.new(mode,size,color)
# mode:模式  size:大小(宽高)，二元组类型，单位像素  color:颜色，可以使用颜色名、16进制或RGB数字
img3 = Image.new('RGB', (450, 450), (255, 0, 0))
# img3.show()

# 3、转换格式（保存图像）
# Image.save(file)
# 查看图像类型格式
print(img1.format)
# 另存为JPG类型的图片
img1.save(r'C:\Users\cc\Desktop\cat_f.jpg')
# 打开保存的图片
img4 = Image.open(r'C:\Users\cc\Desktop\cat_f.jpg')
# 查看图像类型格式
print(img4.format)

# 4、创建缩略图（图像缩放）
# mage.thumbnail(size, resample=3)
# 修改当前图像制作成缩略图，该缩略图尺寸不大于给定的图像尺寸
# size:最终图像大小，元组类型  resample:过滤器，滤波器

# 缩放图像
img4.thumbnail((128, 128), Image.LANCZOS)
# 展示图像
# img4.show()

# 5、融合图像
# Image.blend(image1, image2, alpha)
# 将图像image1和图像image2根据alpha值进行融合，公式为：out = image1 * (1.0 - alpha) + image2 * alpha
# image1和image2表示两个大小和模式相同的图像，alpha介于0和1之间；若alpha为0，则返回image1图像，若alpha为1，则返回image2图像

# 蓝色图像
image1 = Image.new('RGB', (128, 128), (0, 0, 255))
# 红色图像
image2 = Image.new('RGB', (128, 128), (255, 0, 0))
# 取中间值
img5 = Image.blend(image1, image2, 0.5)
# image1.show()
# image2.show()
# 显示紫色图像
# img5.show()

# 6、像素点处理
# Image.eval(image, *args)
# image:需要处理的图像对象  *args:函数对象，有一个整数作为参数
# 若变量image图像有多个通道，则函数会作用于每一个通道；且函数对每个像素点只处理一次，所以不能使用随机组件和其他生成器

# 将每个像素值翻倍（亮度翻倍）
evl1 = Image.eval(img1, lambda x: x*2)
# evl1.show()
# 将每个像素值减半（亮度减半）
evl2 = Image.eval(img1, lambda x: x/2)
# evl2.show()

# 7、合成图像
# Image.composite(image1, image2, mask)
# mask:图像的模式，可以为1、L或RGBA
# 将给定的两张图像以mask作为透明度，创建一张新的图像，所有图像必须有相同的尺寸

img_f = Image.open(r'C:\Users\cc\Desktop\flower.png')
# 分离img1的通道
r, g, b = img1.split()
# 合成图像
img6 = Image.composite(img1, img_f, mask=b)
# img6.show()

# 8、单通道合并多通道图像
# Image.merge(mode,bands)
# 将一组单通道图像合并成多通道图像
# mode：输出图像的模式  bands：输出图像中每个通道的序列

# 将图1三个通道分开
im_split = img1.split()
# 分别显示三个单通道图像
im_split[0].show()
im_split[1].show()
im_split[2].show()
# 将三个通道再次合并
img7 = Image.merge('RGB', im_split)
img7.show()
# 将图2三个通道分开
im2_split = img5.split()
# 将图2的第1个通道和图1的第2、3通道合成一张图像
rgbs = [im2_split[0], im_split[1], im_split[2]]
img8 = Image.merge('RGB', rgbs)
img8.show()

# 3、Image模块的方法

# 1）转换图像模式
# Image.convert(mode=None, matrix=None, dither=None, palette=0, colors=256)
'''
mode：转换的模式，支持每种模式转换为P、RGB和CMYK；有matrix参数只能转L、RGB；当模式间不能转换时，可以先转RGB，再转其他
matrix：转换矩阵，必须为包含浮点值长为4或12的元组
dither：抖动方法，RGB转换为P；RGB或L转换为1时使用；有matrix参数可无dither
palette：调色板，在RGB转换为P时使用，值为WEB或ADAPTIVE
colors：调色板的颜色值，默认256
'''
# RGB彩色模式转换为L模式的计算公式如下：
# L = R * 299/1000 + G * 587/1000 + B * 114/1000

# 将图像转换成黑白色
img9 = img1.convert('L')
# img9.show()

# 2）复制图像
# Image.copy()

# 3）抠图、截图
# Image.crop(box)
# box：相对图像左上角坐标(0,0)的矩形坐标元组, 顺序为(左, 上, 右, 下)
# 该方法从图像中获取box矩形区域的图像，相当于从图像中抠一个矩形区域出来

print(img1.size)
# 定义图像的坐标位置
box = (100, 100, 250, 250)
region = img1.crop(box)
print(region.size)
# region.show()

# 4）模糊、增强过滤
# Image.filter(filter)
# filter：过滤器
'''
BLUR	            模糊滤波，处理之后的图像会整体变得模糊
CONTOUR	            轮廓滤波，将图像中的轮廓信息全部提取出来
DETAIL	            细节增强滤波，会使得图像中细节更加明显
EDGE_ENHANCE        边缘增强滤波，突出、加强和改善图像中不同灰度区域之间的边界和轮廓的图像增强方法
EDGE_ENHANCE_MORE	深度边缘增强滤波，会使得图像中边缘部分更加明显
EMBOSS	            浮雕滤波，会使图像呈现出浮雕效果
FIND_EDGES	        寻找边缘信息的滤波，会找出图像中的边缘信息
SHARPEN	            锐化滤波，补偿图像的轮廓，增强图像的边缘及灰度跳变的部分，使图像变得清晰
SMOOTH	            平滑滤波，突出图像的宽大区域、低频成分、主干部分或抑制图像噪声和干扰高频成分，使图像亮度平缓渐变，减小突变梯度，改善图像质量
SMOOTH_MORE	        深度平滑滤波，会使得图像变得更加平滑
'''
from PIL import ImageFilter

# 模糊滤波
img_f1 = img1.filter(ImageFilter.BLUR)
# img_f1.show()
# 轮廓滤波
img_f2 = img1.filter(ImageFilter.CONTOUR)
# img_f2.show()
# 细节增强
img_f3 = img1.filter(ImageFilter.DETAIL)
# img_f3.show()

# 5）获取图像中每个通道名称的元组
# Image.getbands()
print(img1.getbands())      # ('R', 'G', 'B')
print(img9.getbands())      # ('L',)

# 6）计算图像中非零区域的边界框
# Image.getbbox()
# 返回边界框左、上、右、下像素坐标的四元组
print(img3.getbbox())       # (0, 0, 450, 450)

# 7）获取图像中颜色的使用列表
# Image.getcolors(maxcolors=256)
# maxcolors：最大颜色数，默认限制256色
# 若结果像素值数量大于maxcolors则返回None，返回值为(count, pixel)二元组列表，表示(出现的次数，像素的值)
print(img9.getcolors(maxcolors=2))
print(img9.getcolors(maxcolors=3))
print(img9.getcolors(maxcolors=255))

# 8）对图像的的每个像素点进行操作，返回图像的副本
# Image.point(lut, mode=None)
# lut：一个查找表，包含图像中每个波段的256个值，对每个可能的像素值调用一次函数，结果表将应用于图像的所有带区
# mode：输出模式

# 调整灰色图像的对比度
img_point = img9.point(lambda i: i < 80 and 255)
# img_point.show()

# 也可对RGB三个通道分别处理对比度

# 9）图像缩放
# Image.resize(size, resample=0, box=None)
# size：结果图像大小，单位像素，二元组(宽,高)类型
# resample：重新采样滤波器，BILINEAR双线性滤波、ANTIALIAS/LANCZOS平滑滤波、BICUBIC双立方滤波、NEAREST最近滤波（图像模式1或P设置）
# box：四元组类型，给出图像应该缩放的区域，值应在(0, 0, 宽, 高)矩形内
# 返回获取调整大小后的图像；即在原图中抠一个矩形区域（box）进行滤波处理（resample），最后以指定大小（size）返回

img10 = img_f.resize((500, 400), Image.NEAREST)
# img10.show()

# 其它方法补充
# 10）获取图像中每个像素的通道对象元组
# Image.getdata(band=None)
# band：需要获取的对应通道值，如RGB像素值为(R,G,B)元组，要返回某个波段，可传入对应索引（如0获取R波段）

# 11）获取图像中每个通道的最小值与最大值
# Image.getextrema()
# 对于单波段图像，包含最小和最大像素值的二元组；对于多波段图像，每个波段包含最大和最小像素值的二元组

# 12）通过传入坐标返回像素值
# Image.getpixel(xy)：
# xy：坐标，以(x,y)表示

# 13）将另一个图像粘贴到此图像中
'''
Image.paste(im, box=None, mask=None)
- im：源图像或像素值（整数或元组）
- box：一个可选的4元组，给出要粘贴到的区域。如果使用2元组，则将其视为左上角（默认）
- mask：可选的遮罩图像
'''
# 14）图像旋转角度
'''
Image.rotate(angle)
angle：逆时针角度
'''
# 参考文档：https://www.osgeo.cn/pillow/reference/Image.html

# 给图片添加文本
"""
ImageDraw.Draw.text((x,y), text, font, fill)
- (x,y)：添加文本的起始坐标位置，图像左上角为坐标原点
- text：文本字符串
- font：ImageFont对象
- fill：文本填充颜色
"""
# 创建字体对象
"""
ImageFont.truetype(font, size)
- font：字体文件（路径）
- size：字体大小
"""


# 4、案例：图像水印
# 1）ImageDraw.Draw.text(x,y,text,fill,font)
# x、y:水印的x坐标和y坐标  text:水印内容  fill:填充  font:字体
# 2）创建字体对象
"""
ImageFont.truetype(font, size)
font：字体文件（路径）   size：字体大小
"""

# 单个水印（右下角水平方向）
def img_waterMark(image, path, content):
    img = Image.open(image)
    # 给水印添加透明度需要转换图片模式为RGBA
    mode = img.mode
    rgba_img = img if mode == 'RGBA' else img.convert('RGBA')
    img_canvas = Image.new('RGBA', rgba_img.size, (255, 255, 255, 0))
    img_draw = ImageDraw.Draw(img_canvas)
    # 设置水印文字及字体大小
    font = ImageFont.truetype("arial.ttf", 24)
    text_x_width, text_y_height = img_draw.textsize(content, font=font)
    # 设置水印位置
    text_xy = (rgba_img.size[0] - text_x_width - 10, rgba_img.size[1] - text_y_height - 10)
    # 设置文本填充（颜色）和透明度
    img_draw.text(text_xy, content, font=font, fill=(255, 255, 255, 128))
    # 将原图片与水印文本合成
    image_with_text = Image.alpha_composite(rgba_img, img_canvas)
    # 将图片模式转换回去
    if mode != image_with_text.mode:
        image_with_text = image_with_text.convert(mode)
    image_with_text.save(path)
    return image_with_text

# img_waterMark(r'C:\Users\cc\Desktop\cat.jpg', r'C:\Users\cc\Desktop\cat_watermark.jpg', 'Image_WaterMark')

# 5、自定义扩展水印

# filestools模块是基于Pillow库二次封装的模块，其提供了add_mark()方法可用于自定义生成水印图像
# 安装：pip install filestools
'''
add_mark()
- file：图片文件路径
- out：输出目录
- mark：水印文字 color：颜色 size：大小 opacity：透明度 space：间隔 angle：旋转角度（逆时针）
'''

# from watermarker.marker import add_mark
#
# add_mark(
#     file=r'C:\Users\cc\Desktop\cat.jpg',
#     out=r'C:\Users\cc\Desktop\filestools',
#     mark="FilestoolsWaterMark",
#     opacity=0.5,
#     size=20,
#     angle=30,
#     space=75
# )

# 根据filestools模块自定义扩展水印

import math
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops

# 1）Pillow库的ImageChops（信道）模块
'''
ImageChops.difference()：返回两个图像之间逐个像素差异的绝对值
'''
# 2）Pillow库的ImageEnhance（增强）模块
'''
ImageEnhance.Brightness()：调整图像亮度
'''
# 生成水印图像
def create_watermark(text, size, font, opacity):
    mark = Image.new(mode='RGBA', size=(len(text) * size, size + 20))
    img_draw = ImageDraw.Draw(im=mark)
    img_draw.text(xy=(0, 0), text=text, fill=(255, 255, 255, 128), font=ImageFont.truetype(font, size=size))
    # 裁剪空白
    img_rgba = Image.new(mode='RGBA', size=mark.size)
    bbox = ImageChops.difference(mark, img_rgba).getbbox()
    mark = mark.crop(bbox) if bbox else mark
    # 设置透明度
    alpha = mark.split()[3]
    assert 0 <= opacity <= 1, "The parameter opacity must be between 0 and 1"
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    mark.putalpha(alpha)
    return mark

# create_watermark('Create_Watermark', 20, font='arial.ttf', opacity=1.0).show()

# 根据原图片生成扩展水印图片
def watermark_generator(img, text, font, size, opacity, angle, space):
    # 水印图片
    mark = create_watermark(text, size, font, opacity)
    # 将水印图片扩展并旋转生成水印大图
    w, h = img.size
    c = int(math.sqrt(w ** 2 + h ** 2))
    mark_rgba = Image.new(mode='RGBA', size=(c, c))
    y, idx = 0, 0
    mark_w, mark_h = mark.size
    while y < c:
        x = -int((mark_w + space) * 0.5 * idx)
        idx = (idx + 1) % 2
        while x < c:
            mark_rgba.paste(mark, (x, y))
            x = x + mark_w + space
        y = y + mark_h + space
    # 将水印大图旋转一定角度
    mark_rgba = mark_rgba.rotate(angle)
    return mark_rgba

# img = Image.open(r'C:\Users\cc\Desktop\cat.jpg')
# watermark_generator(img, 'Create_Watermark', 'arial.ttf', 20, 1.0, 30, 75).show()

# 添加水印
def add_watermark(image, out, content, font="arial.ttf", size=20, opacity=1.0, angle=30, space=75):
    img = Image.open(image)
    # 给水印添加透明度需要转换图片模式为RGBA
    mode = img.mode
    rgba_img = img if mode == 'RGBA' else img.convert('RGBA')
    watermark = watermark_generator(img, content, font, size, opacity, angle, space)
    w, h = img.size
    c = int(math.sqrt(w ** 2 + h ** 2))
    rgba_img.paste(watermark, (int((w - c) / 2), int((h - c) / 2)), mask=watermark.split()[3])
    if rgba_img.mode != mode:
        img = rgba_img.convert(mode)
    img.save(out)
    return img

# add_watermark(r'C:\Users\cc\Desktop\cat.jpg', r'C:\Users\cc\Desktop\add_watermark.jpg', 'Create_Watermark')


