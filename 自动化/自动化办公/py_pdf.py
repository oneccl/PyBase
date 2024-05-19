
# Python操作PDF

# Python操作PDF主要有两个库：PyPDF2和pdfplumber
# PyPDF2是一个用于处理PDF文件的Python第三方库
# pdfplumber是一个用于解析PDF文档的第三方库，可以解析、提取、转换PDF文档数据
# 官网：
"""
PyPDF2：https://pypi.org/project/PyPDF2/
pdfplumber：https://github.com/jsvine/pdfplumber
"""
# 安装：
# pip install PyPDF2
# pip install pdfplumber

# 1、批量拆分
'''操作步骤
1）读取PDF的整体内容
2）遍历每一页，以step为间隔将PDF存成小文件块
3）将小文件块重新保存为新的PDF文件
'''
import os
from PyPDF2 import PdfReader, PdfWriter

# filepath:读取文件路径  filename:保存文件的统一命名  dirpath:保存文件路径  step:每隔多少页生成一个文件
def split_pdf(filepath, dirpath, filename, step):
    # 创建保存目录
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    pdf_reader = PdfReader(filepath)
    # 读取每一页的数据
    page_list = pdf_reader.pages
    pages = len(page_list)
    for page in range(0, pages, step):
        pdf_writer = PdfWriter()
        # 拆分pdf，每step页的拆分为一个文件，如step=5，表示0-4页、5-9页...各为一个文件
        for index in range(page, page + step):
            if index < pages:
                pdf_writer.add_page(page_list[index])
        # 保存拆分后的小文件
        save_path = os.path.join(dirpath, filename + str(int(page / step) + 1) + '.pdf')
        print(save_path)
        with open(save_path, "wb") as out:
            pdf_writer.write(out)
    print("保存路径: " + dirpath)

# split_pdf(r'C:\Users\cc\Desktop\test.pdf', r'C:\Users\cc\Desktop\PDF', 'pdf_split_', step=2)

# 2、批量合并
'''操作步骤
1）确定合并文件顺序
2）循环追加到一个文件块中
3）保存为一个新文件
'''

# filepath:要合并的PDF文件目录  filename:原文件的统一命名  dirpath:合并后的保存路径
def concat_pdf(filepath, dirpath, filename):
    pdf_writer = PdfWriter()
    # ['pdf_split_1.pdf', 'pdf_split_2.pdf']
    list_filename = os.listdir(filepath)
    # 对文件进行排序
    list_filename.sort(key=lambda x: int(x[:-4].replace(filename, "")))
    for filename in list_filename:
        file_path = os.path.join(filepath, filename)
        print(file_path)
        # 读取文件并获取文件的页数
        pdf_reader = PdfReader(file_path)
        page_list = pdf_reader.pages
        pages = len(page_list)
        # 逐页添加
        for page in range(pages):
            pdf_writer.add_page(page_list[page])
    # 保存合并后的文件
    with open(dirpath, "wb") as out:
        pdf_writer.write(out)
    print("保存路径: " + dirpath)

# concat_pdf(r'C:\Users\cc\Desktop\PDF', r'C:\Users\cc\Desktop\pdf_concat.pdf', "pdf_split_")

# 3、内容提取（文字）
# 表格会以文本形式提取每行内容
import pdfplumber

# filepath:PDF文件路径  pageNum:指定仅解析第几页（默认全部）
# 返回所有行（lines）的列表
def extract_text(filepath, pageNum=None):
    with pdfplumber.open(filepath) as pdf:
        page_list = pdf.pages
        # 提取指定页文字（索引从0开始）
        if pageNum is not None:
            page = page_list[pageNum]
            print(page.extract_text())
            return page.extract_text().split('\n')
        else:
            # 提取全部内容
            res = ""
            for index in range(len(page_list)):
                page = page_list[index]
                if index == 0:
                    res = res + page.extract_text()
                else:
                    res = res + '\n' + page.extract_text()
                print(page.extract_text())
            return res.split('\n')

# 提取指定页文字内容
# print(extract_text(r'C:\Users\cc\Desktop\test.pdf', 2))
# 提取全部文字内容
# print(extract_text(r'C:\Users\cc\Desktop\test.pdf'))

# 4、提取内容（表格）
import pandas as pd

# filepath:PDF文件路径  pageNum:指定仅解析第几页（默认全部）
# 返回所有表格转换为df的列表
def extract_table(filepath, pageNum=None):
    with pdfplumber.open(filepath) as pdf:
        page_list = pdf.pages
        # 提取指定页表格（索引从0开始）
        if pageNum is not None:
            page = page_list[pageNum]
            # table = page.extract_table()
            # 若指定页存在多个表格，则使用extract_tables()
            return table2df(page)
        else:
            # 提取全部表格
            df_list = []
            for index in range(len(page_list)):
                page = page_list[index]
                df_ls = table2df(page)
                if len(df_ls) != 0:
                    df_list.append(df_ls)
            return df_list

# 表格转换为DataFrame
def table2df(page):
    df_list = []
    tables = page.extract_tables()
    for table in tables:
        df_table = pd.DataFrame(table[1:], columns=table[0])
        df_list.append(df_table)
    return df_list

# 提取指定页表格
# df_tables = extract_table(r'C:\Users\cc\Desktop\test.pdf', 2)
# for df_table in df_tables:
#     print(df_table.to_string())
# 提取全部表格
# df_tables = extract_table(r'C:\Users\cc\Desktop\test.pdf')
# for page_tabs in df_tables:
#     for page_tab in page_tabs:
#         print(page_tab.to_string())

# 5、提取图片
# fitz模块可用于提取pdf中的图片
# 安装：pip install PyMuPDF
'''操作步骤
1）使用fitz打开文档，获取文档详细数据
2）遍历每个元素，通过正则找到图片的索引位置
3）使用Pixmap将索引对应的元素生成图片
4）通过size过滤较小的图片
'''
import re
import fitz

# filepath:PDF文件路径  dirpath:图片保存目录  filename:保存图片的统一命名  fmt:保存格式(默认PNG)
def extract_image(filepath, dirpath, filename, fmt='PNG'):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    # 使用正则表达式来查找图片
    check_XObject = r"/Type(?= */XObject)"
    check_Image = r"/Subtype(?= */Image)"
    img_count = 0
    # 打开PDF
    pdf_info = fitz.open(filepath)
    xref_len = pdf_info.xref_length()
    # 打印PDF的信息
    print(f"页数: {len(pdf_info)}, 对象: {xref_len - 1}")
    # 遍历PDF中的对象，仅操作图像
    for index in range(1, xref_len):
        text = pdf_info.xref_object(index)
        is_XObject = re.search(check_XObject, text)
        is_Image = re.search(check_Image, text)
        # 若不是对象也不是图片，则跳过
        if is_XObject or is_Image:
            img_count += 1
            # 根据索引生成图像
            pix = fitz.Pixmap(pdf_info, index)
            pic_filepath = os.path.join(dirpath, filename + str(img_count) + f'.{fmt.lower()}')
            print(pic_filepath)
            # size可以反映像素，可以通过设置一个阈值过滤
            if pix.size < 10000:
                continue
            # 若一个像素的大小≥5，则转换CMYK色彩模式
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            # 存储为指定格式（默认PNG）
            pix.save(filename=pic_filepath, output=fmt.upper())
    print(f"保存总计: {img_count}")

# 提取图片内容
# extract_image(r'C:\Users\cc\Desktop\test.pdf', r'C:\Users\cc\Desktop', 'img_', 'jpg')

# 6、PDF添加水印
# 方式1：自己准备水印PDF文件:
# 新建watermark.docx->设计->水印->自定义水印->文字水印->输入文字，选择字体字号颜色和版式->另存为PDF

# filepath:原PDF文件  markpath:水印PDF文件  dirpath:添加水印后的保存文件
def pdf_watermark(filepath, markpath, dirpath):
    pdf_reader = PdfReader(filepath)
    page_list = pdf_reader.pages
    pages = len(page_list)
    mark_reader = PdfReader(markpath)
    pdf_writer = PdfWriter()
    # 给每页添加水印
    for page in range(pages):
        page = page_list[page]
        page.merge_page(mark_reader.pages[0])
        pdf_writer.add_page(page)
    with open(dirpath, 'wb') as out:
        pdf_writer.write(out)

# pdf_watermark(r'C:\Users\cc\Desktop\test.pdf', r'C:\Users\cc\Desktop\watermark.pdf', r'C:\Users\cc\Desktop\test_watermark.pdf')

# 方式2：创建PDF水印文件
# 安装：pip install reportlab

from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

def create_pdf_watermark(filename, content):
    # 默认页面大小为A4（21cm*29.7cm）
    c = canvas.Canvas(filename)
    # 画布平移保证文字完整性，坐标系左下为(0,0)
    c.translate(10 * cm, 5 * cm)
    # 设置旋转角度
    c.rotate(45)
    # 设置字体大小
    c.setFont("Helvetica", 20)
    # 指定填充颜色
    c.setFillColorRGB(0, 0, 0, 0.1)
    # 设置透明度，1为不透明
    c.setFillAlpha(0.1)
    # 水印形状
    for i in range(5):
        for j in range(10):
            a = 10 * (i - 2)
            b = 5 * (j - 3)
            c.drawString(a * cm, b * cm, content)
    # 保存pdf文件
    c.save()

# create_pdf_watermark(r'C:\Users\cc\Desktop\create_watermark.pdf', 'Watermark_Test')

# 7、加密与解密
# 1）加密
# filepath:PDF文件  dirpath:加密后文件  passwd:密码
def pdf_encrypt(filepath, dirpath, passwd):
    pdf_reader = PdfReader(filepath)
    page_list = pdf_reader.pages
    pages = len(page_list)
    pdf_writer = PdfWriter()
    for index in range(pages):
        pdf_writer.add_page(page_list[index])
    # 添加密码
    pdf_writer.encrypt(passwd)
    with open(dirpath, "wb") as out:
        pdf_writer.write(out)

# 文档加密
pdf_encrypt(r'C:\Users\cc\Desktop\test.pdf', r'C:\Users\cc\Desktop\test_encrypt.pdf', passwd='123456')

# 2）解密
from PyPDF2 import PdfReader, PdfWriter, PasswordType
import os

# filepath：加密的PDF文件  passwd：密码
def pdf_decrypt(filepath: str, passwd: str):
    try:
        pdf_reader = PdfReader(filepath, strict=False)
        if pdf_reader.is_encrypted:
            decrypt = pdf_reader.decrypt(passwd)
            if decrypt is not PasswordType.OWNER_PASSWORD:
                raise Exception("密码不正确！")
            if pdf_reader is not None:
                pdf_writer = PdfWriter()
                pdf_writer.append_pages_from_reader(pdf_reader)
                pathfile, suffix = os.path.splitext(filepath)
                outpath = "".join(pathfile + "_decrypt" + suffix)
                with open(outpath, "wb") as out:
                    pdf_writer.write(out)
        return pdf_reader
    except Exception as e:
        print(e)

pdf_decrypt(r'C:\Users\cc\Desktop\test_encrypt.pdf', passwd='123456')

# 8、PDF转Word
# pdf2docx是一个将PDF文档转换为Microsoft Word文档格式（.docx）的第三方库。该转换可以使用户更方便地编辑和修改PDF文档内容，同时保留原始文档的格式和布局
# 安装：pip install pdf2docx

from pdf2docx import Converter

def pdf2word(pdf, word):
    with Converter(pdf) as cv:
        # 默认start=0，end=None，pages:指定页面转换
        cv.convert(word, start=0, end=None)

# pdf2docx在命令行执行
# pdf2docx也可以作为一个命令行工具，直接在命令窗口中使用：
'''
pdf2docx convert pdf_path docx_path
'''
# 可通过--start、--end和--pages指定页面范围



