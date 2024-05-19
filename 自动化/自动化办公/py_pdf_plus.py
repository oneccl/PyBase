
# PyMuPDF库

# 安装：pip install PyMuPDF

path = r'file.pdf'

# 导入库
import fitz

# 文档操作
# 打开文档
with fitz.open(path) as doc:
    ...

# 文档属性与方法
with fitz.open(path) as doc:
    # 文档属性
    print(doc.page_count)    # PDF页数（Int类型）
    print(doc.metadata)      # PDF元数据（字典类型）
    # 文档方法
    print(doc.get_toc())     # 获取文档目录结构（List类型）
    print(doc.pages())       # pages(start,stop,step)：返回每个Page对象的迭代器
    print(doc.load_page(1))  # load_page(page_id)：返回指定页的Page对象（页数索引从0开始）
    print(doc[1])            # 返回指定页的Page对象（页数索引从0开始）


# 页面操作
with fitz.open(path) as doc:
    for page in doc.pages():
        print(page.get_links())    # 获取页面的链接（List[Dict]类型）
        print(page.links())        # 返回页面链接对象的迭代器
        print(page.annots())       # 无get_annots()方法，返回页面批注Annot对象的迭代器
        print(page.widgets())      # 无get_widgets()方法，返回页面表单域Widget对象的迭代器

# 文档页面其他操作
# 删除PDF页面
with fitz.open(path) as doc:
    # 删除PDF第2页
    doc.delete_page(1)
    # 另存为新的PDF文件
    doc.save("deleted1.pdf")

# 插入/修改PDF文本
with fitz.open(path) as doc:
    # 定义插入/修改的文本的起点
    p = fitz.Point(75, 150)
    # 文本
    text = "line1\nline2."
    # 插入/修改指定页面
    page = doc[0]
    rc = page.insert_text(p, text)
    # 另存为新的PDF文件
    doc.save("modified.pdf")


# 提取文本
'''
page.get_text(opt)：opt使用下面字符串以获取不同的文本格式
- text：默认，提取纯文本（带换行符）
- blocks：按文本块（段落）提取，返回列表类型
- words：按单词提取（不包含空格的字符串），返回列表类型
- html：创建包含任何图像页面的完整可视化版本，可以通过浏览器显示
- dict/json：与html相同的信息级别，但以字典或JSON字符串的形式提供
- rawdict/rawjson：dict/json的超集，它还提供字符详细信息，如XML
- xhtml：文本信息级别的text版本，包含图像，可以通过浏览器显示
- xml：不包含图像，但包含完整的位置和字体信息，包括每个文本字符
'''
extracted_text = ""
with fitz.open(path) as doc:
    for page in doc.pages():
        extracted_text += page.get_text()
    print(extracted_text)

# 搜索文本
# page.search_for(text)：返回文本字符串出现在页面上的位置（Rect对象列表）
doc = fitz.open(path)
page = doc[0]
print(page.search_for("mupdf"))   # [Rect(79.66999816894531, 261.78997802734375, 90.40835571289062, 271.78997802734375)]
doc.close()

# 提取表格
# page.find_tables()：获取该页面所有Table对象，返回List类型
with fitz.open(path) as doc:
    page = doc.load_page(1)
    tables = page.find_tables()
    for table in tables:
        # 表格Table对象转换为DataFrame对象
        tab_df = table.to_pandas()
        print(tab_df.head().to_string())


# 提取图像
# page.get_images(full=True)：获取该页面所有Image对象，返回List类型
# doc.extract_image(xref)：提取该Image图像，返回Dict类型
with fitz.open(path) as doc:
    page = doc.load_page(1)
    images = page.get_images(full=True)
    for image in images:
        # 提取图像
        base_image = doc.extract_image(image)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]   # 后缀
        # 保存图像
        image_name = f"image_{1}.{image_ext}"
        with open(image_name, "wb") as img:
            img.write(image_bytes)


# 拆分与合并
# doc.insert_pdf(docsrc,from_page,to_page,start_at,rotate,links,annots,show_progress,final)：PDF插入
# - docsrc：复制源PDF文档，如果from_page>to_page，则复制顺序相反
# - from_page：要复制的第一个源页面，默认从0开始
# - to_page：要复制的第一个源页面，默认最后一页
# - start_at：from_page将成为目标中的页码
# - rotate：旋转复制的页面的角度，默认-1不改变
# - links：是否也复制链接
# - annots：是否也复制注释
# - show_progress：进度消息间隔，默认0表示没有消息
# - final：是否在源PDF中的最后插入内容
# 打开需要拆分的PDF
with fitz.open(path) as doc:
    # 创建一个空的PDF文档
    with fitz.open() as doc1:
        # 将doc前2页添加到doc1
        doc1.insert_pdf(doc, to_page=1)
        # 命名并保存
        doc1.save("doc_prev2.pdf")

# 创建一个空的PDF文档
with fitz.open() as doc:
    # 要合并的PDF
    doc1 = fitz.open("doc1.pdf")
    doc2 = fitz.open("doc2.pdf")
    # 合并
    doc.insert_pdf(doc1)
    doc.insert_pdf(doc2)
    # 命名并保存
    doc.save("doc1_join_doc2.pdf")


