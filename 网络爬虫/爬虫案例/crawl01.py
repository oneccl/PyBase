import numpy as np
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup

# 国家统计局-教育局
# 2021年统计数据（各级各类学历教育学生情况）
url = 'http://www.moe.gov.cn/jyb_sjzl/moe_560/2021/quanguo/202301/t20230104_1038067.html'

out_path = r"C:\Users\cc\Desktop"

def get_html_str(callback):
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url=url, headers=headers)
    resp.encoding = resp.apparent_encoding
    html_str = resp.text
    return callback(html_str)

def bs4_html_parser(html_str):
    soup = BeautifulSoup(html_str, 'lxml')
    # 标题
    title = soup.h1.text
    # 表格
    trs = soup.find(attrs={'class': 'TRS_PreExcel TRS_PreAppend'}).table.tbody.find_all('tr')
    arr2d = []
    for tr in trs:
        tds = tr.find_all('td')
        row = []
        # 清洗
        for td in tds:
            td1 = td.text.replace('\n', '').replace('\u3000', '')
            td2 = re.sub('[A-Za-z]+', '', td1).strip()
            td3 = re.sub('\\s+', '', td2).replace('´', '').replace('-', '').replace('.', '').strip()
            row.append(td3)
        if len(row) == 4:
            arr2d.append(row)
    # 转为DataFrame
    data = pd.DataFrame(arr2d)
    data.columns = data.loc[0].tolist()
    data.drop(index=0, inplace=True)
    data.rename(columns={'': '分类'}, inplace=True)
    data['分类'] = data['分类'].apply(lambda x: re.sub('[0-9]+', '', x))
    data.map(lambda x: np.nan if x == '' else x)
    data['年份'] = 2021
    data.to_csv(fr'{out_path}\{title}.csv', index=False, encoding='utf-8')
    return data


if __name__ == '__main__':
    df = get_html_str(bs4_html_parser)
    print(df.to_string())


# 如何使用Excel打开CSV
'''
1） 新建空白Excel，数据->从文本/CSV导入
2） 选择不检测数据类型，点击加载
3） 剪切第二行列字段，覆盖到第一行，删除第二行空行，保存
'''

