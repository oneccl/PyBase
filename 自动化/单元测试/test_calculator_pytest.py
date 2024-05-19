
# Python单元测试
# Unittest是Python自带的单元测试框架，Pytest是一个第三方单元测试框架，具有丰富的插件生态，兼容Unittest测试集，社区繁荣

# Pytest特性
# Pytest可以和Selenium、Requests、Appium结合实现Web自动化、接口自动化、App自动化
# Pytest可以实现测试用例的跳过以及reruns失败用例重试
# Pytest可以和Allure生成非常美观的测试报告
# Pytest可以和Jenkins持续集成

# 例如：
"""
pytest-html            生成html格式的自动化测试报告
pytest-xdist           测试用例分布式执行，多CPU分发
pytest-ordering        用于改变测试用例的执行顺序
pytest-rerunfailures   用例失败后重跑
allure-pytest          用于生成美观的测试报告
pytest-selenium        集成Selenium
"""

# 安装：pip install pytest

# Pytest框架的用例执行入口：pytest.main(args参数列表)

# 1、Pytest测试用例的运行参数
'''
-s: 显示程序中的print/logging输出
-v: 丰富信息模式，输出更详细的用例执行信息
-vs: 该两个参数可一起使用
-q: 安静模式，不输出环境信息
-x: 出现一条测试用例失败就退出测试
-n=num: 多线程或分布式运行测试用例（需要先安装pytest-xdist）
--html=./report.html: 生成xml/html格式测试报告（需要先安装pytest-html）
--reruns num：重试运行测试用例（需要先安装pytest-rerunfailures）
'''

# 2、Pytest测试用例的运行方式
# 1）主函数模式
'''
pytest.main()              运行当前目录下所有以test开头或结尾的文件
pytest.main(['test.py'])   运行指定模块
pytest.main(['./test'])    运行指定目录
通过nodeid指定用例运行：nodeid有模块名、分隔符、类名、方法名、函数名组成
pytest.main(["test.py::Test_Class"])                 指定类名
pytest.main(["test.py::Test_Class::test_method"])    指定方法
'''
# 2）终端命令行模式
'''
pytest                     运行所有，当传入参数-s、-v、-x时，相当于命令行输入pytest -s -v -x
pytest test.py             运行指定模块
pytest ./test              运行指定目录
pytest test.py::Test_Class::test_method       指定方法
'''
# 3）通过pytest.ini配置文件运行
'''
位置：一般放在项目的根目录
编码：必须是ANSI，可以使用Notepad++修改编码格式
作用：改变Pytest默认的行为
运行规则：不管是主函数的模式运行，还是命令行模式运行，都会去读取这个配置文件
'''
# 例如：
'''
[pytest]
# 命令行参数，多个参数使用空格分隔
addopts = -vs
# 测试用例文件夹（目录）
testpaths = ../pytestproject
# 测试搜索的模块文件名称
python_files = test*.py
# 测试搜索的测试类名
python_classes = Test*
# 测试搜索的测试函数名
python_functions = test*
'''

# 3、Pytest测试用例的执行顺序

# pytest：默认从上到下
# 改变默认执行顺序：使用mark标记
# @pytest.mark.run(order=3)

# 4、基本使用

# 1）拿到开发的代码：见Calculator.python文件
from Calculator import Calculator

# 2）测试用例编写规则
"""
1）测试文件以test开头或结尾
2）测试类以Test开头，且不能带init()方法
3）测试函数以test开头
4）断言使用基本的assert即可
"""
import pytest

# 3）编写测试用例
# class TestCalculator():
#     def test_add(self):
#         c = Calculator()
#         result = c.add(10, 5)
#         assert result == 15
#
#     def test_sub(self):
#         c = Calculator()
#         result = c.sub(10, 5)
#         assert result == 5
#
#     def test_mul(self):
#         c = Calculator()
#         result = c.mul(10, 5)
#         assert result == 50
#
#     def test_div(self):
#         c = Calculator()
#         result = c.div(10, 5)
#         assert result == 2
#
# if __name__ == '__main__':
#     pytest.main(['-s', 'test_calculator_pytest.py'])
'''
============================= test session starts =============================
collecting ... collected 4 items

test_calculator_pytest.py::TestCalculator::test_add PASSED               [ 25%]
test_calculator_pytest.py::TestCalculator::test_sub PASSED               [ 50%]
test_calculator_pytest.py::TestCalculator::test_mul PASSED               [ 75%]
test_calculator_pytest.py::TestCalculator::test_div PASSED               [100%]

============================== 4 passed in 0.02s ==============================
'''

# 5、用例前置和后置（使用Fixture夹具）
# 添加Fixture夹具（作用于函数）:
# 前置：用例执行前执行  后置：用例执行后执行
@pytest.fixture()
def set_up():
    print("前置执行 Start")
    yield
    print("后置执行 End")

'''
方式1：将夹具函数名称作为参数传递到测试用例函数当中
方式2：@pytest.mark.usefixtures("夹具函数名称")
方式3：@pytest.fixture(autouse=True)，设置了autouse就可以不用上述两种手动方式，默认就会使用夹具
'''
# class TestCalculator():
#     def test_add(self, set_up):
#         c = Calculator()
#         result = c.add(10, 5)
#         assert result == 15
#
#     @pytest.mark.usefixtures("set_up")
#     def test_mul(self):
#         c = Calculator()
#         result = c.mul(10, 5)
#         assert result == 50
#
# if __name__ == '__main__':
#     pytest.main(['-s', 'test_calculator_pytest.py'])
'''
============================= test session starts =============================
collecting ... collected 2 items

test_calculator_pytest.py::TestCalculator::test_add 前置执行 Start
PASSED               [ 50%]后置执行 End

test_calculator_pytest.py::TestCalculator::test_mul 前置执行 Start
PASSED               [100%]后置执行 End

============================== 2 passed in 0.02s ==============================
'''

# 6、参数化测试
'''
若只有一个参数，则使用值列表，例如@pytest.mark.parametrize("num1", [1, 2, 3])
若有多个参数，则使用元祖列表，一个参数对应一个元祖，例如@pytest.mark.parametrize("num1, num2", [(2, 3, 5), (1, 2, 3)])
当装饰器@pytest.mark.parametrize装饰测试类时，会将数据集合传递给类的所有测试用例方法（测试方法参数名与装饰器参数名保持一致）
一个函数或一个类可以装饰多个@pytest.mark.parametrize，当参数化有多个装饰器时，用例数是N*M
'''

# 7、断言
# Pytest使用assert进行断言

# 8、测试用例重跑
# 需要安装额外的插件pytest-rerunfailures
# 使用装饰器@pytest.mark.flaky(reruns=2)

# 9、跳过测试用例
# 使用装饰器@pytest.mark.skip

# 10、Allure
# Allure是一款轻量级并且非常灵活的开源测试报告框架；支持绝大多数测试框架，如TestNG、Pytest、JUint等；Allure简单易用，易于集成

# 1）配置环境变量：Path下配置安装路径/bin
# 验证：cmd下执行：allure

# 2）安装：pip install allure-pytest
# allure-pytest是Pytest的一个插件，通过它我们可以生成Allure所需要的用于生成测试报告的数据

# 3）Allure特性
'''
@allure.feature        用于描述被测试产品需求
@allure.story          用于描述feature的用户场景，即测试需求
with allure.step()     用于描述测试步骤，将会输出到报告中
allure.attach          用于向测试报告中输入一些附加的信息，通常是一些测试数据、截图
'''
# import os
# import allure
#
# class TestCalculator():
#
#     @allure.feature("两数相乘")
#     @allure.story("相乘结果")
#     def test_mul(self):
#         with allure.step("查看相乘结果"):
#             allure.attach("6", "附加信息")
#         c = Calculator()
#         result = c.mul(10, 5)
#         assert result == 50
#
# if __name__ == '__main__':
#     pytest.main(['--alluredir', './result', 'test_calculator_pytest.py'])
#     # 将测试报告转为HTML格式
#     cmd = 'allure ' + 'generate ' + './result ' + '-o ' + './html ' + '--clean'
#     # system函数可以将字符串转化成命令在服务器上运行
#     os.system(cmd)


# https://blog.csdn.net/Tencent_TEG/article/details/115713186

