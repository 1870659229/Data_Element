"""conftest: 强制 UTF-8 模式，确保 pytest 能正确读取含中文/全角字符的源文件"""
import os
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
