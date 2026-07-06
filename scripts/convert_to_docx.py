"""将manuscript_draft.md转换为docx格式
使用pypandoc调用pandoc进行转换，自动处理图片路径
"""
import os
import sys
from pathlib import Path

# 尝试导入pypandoc
try:
    import pypandoc
except ImportError:
    print("Error: pypandoc not installed. Run: pip install pypandoc")
    sys.exit(1)

# 路径配置
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "paper_rewriting_output" / "manuscript_draft.md"
OUTPUT_FILE = ROOT_DIR / "paper_rewriting_output" / "manuscript_draft.docx"
IMG_DIR = ROOT_DIR / "output" / "img"

def convert_md_to_docx():
    """转换markdown到docx"""
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        return False

    print(f"Converting: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    # 使用pypandoc转换
    # extra_args配置图片路径和资源文件夹
    extra_args = [
        '--resource-path=' + str(IMG_DIR),  # 设置图片搜索路径
        '--standalone',  # 生成完整文档
        '--toc',  # 生成目录（可选）
    ]

    try:
        output = pypandoc.convert_file(
            str(INPUT_FILE),
            'docx',
            outputfile=str(OUTPUT_FILE),
            extra_args=extra_args
        )
        print(f"Conversion successful!")
        print(f"Output file: {OUTPUT_FILE}")
        print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
        return True
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

if __name__ == '__main__':
    success = convert_md_to_docx()
    sys.exit(0 if success else 1)
