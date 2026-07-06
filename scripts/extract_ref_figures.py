import fitz
import os

OUTPUT_DIR = r"D:\py_project\Data_Element\output\img\ref_paper_pages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pdfs = [
    (r"D:\py_project\Data_Element\参考范文\多源数据TOPSIS物流优化.pdf", "物流TOPSIS", [8, 9, 10, 11, 14, 15, 29, 36, 37, 43]),
    (r"D:\py_project\Data_Element\参考范文\GBDT-NSGA-II信贷风险评估.pdf", "信贷GBDT", [7, 8, 12, 16, 20, 23, 28, 32, 44, 45]),
    (r"D:\py_project\Data_Element\参考范文\烟阻质量预测建模.pdf", "烟阻预测", [4, 5, 8, 9, 10, 14, 15, 16, 17, 18]),
]

for pdf_path, label, pages in pdfs:
    doc = fitz.open(pdf_path)
    for pg_num in pages:
        if pg_num <= len(doc):
            page = doc[pg_num - 1]
            pix = page.get_pixmap(dpi=200)
            out_path = os.path.join(OUTPUT_DIR, f"{label}_p{pg_num}.png")
            pix.save(out_path)
            print(f"Saved: {out_path}")
    doc.close()

print("Done!")
