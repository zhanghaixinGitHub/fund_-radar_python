"""真实PDF解析的并发与失败恢复回归，合成文件仅用于隔离测试。"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from app.services.fund_rating_pdf import text_pages


def test_native_pdf_concurrent_read_and_error_recovery(tmp_path):
    # 构造一个具有文字和正确xref的最小PDF，实际进入已安装的原生解析器。
    stream = b"BT /F1 12 Tf 10 80 Td (Actual PDF text) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    content, offsets = b"%PDF-1.4\n", []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(content)
    content += b"xref\n0 6\n0000000000 65535 f \n"
    content += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    content += f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF".encode()
    path = tmp_path / "actual-test.pdf"
    path.write_bytes(content)
    from app.integrations.dbfund_supplement import article_text
    from app.services.fund_earnings_batch_v4 import extract_pdf

    def read(index):
        # 页面读取、自动采集与既有财报解析在同一线程池交错，必须共用锁。
        if index % 3 == 1:
            return article_text(content, True)[0]
        if index % 3 == 2:
            return extract_pdf(content, 10)[0]
        return text_pages(path, limit=1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(read, range(32)))
    assert all(len(result) == 1 and "Actual PDF text" in result[0] for result in results)
    with pytest.raises(FileNotFoundError):
        text_pages(tmp_path / "missing.pdf")
    assert "Actual PDF text" in text_pages(path)[0]
