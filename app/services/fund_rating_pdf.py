"""评分逐页文字读取：优先库内提取信息，首次采集仍受原生库互斥锁保护。"""

from contextlib import closing
from pathlib import Path

from app.core.pdf import PDF_LOCK
from app.services.material_content import document_pdfium


def text_pages(path: Path, *, limit: int | None = None) -> list[str]:
    """返回已提取的真实页文本，不要求原PDF存在，不缓存业务核验结论。

    PDFium不允许不同线程同时读取，即便是不同文件。评分的身份核验、管理
    事实和清单编译共用此锁；页及文字页必须先于文档关闭，避免析构跨线程。
    """
    with PDF_LOCK:
        with document_pdfium(path) as document:
            count = len(document) if limit is None else min(limit, len(document))
            pages = []
            for index in range(count):
                with closing(document[index]) as page, closing(page.get_textpage()) as text_page:
                    pages.append(text_page.get_text_range())
            return pages
