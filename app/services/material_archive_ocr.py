"""外接盘扫描原件的全文OCR候选；只保存文字与行列位置，不存像素，不自动发布。"""

from __future__ import annotations

import io
import json
from pathlib import Path

from app.services import material_content as store
from app.services.material_text_pilot import _part

ENGINE = "OCR_PAGE_TEXT_V1"


def capture_pdf(target, *, recognizer, repo=None, limit=None, progress=None):
    """处理无原生文字的扫描页，保留每行及精确识别字符串；核验前不得作为正式事实。

    数据来源始终是D盘PDF。图像在内存解码，OCR输出包含位置和原始分数但不据此
    宣称正确；跨列顺序、表格归属、数字/符号及图表关系仍须独立复核。
    """
    import numpy as np
    import pymupdf

    from app.services.material_source_archive import archive_path

    repo = repo or store.repository()
    if not archive_path(target):
        raise ValueError("MATERIAL_OCR_MUST_READ_EXTERNAL_ORIGINAL")
    value = repo.get(target, fresh=True)
    if value is None or value.kind != "PDF":
        raise ValueError("MATERIAL_OCR_PDF_REQUIRED")
    completed = []
    with pymupdf.open(Path(target)) as document:
        for index, page in enumerate(document):
            if limit is not None and len(completed) >= limit:
                break
            if page.get_text().strip():
                continue
            previous = repo.page(value.source_hash, ENGINE, index)
            if previous is not None:
                completed.append(json.loads(previous))
                continue
            # 原始整页扫描优先直接解码原图，避免整页再次渲染成大PNG或先写回C盘。
            fallback_reason = None
            try:
                images = page.get_image_info(xrefs=True, hashes=True)
            except ValueError as error:
                # 部分PDF的图像xref目录遍历会报bad page number，但该页可以正常渲染。
                # 只回退内存渲染，不跳过此页、不将空白或上页OCR当作当前页内容。
                images = []
                fallback_reason = type(error).__name__+":"+str(error)[:200]
            whole = [image for image in images if image["xref"] and pymupdf.Rect(image["bbox"]).get_area()
                     >= page.rect.get_area() * 0.85]
            if len(whole) == 1:
                from PIL import Image

                raw_image = document.extract_image(whole[0]["xref"])["image"]
                with Image.open(io.BytesIO(raw_image)) as image:
                    pixels = np.array(image.convert("RGB"))[:, :, ::-1].copy()
            else:
                bitmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                pixels = np.frombuffer(bitmap.samples, dtype=np.uint8).reshape(
                    bitmap.height, bitmap.width, bitmap.n)[:, :, ::-1].copy()
            boxes, _ = recognizer(pixels, text_score=0.0)
            lines = [{"box": box, "text": text, "score": float(score)} for box, text, score in boxes or []]
            result = {"source_hash": value.source_hash, "page": index + 1, "status": "UNVERIFIED_TEXT_CANDIDATE",
                      "text": "\n".join(line["text"] for line in lines), "lines": lines,
                      "source_dimensions": [int(pixels.shape[1]), int(pixels.shape[0])],
                      "remaining": ["独立核对全文、数字正负号、表格行列和图表关系"],
                      "binary_payload_saved": False, "can_publish_without_review": False}
            if fallback_reason:
                result["in_memory_render_fallback"] = fallback_reason
            _part(repo, value.source_hash, ENGINE, index, result)
            completed.append(result)
            if progress:
                progress({"source_hash": value.source_hash, "OCR_pages_saved": len(completed),
                          "total_pages": len(document)})
    return {"source_hash": value.source_hash, "OCR_pages_saved": len(completed),
            "status": "OCR_CANDIDATES_SAVED_REVIEW_REQUIRED"}
