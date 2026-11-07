"""本地视觉文字提取候选：完整行文及位置入库，核验之前不替代原图。

仅离线试点命令加载OCR依赖并限制线程；正常资料读取不用启动识别模型。
图像仅在内存裁切，数据库不保存图片或Base64。图表曲线与关系仍需单独核对。
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from app.services import material_content as store
from app.services import material_text_pilot as pilot


def engine():
    """沿用已验证的本地识别器，单个进程最多两条推理线程，避免占满CPU。"""
    import rapidocr_onnxruntime.utils as runtime
    from onnxruntime import SessionOptions
    from rapidocr_onnxruntime import RapidOCR

    def options():
        result = SessionOptions()
        result.intra_op_num_threads = 2
        result.inter_op_num_threads = 1
        result.add_session_config_entry("session.intra_op.allow_spinning", "0")
        result.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return result

    runtime.SessionOptions = options
    return RapidOCR()


def extract(source, *, repo, recognizer):
    """逐区域输出全部识别文字；不把OCR分数、无报错或字符数当成完整验收。"""
    import numpy as np
    import pymupdf

    from app.services.material_document import read_document

    h = source["source_hash"]
    candidate = read_document(h, repo=repo, policy=pilot.CANDIDATE)
    path = next((Path(p) for p in source["existing_paths"] if Path(p).is_file()), None)
    if path is None or store.checksum(path.read_bytes()) != h:
        raise ValueError("TEXT_PILOT_OCR_SOURCE_CHANGED")
    results = []
    with pymupdf.open(path) as document:
        for page in candidate["content"]["pages"]:
            for region in page["visual_regions"]:
                if region["kind"] == "raster_grid":
                    continue
                if region["region"] >= 10000:
                    raise ValueError("TEXT_PILOT_TOO_MANY_REGIONS")
                index = (page["page"] - 1) * 10000 + region["region"]
                cached = repo.page(h, "TEXT_ONLY_OCR_V1", index)
                if cached is not None:
                    import json

                    results.append(json.loads(cached))
                    continue
                crop = document[page["page"] - 1].get_pixmap(
                    matrix=pymupdf.Matrix(3, 3),
                    clip=pymupdf.Rect(region["bbox"]),
                    alpha=False,
                )
                pixels = np.frombuffer(crop.samples, dtype=np.uint8).reshape(crop.height, crop.width, crop.n)
                # RapidOCR接受BGR数组；保留全部识别候选，不按置信度裁掉低分文字。
                boxes, elapsed = recognizer(pixels[:, :, :3][:, :, ::-1].copy(), text_score=0.0)
                lines = [{"box": box, "text": text, "score": float(score)} for box, text, score in boxes or []]
                value = {
                    "source_hash": h,
                    "page": page["page"],
                    "region_hash": region["region_hash"],
                    "kind": region["kind"],
                    "bbox": region["bbox"],
                    "preview_sha256": page["preview_sha256"],
                    "text": "\n".join(r["text"] for r in lines),
                    "lines": lines,
                    "ocr_engine": "RapidOCR",
                    "seconds": elapsed,
                    "engine_version": importlib.metadata.version("rapidocr-onnxruntime"),
                    "status": "UNVERIFIED",
                    "can_replace_image": False,
                    "remaining": ["全部文字、数值与关系尚待逐项对照；图表曲线数据未完成提取"],
                }
                pilot._part(repo, h, "TEXT_ONLY_OCR_V1", index, value)
                results.append(value)
    return results
