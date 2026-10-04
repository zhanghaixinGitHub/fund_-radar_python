"""002112 近年公告正文接入：本地来源快照、逐日持仓关联及可追溯文本输入。

只生成独立研究目录，不修改材料缓存、旧实验或现用模型。日期来自披露目录；
下载时间只作采集凭证。当前存档无法证明历史首见，因此结果只用于历史重建研究。
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from scripts import fund_002112_existing_data_inputs_v1 as holdings
from scripts import fund_002112_existing_events_sources_v1 as io
from scripts.fund_002112_news_correction_sources_v1 import LEXICON

ROOT = io.RESEARCH / "recent-body-integration/20261001-v1"
PREVIOUS = io.RESEARCH / "update-frequency-development/20261001-v1"
REPORTS = io.RESEARCH / "news-holdings-correction/20261001-v1/admitted-reports.json"
BUNDLE = io.RESEARCH / "existing-data-experiment/20260930-v1/snapshot/normalized-inputs.json"
INDEX = io.PY / "data/fund-materials/002112.json"
KINDS = ("company", "fund", "news")
WINDOWS = (1, 5, 20)
HASH_DIM = 2048
MIN_DAY = "2023-11-01"
MAX_DAY = "2026-09-29"


def read_lines(path):
    """按文件换行读取JSONL；PDF正文中的U+2028等分段符不是记录分隔符。"""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def next_midnight(day):
    """只有发布日期时，整日结束后才允许使用；不把当天收盘后公告放进当天早盘。"""
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat() + "T00:00:00+08:00"


def pdf_day(value):
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", value or "")
    return "-".join(match.groups()) if match else None


def local_source(relative, research=io.RESEARCH):
    """按生产材料存储的覆盖规则读取：增量目录优先，旧目录作为后备。"""
    for path in (research / "materials-live" / relative, research / relative):
        if path.is_file():
            return path
    return None


def body_relative(entry):
    """列表是索引，不是正文；通过原采集主键定位实际 JSON，不按标题模糊拼接。"""
    if entry["kind"] == "company":
        if not entry["id"].startswith("company-"):
            raise ValueError("INVALID_COMPANY_ID")
        return Path("supplement/company-documents") / (io.digest(entry["id"][8:]) + ".json")
    if not entry["id"].startswith("fund-"):
        raise ValueError("INVALID_FUND_DOCUMENT_ID")
    return Path("supplement/documents") / (entry["id"][5:] + ".json")


class Snapshot:
    """将本轮真正读取的 JSON 字节按摘要保存；原始 PDF 校验摘要但不重复复制大文件。"""

    def __init__(self, root):
        self.root = root
        self.files = {}

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        import hashlib

        digest = hashlib.sha256(raw).hexdigest()
        dest = self.root / "sources" / (digest + ".json")
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as handle:
                handle.write(raw)
        self.files[str(path)] = {"sha256": digest, "snapshot": str(dest), "bytes": len(raw)}
        return io.payload(json.loads(raw.decode("utf-8-sig")))


def resolve_document(entry, snapshot, research=io.RESEARCH):
    """展开正文并检查身份、原文件摘要和已知较晚版本；缺失原因保留在行内。

    textComplete 只是界面提示，不能决定是否读取正文；是否采用由实际文本、
    文档主键、证券代码和原文件校验决定。旧 training_eligible 不跨实验继承。
    """
    doc = {
        "id": entry["id"], "kind": entry["kind"], "title": entry["title"],
        "stock_code": entry.get("stockCode"), "published_date": entry["publishedDate"],
        "source_url": entry.get("sourceUrl"), "title_available_at": next_midnight(entry["publishedDate"]),
        "body": "", "body_available_at": None, "body_exclusions": [],
        "index_text_complete": entry.get("textComplete"), "historical_first_seen_proven": False,
    }
    source = local_source(body_relative(entry), research)
    if source is None:
        doc["body_exclusions"].append("BODY_FILE_MISSING")
        return doc
    saved = snapshot.read(source)
    doc["source_path"] = str(source.resolve())
    doc["source_sha256"] = snapshot.files[doc["source_path"]]["sha256"]
    doc["old_training_eligible"] = saved.get("training_eligible")
    receipt = saved.get("receipt", {})
    doc["receipt"] = receipt
    pages = saved.get("pages", [])
    text = "\n".join(pages) if pages else saved.get("text", "")
    doc["extracted_chars"] = len(text)
    doc["text_status"] = saved.get("text_status")
    errors = doc["body_exclusions"]
    if entry["kind"] == "company":
        if str(saved.get("announcement_id")) != entry["id"][8:]:
            errors.append("ANNOUNCEMENT_ID_MISMATCH")
        if saved.get("stock_code") != (entry.get("stockCode") or "").split(".")[0]:
            errors.append("STOCK_ID_MISMATCH")
        if re.sub(r"\s+", "", saved.get("title", "")) != re.sub(r"\s+", "", entry["title"]):
            errors.append("TITLE_IDENTITY_MISMATCH")
        if entry["id"][8:] not in receipt.get("url", ""):
            errors.append("RECEIPT_URL_IDENTITY_MISMATCH")
        announced = saved.get("announced_at_source")
        if announced:
            # 来源带时分秒时也保留更保守的目录日期边界。
            doc["title_available_at"] = max(doc["title_available_at"], announced)
    else:
        catalog = saved.get("catalog", {})
        if catalog.get("title") != entry["title"]:
            errors.append("CATALOG_TITLE_MISMATCH")
        # 管理公司的宣传、获奖、人员访谈，不等同于002112的投资事件或国家政策。
        if entry["kind"] == "news" and not saved.get("fund_name_or_code_mentioned"):
            doc["scope_exclusion"] = "MANAGER_NEWS_WITHOUT_EXPLICIT_FUND_RELEVANCE"
        for key in ("creationDate", "modificationDate"):
            if catalog.get(key):
                stamp = datetime.fromtimestamp(int(catalog[key]) / 1000, io.ZONE).isoformat()
                doc["title_available_at"] = max(doc["title_available_at"], stamp)
    if len(text.strip()) < 100:
        errors.append("BODY_TOO_SHORT")
    if saved.get("text_status") != "TEXT_EXTRACTED":
        errors.append("INCOMPLETE_TEXT_EXTRACTION")
    raw = local_source(Path(receipt.get("file", "__missing__")), research)
    doc["raw_path"] = str(raw.resolve()) if raw else None
    raw_valid = bool(raw and receipt.get("sha256") and io.sha(raw) == receipt["sha256"])
    doc["raw_hash_verified"] = raw_valid
    if not raw_valid:
        errors.append("RAW_FILE_MISSING_OR_HASH_MISMATCH")
    body_time = doc["title_available_at"]
    if raw_valid and raw.suffix.lower() == ".pdf":
        try:
            with pdfium.PdfDocument(raw) as pdf:
                meta = pdf.get_metadata_dict()
            doc["pdf_dates"] = {k: pdf_day(meta.get(k)) for k in ("CreationDate", "ModDate")}
            for day in doc["pdf_dates"].values():
                if day and day > entry["publishedDate"]:
                    # 只延后正文，不把后来版本冒充首次公告正文，也不因此删掉原公告标题。
                    body_time = max(body_time, next_midnight(day))
        except Exception as exc:
            errors.append("PDF_METADATA_UNREADABLE:" + type(exc).__name__)
    if not errors:
        doc["body"] = text
        doc["body_available_at"] = body_time
        doc["body_sha256"] = io.digest(text)
    return doc


def choose_report(reports, cutoff):
    report = holdings.select_report(reports, datetime.fromisoformat(cutoff))
    if report and (date.fromisoformat(cutoff[:10]) - date.fromisoformat(report["report_end"])).days <= 210:
        return report
    return None


def weights(report):
    """净资产占比除100，不把已披露前十大重新归一为全部仓位。"""
    return {h["stock_code"]: float(h["nav_weight_pct"]) / 100 for h in report["holdings"]} if report else {}


def event_weight(doc, cutoff, report, reports):
    """只用截止日已知信息；公司需在公告时与预测时两份已披露名单中均有关联。"""
    if doc["title_available_at"] > cutoff or doc.get("scope_exclusion"):
        return 0.0
    if doc["kind"] != "company":
        return 1.0
    then = choose_report(reports, doc["title_available_at"])
    return min(weights(then).get(doc["stock_code"], 0.0), weights(report).get(doc["stock_code"], 0.0))


def literal_facts(text):
    """固定词典提取原文片段，不调用大模型、不推断股价涨跌，保留触发短句供复核。"""
    found = {}
    compact = re.sub(r"\s+", "", text)
    for name, pattern in LEXICON.items():
        match = re.search(pattern, compact)
        if match:
            found[name] = compact[max(0, match.start() - 25):match.end() + 35]
    return found


def protocol(root=ROOT):
    """在读新一轮结果前固定六组，不因高低分追加参数搜索。"""
    path = root / "protocol.json"
    if path.exists():
        return io.read(path)
    value = {
        "created_at": io.now(), "version": "RECENT_BODY_INTEGRATION_V1",
        "purpose": "修复显示索引未展开正文；比较持仓、数量、标题、正文的实际增量",
        "candidates": ["RF_H", "RF_H_COUNTS", "RF_H_FACTS", "LR_H", "LR_H_TITLE", "LR_H_BODY"],
        "fixed_numeric": "原484行N8+NE7；H9原持仓行情加5项报告覆盖/年龄/集中度",
        "text": {"encoding": "char2-4_unsigned_hash", "hash_dimensions_per_kind": HASH_DIM,
                 "full_body": True, "daily_idf": "FIT_TRAINING_DAYS_ONLY", "windows": WINDOWS,
                 "decay_per_session": 0.9, "literal_dictionary": LEXICON},
        "source_range": [MIN_DAY, MAX_DAY], "catalog": str(INDEX),
        "date_only_available": "NEXT_CALENDAR_DAY_0000", "known_later_pdf_version": "DELAY_BODY_ONLY",
        "holding_link": "MIN_NAV_WEIGHT_AT_PUBLICATION_AND_PREDICTION; EACH_REPORT_MUST_BE_PUBLIC",
        "incomplete_body": "KEEP_TITLE_AND_MISSING_REASON; DO_NOT_TREAT_MISSING_AS_NEUTRAL_SENTIMENT",
        "evaluation": "相同2025年243日，已经反复研究的开发集；不是未见测试集",
        "cadence_sessions": 20, "training_labels": "MATURE_STRICTLY_BEFORE_UPDATE_AS_OF",
        "max_actual_supervised_fits": 79, "expected_fits": 78, "replay_fits": 1,
        "rf": {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 10, "random_state": 0},
        "lr": {"C": 1.0, "max_iter": 3000, "tol": 1e-8, "solver": "lbfgs", "random_state": 0},
        "selection": "CORRECT_DESC_ID_ASC; DESCRIPTIVE_ONLY_NO_ADOPTION",
        "new_2026_labels_or_scores": False, "external_requests": 0, "adoption": False,
        "limitations": ["存档重建，无法证明历史首见及完整修订历史",
                        "现有采集名单并非全市场历史新闻，仍有选择偏差",
                        "近年国家政策和独立新闻正文未因本次接线而自动补齐",
                        "固定短语和文字统计不能等同大模型完整理解正文"],
    }
    io.save(path, value)
    return value


def prepare(root=ROOT):
    """读取真正文、冻结来源并保存无目标标签的逐日输入；可安全复用已完成准备。"""
    if (root / "prepared.json").exists():
        return io.read(root / "prepared.json")
    protocol(root)
    snapshot = Snapshot(root)
    index = snapshot.read(INDEX)
    reports = snapshot.read(REPORTS)
    bundle = snapshot.read(BUNDLE)
    rows = read_lines(PREVIOUS / "inputs.jsonl")
    snapshot.read(PREVIOUS / "update-calendar.json")
    docs = []
    entries = [d for d in index["documents"] if MIN_DAY <= d.get("publishedDate", "") <= MAX_DAY]
    for i, entry in enumerate(entries):
        docs.append(resolve_document(entry, snapshot))
        if i % 500 == 0:
            print(f"正文校验 {i}/{len(entries)}", flush=True)
    docs.sort(key=lambda d: (d["published_date"], d["id"]))
    assert len({d["id"] for d in docs}) == len(docs)
    io.save_lines(root / "documents.jsonl", docs)
    io.save(root / "source-manifest.json", {"at": io.now(), "files": snapshot.files})
    io.save(root / "reports.json", reports)
    # 不复制/读取2026年监督标签。2026正文留在源快照与覆盖报告中，供下一阶段独立使用。
    io.save_lines(root / "base-inputs.jsonl", rows)
    io.save(root / "calendar.json", io.read(PREVIOUS / "update-calendar.json"))
    for doc in docs:
        doc["title_facts"] = literal_facts(doc["title"])
        doc["body_facts"] = literal_facts(doc["body"]) if doc["body"] else {}
    io.save_lines(root / "literal-evidence.jsonl", [
        {"id": d["id"], "body_sha256": d.get("body_sha256"),
         "title_facts": d["title_facts"], "body_facts": d["body_facts"]} for d in docs
    ])
    sessions = bundle["sessions"]
    available_indices = [bisect.bisect_left(sessions, d["title_available_at"][:10]) for d in docs]
    numeric, features, lineage = [], [], []
    wi, wj, wv, bi, bj, bv = [], [], [], [], [], []
    for ri, row in enumerate(rows):
        cutoff = row["as_of"]
        report = choose_report(reports, cutoff)
        end = bisect.bisect_left(sessions, cutoff[:10])
        # D08只见上一交易日收盘，原N/NE行保持不变。
        h, missing = holdings.holdings_features(
            report, bundle["stocks"], sessions[end - 21:end], sessions, datetime.fromisoformat(cutoff)
        )
        context = [float(report["disclosed_nav_pct"]) / 100, float(report["stock_nav_pct"]) / 100,
                   float(report["full_stock_disclosure"]),
                   (date.fromisoformat(cutoff[:10]) - date.fromisoformat(report["report_end"])).days,
                   sum(w ** 2 for w in weights(report).values())] if report else [None] * 5
        numeric.append(row["groups"]["N"] + row["groups"]["NE"] + (h or [None] * 9) + context)
        counts = np.zeros((len(KINDS), len(WINDOWS), 4))
        facts = np.zeros((2, len(KINDS), len(WINDOWS), len(LEXICON)))
        links = []
        for di, doc in enumerate(docs):
            age = end - available_indices[di]
            if not 0 <= age < 20:
                continue
            weight = event_weight(doc, cutoff, report, reports)
            if weight <= 0:
                continue
            body_ok = bool(doc["body"] and doc["body_available_at"] <= cutoff)
            decay = weight * 0.9 ** age
            kind = KINDS.index(doc["kind"])
            for period, window in enumerate(WINDOWS):
                if age >= window:
                    continue
                counts[kind, period] += [1, decay, decay if body_ok else 0, 0 if body_ok else decay]
                for mode, factset in enumerate((doc["title_facts"], doc["body_facts"] if body_ok else {})):
                    facts[mode, kind, period] += np.array([float(k in factset) for k in LEXICON]) * decay
            wi.append(ri)
            wj.append(di)
            wv.append(decay)
            if body_ok:
                bi.append(ri)
                bj.append(di)
                bv.append(decay)
            links.append({"id": doc["id"], "nav_weight": weight, "age_sessions": age,
                          "decayed_weight": decay, "body_used": body_ok})
        features.append({"counts": counts.flatten().tolist(), "facts": facts.flatten().tolist()})
        lineage.append({"target": row["target"], "as_of": cutoff,
                        "report_end": report["report_end"] if report else None,
                        "report_hash": report["raw"]["sha256"] if report else None,
                        "holding_missing": missing, "links": links})
    io.save_lines(root / "inputs.jsonl", [
        {**row, "numeric": nums, **feat} for row, nums, feat in zip(rows, numeric, features, strict=True)
    ])
    io.save_lines(root / "daily-lineage.jsonl", lineage)
    wt = sparse.csr_matrix((wv, (wi, wj)), shape=(len(rows), len(docs)))
    wb = sparse.csr_matrix((bv, (bi, bj)), shape=(len(rows), len(docs)))
    encoder = HashingVectorizer(analyzer="char", ngram_range=(2, 4), n_features=HASH_DIM,
                               alternate_sign=False, norm="l2", lowercase=False)
    for level, matrix in (("title", wt), ("body", wb)):
        blocks = []
        for kind in KINDS:
            selected = [i for i, d in enumerate(docs) if d["kind"] == kind and matrix[:, i].nnz > 0]
            # 只编码实际进入某一天的文本。2026正文不会参与2025的编码/词频拟合。
            encoded = encoder.transform([docs[i][level] for i in selected]) if selected else None
            blocks.append(matrix[:, selected] @ encoded if selected else sparse.csr_matrix((len(rows), HASH_DIM)))
        sparse.save_npz(root / (level + "-matrix.npz"), sparse.hstack(blocks, format="csr"))
    coverage = defaultdict(Counter)
    for di, d in enumerate(docs):
        c = coverage[d["published_date"][:4]]
        c["indexed"] += 1
        c["local_body_file"] += int("source_path" in d)
        c["complete_verified_body"] += int(bool(d["body"]))
        c["body_used_in_2024_2025_inputs"] += int(wb[:, di].nnz > 0)
        c[d["kind"] + "_indexed"] += 1
    prepared = {
        "at": io.now(), "coverage_by_year": dict(coverage),
        "body_exclusions": dict(Counter(e for d in docs for e in d["body_exclusions"])),
        "scope_exclusions": dict(Counter(d["scope_exclusion"] for d in docs if d.get("scope_exclusion"))),
        "body_delayed_by_known_version": sum(bool(d["body"] and d["body_available_at"] > d["title_available_at"])
                                              for d in docs),
        "rows": len(rows), "development_rows": sum(r["target"].startswith("2025") for r in rows),
        "development_days_with_body": sum(r["target"].startswith("2025") and wb[i].nnz > 0
                                           for i, r in enumerate(rows)),
        "body_document_day_links": wb.nnz, "numeric_columns": len(numeric[0]),
        "holding_complete_rows": sum(not r["holding_missing"] for r in lineage),
        "new_2026_labels_or_scores": False,
    }
    io.save(root / "prepared.json", prepared)
    return prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(io.canonical(prepare(args.root)), flush=True)
