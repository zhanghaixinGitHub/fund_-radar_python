"""业绩补采第二版：窗口去重、保守标题身份和有锚点的同比字段。

只处理公开资料，不导入模型或标签。版本一冻结实现保持原样。
"""

import re
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

from app.services.fund_information_history_v1 import normalize


def merge_windows(windows):
    """只合并同公司相交或相邻的登记窗口，保留所有基金/报告来源，不跨缺口扩窗。"""
    groups = defaultdict(list)
    for w in windows:
        if date.fromisoformat(w["start"]) > date.fromisoformat(w["end"]):
            raise ValueError("REVERSED_WINDOW")
        groups[w["stock"]].append(w)
    result = []
    for stock, entries in sorted(groups.items()):
        merged = []
        for w in sorted(entries, key=lambda x: (x["start"], x["end"])):
            source = {k: w[k] for k in ("fund_code", "report_sha256", "target_records")}
            if merged and w["start"] <= str(date.fromisoformat(merged[-1]["end"]) + timedelta(days=1)):
                merged[-1]["end"] = max(merged[-1]["end"], w["end"])
                merged[-1]["sources"].append(source)
            else:
                merged.append({"stock": stock, "start": w["start"], "end": w["end"], "sources": [source]})
        result.extend(merged)
    return result


def canonical_title(text):
    """仅统一明确同义字形；保留年份、季别、摘要/全文及更正标记，不用模糊匹配。"""
    text = normalize(text)
    digits = str.maketrans("〇○零一二三四五六七八九", "000123456789")
    text = re.sub(r"[二][〇○零][〇○零一二三四五六七八九]{2}(?=年)", lambda m: m.group().translate(digits), text)
    text = text.replace("年年度报告", "年度报告")
    text = re.sub(r"(?<=年)([一二三四])季度报告", r"第\1季度报告", text)
    return text


def identity(row, pages):
    """前两页同时有独立公司代码和等价报告标题才通过；英文/复杂修订留待核验。"""
    raw_front = "".join(pages[:2])
    front = canonical_title(raw_front)
    title = canonical_title(row["title_plain"])
    match = re.search(r"20\d{2}(?:年|年度)", title)
    # 目录常带证券简称前缀，删除前缀后仍保留完整报告身份与修订限定。
    anchor = title[match.start() :] if match else title
    code = row["secCode"]
    stock_ok = bool(re.search(r"(?<!\d)" + re.escape(code) + r"(?!\d)", raw_front))
    title_ok = bool(anchor) and anchor in front
    return {
        "passed": stock_ok and title_ok,
        "stock_verified": stock_ok,
        "catalog_title": row["title_plain"],
        "canonical_anchor": anchor,
        "title_verified": title_ok,
        "rule": "CODE_AND_EXPLICIT_EQUIVALENT_FULL_TITLE_IN_FIRST_TWO_PAGES",
    }


def review_yoy(spec, pages):
    """审核原文已经给出的同比百分比，不从负基数推算，不混为环比或百分点。

    reported_values 保存文字原值；增长/下降单独保存。只有明确增长/下降且
    原值非负，才给出带方向值；裸数或负基数时不推断经济含义。
    """
    if spec["comparison"] != "YEAR_ON_YEAR" or spec["unit"] != "PERCENT":
        raise ValueError("NOT_YEAR_ON_YEAR_PERCENT")
    if not spec.get("metric") or not spec.get("basis") or not spec.get("period_start") or not spec.get("period_end"):
        raise ValueError("YOY_BASIS_MISSING")
    if spec.get("direction_word") not in {"增长", "下降", "原文有符号数值"}:
        raise ValueError("YOY_DIRECTION_UNKNOWN")
    if spec.get("base_state") not in {"POSITIVE", "NEGATIVE", "ZERO", "UNKNOWN"}:
        raise ValueError("YOY_BASE_STATE_MISSING")
    page = pages[spec["page"] - 1]
    quote = normalize(spec["quote"])
    if not quote or normalize(page).count(quote) != 1 or "%" not in quote:
        raise ValueError("YOY_ANCHOR_INVALID")
    if "百分点" in quote:
        raise ValueError("PERCENTAGE_POINT_NOT_PERCENT_CHANGE")
    # 数值直接锚定百分号，避免把盈利金额或相邻年份当作同比。
    literals = re.findall(r"(?<![\d.])([+-]?\d+(?:\.\d+)?)%", quote.replace("−", "-"))
    wanted = [Decimal(str(v)) for v in spec["values"]]
    if len(wanted) not in (1, 2) or any(not v.is_finite() for v in wanted):
        raise ValueError("INVALID_YOY_VALUES")
    # 区间分隔符 - 与负号在同一行可能重合：人工字段须以原文明确增长/下降解释，保留原值。
    available = [Decimal(v) for v in literals]
    for value in wanted:
        if value not in available:
            raise ValueError("YOY_VALUE_NOT_IN_ANCHOR")
    if spec["direction_word"] in {"增长", "下降"} and spec["direction_word"] not in quote:
        raise ValueError("YOY_DIRECTION_NOT_IN_ANCHOR")
    signed = None
    if spec["base_state"] == "POSITIVE" and all(v >= 0 for v in wanted):
        factor = Decimal(-1) if spec["direction_word"] == "下降" else Decimal(1)
        if spec["direction_word"] in {"增长", "下降"}:
            signed = [str(v * factor) for v in wanted]
    return {
        **spec,
        "reported_values": [str(v) for v in wanted],
        "signed_percent": signed,
        "anchor": {"page": spec["page"], "offset": normalize(page).index(quote), "text": quote},
        "derived_from_money": False,
        "training_ready": False,
    }
