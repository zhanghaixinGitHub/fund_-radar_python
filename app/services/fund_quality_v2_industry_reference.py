"""从协会按证券代码公布的行业分类表重建证券映射，不替代发行人映射或完整V2指标。

只接受正文中每行明确印出的门类代码，不从基金名称、指数名称或相邻行继承分类。
观察期与发布日期分别记录；事后取得的分类不能被宣称为观察期末已知的信息。
"""

import hashlib
import io
import re
from datetime import date

from pypdf import PdfReader

from app.services.fund_quality_v2_index_reference import equity_slice_distance
from app.services.fund_rating_rules import decimal


def parse_capco_security_pages(pages: list[str], *, period_end: str) -> dict:
    """解析按股票代码排序的完整分类表，返回逐证券、逐页的原文定位。

    门类使用原表A至S单字母代码；制造业的次类CG等不作为门类。每个六位
    证券行都必须单独解析，重复、缺列、空页或标题观察期不符时整件拒绝。
    """
    end = date.fromisoformat(period_end)
    half = "上半年" if (end.month, end.day) == (6, 30) else "下半年"
    if (end.month, end.day) not in {(6, 30), (12, 31)} or not 1 <= len(pages) <= 1000:
        raise ValueError("QUALITY_INDUSTRY_PERIOD_OR_PAGE_COUNT")
    title = f"{end.year}年{half}上市公司行业分类结果"
    if title not in re.sub(r"\s+", "", pages[0]):
        raise ValueError("QUALITY_INDUSTRY_TITLE_PERIOD_MISMATCH")
    result = {}
    for page_number, text in enumerate(pages, 1):
        # 独立行首六位数字标识证券行；不要求名称紧贴代码，以兼容PDF提取空格。
        starts = list(re.finditer(r"(?m)^\s*([0-9]{6})(?=\D)", text))
        if not starts:
            raise ValueError(f"QUALITY_INDUSTRY_PAGE_WITHOUT_ROWS:{page_number}")
        for i, start in enumerate(starts):
            block = text[start.start() : starts[i + 1].start() if i + 1 < len(starts) else len(text)]
            match = re.match(r"\s*([0-9]{6})\s*(.*?)\s+([A-S])\s+(.+)", block, re.S)
            if not match or not match[2].strip() or not re.search(r"\b[0-9]{2}\b", match[4]):
                raise ValueError(f"QUALITY_INDUSTRY_ROW_INCOMPLETE:{page_number}:{start[1]}")
            code, name, industry = match[1], re.sub(r"\s+", "", match[2]), match[3]
            if code in result:
                raise ValueError(f"QUALITY_INDUSTRY_DUPLICATE_SECURITY:{code}")
            result[code] = {
                "security_code": code,
                "security_name": name,
                "industry_code": industry,
                "page": page_number,
                "excerpt": block.strip(),
            }
    return {
        "period_end": period_end,
        "classification_level": "CAPCO_PRIMARY_CATEGORY_A_S",
        "security_count": len(result),
        "page_count": len(pages),
        "securities": result,
        "scope": "SECURITY_CLASSIFICATION_NOT_ISSUER_IDENTITY",
    }


def rebuild_capco_reference(raw: bytes, *, expected_sha256: str, period_end: str, published_on: str) -> dict:
    """核对实际PDF内容摘要并重读全表；发布日期与观察期分别保留。

    来源URL、取得时间、发布正文与附件关联仍由外层原件回执校验，摘要相符
    只是文件一致性证据，不能独自证明来源、使用权限或事前可得性。
    """
    if len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("QUALITY_INDUSTRY_SOURCE_HASH_OR_SIZE")
    if date.fromisoformat(published_on) < date.fromisoformat(period_end):
        raise ValueError("QUALITY_INDUSTRY_PUBLICATION_BEFORE_OBSERVATION")
    reader = PdfReader(io.BytesIO(raw))
    result = parse_capco_security_pages([page.extract_text() or "" for page in reader.pages], period_end=period_end)
    result.update(source_hash=expected_sha256, published_on=published_on)
    return result


def equity_industry_diagnostic(
    positions: list[dict],
    reference: dict,
    classification: dict,
    *,
    expected_stock_amount: str,
    disclosed_industries: list[dict] | None = None,
) -> dict:
    """同日股票金额完整对账后比较行业切片；指数缺任一成分分类即拒绝。

    只覆盖股票部分，不能作为合同全资产目标、现金目标或最终行业复制偏离。
    指数披露的舍入残差原样保留，不把合计强制归一到1。
    实际持仓可直接使用基金原件完整行业金额表；未上市新股缺证券映射时仍
    保留缺口，不把行业表剩余金额反推为该证券的行业或发行人证明。
    """
    equity_slice_distance(positions, reference, expected_stock_amount=expected_stock_amount)
    if reference["observed_on"] != classification["period_end"]:
        raise ValueError("QUALITY_INDUSTRY_REFERENCE_DATE_MISMATCH")
    securities = classification["securities"]
    actual, target, mapping = {}, {}, {}
    unmapped_actual = []
    total = decimal(expected_stock_amount)
    for code in {row["stock_code"] for row in positions} | reference["weights"].keys():
        if not re.fullmatch(r"[0-9]{6}\.(?:SH|SZ|BJ)", code):
            raise ValueError(f"QUALITY_INDUSTRY_SECURITY_UNMAPPED:{code}")
        if code[:6] not in securities:
            if code in reference["weights"] or disclosed_industries is None:
                raise ValueError(f"QUALITY_INDUSTRY_SECURITY_UNMAPPED:{code}")
            unmapped_actual.append(code)
            continue
        mapping[code] = securities[code[:6]]["industry_code"]
    known_amounts = {}
    for row in positions:
        if row["stock_code"] in mapping:
            industry = mapping[row["stock_code"]]
            known_amounts[industry] = known_amounts.get(industry, 0) + decimal(row["market_value_cny"])
    if disclosed_industries is not None:
        amounts = {}
        for row in disclosed_industries:
            industry, amount = row["industry_code"], decimal(row["value_cny"])
            if not re.fullmatch(r"[A-S]", industry) or amount < 0:
                raise ValueError("QUALITY_INDUSTRY_DISCLOSED_ROW_INVALID")
            # 同一门类可能按指数投资/积极投资分表披露，必须相加而非字典覆盖。
            amounts[industry] = amounts.get(industry, 0) + amount
        if abs(sum(amounts.values()) - total) > decimal("0.02"):
            raise ValueError("QUALITY_INDUSTRY_DISCLOSED_AMOUNTS_INCOMPLETE")
        if any(value - amounts.get(industry, 0) > decimal("0.02") for industry, value in known_amounts.items()):
            raise ValueError("QUALITY_INDUSTRY_DISCLOSED_AND_SECURITY_MAPPING_CONFLICT")
        actual = {key: value / total for key, value in amounts.items()}
    else:
        actual = {key: value / total for key, value in known_amounts.items()}
    for code, weight in reference["weights"].items():
        industry = mapping[code]
        target[industry] = target.get(industry, 0) + decimal(weight)
    return {
        "status": "EQUITY_INDUSTRY_SLICE_DIAGNOSTIC_NOT_V2_METRIC",
        "observed_on": reference["observed_on"],
        "classification_published_on": classification["published_on"],
        "actual_industries": {key: str(value) for key, value in sorted(actual.items())},
        "target_industries": {key: str(value) for key, value in sorted(target.items())},
        "half_l1_distance": str(
            sum(abs(actual.get(key, 0) - target.get(key, 0)) for key in actual.keys() | target.keys()) / 2
        ),
        "mapped_security_count": len(mapping),
        "unmapped_actual_securities": sorted(unmapped_actual),
        "actual_industry_basis": "ORIGINAL_REPORT_COMPLETE_INDUSTRY_AMOUNTS"
        if disclosed_industries is not None
        else "COMPLETE_SECURITY_MAPPING",
        "classification_source_hash": classification["source_hash"],
        "limits": ["股票部分之外的完整资产目标尚未核实", "证券分类不等于发行人身份", "未完成跨产品校准和独立验收"],
    }
