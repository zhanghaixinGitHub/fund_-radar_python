"""综合资料中可核验的公开净值补充；不改写供应商净值库或冒用其来源身份。"""

import hashlib
import json
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.services.direction_1d_protocol import digest

POLICY = "OFFICIAL_PUBLISHER_NAV_DOCUMENT_V1"
DISTRIBUTOR_POLICY = "DISCLOSED_DISTRIBUTOR_NAV_DOCUMENT_V1"


def supplement_nav(
    profile: dict, material: dict, documents: dict, observed: list, window: dict
) -> tuple[list, dict | None]:
    """只接纳本基金已实际取得且通过对应披露契约的原件，补充当前比较基准日。

    净值仍只作比较基准及历史观察，不参与独立决定方向。本函数不采集、不写
    nav_daily、不登记或启用新供应商；新增行保留自己的文档身份和实际取得时间。
    管理人官网格式和依法披露的销售机构格式分别验证身份、原始响应与字段口径。
    供应商已有同日值时必须一致，存在冲突立即拒绝，不能用补充值静默覆盖冲突。
    """
    evidence = material.get("nav_supplement")
    if evidence is None:
        return observed, None
    if isinstance(evidence, dict) and evidence.get("policy") == DISTRIBUTOR_POLICY:
        return _distributor_nav(profile, material, documents, observed, window)
    if not isinstance(evidence, dict) or evidence.get("policy") != POLICY:
        raise ValueError("COMPREHENSIVE_NAV_SUPPLEMENT_INVALID")
    document = documents.get(evidence.get("document_ref"), {})
    code, base = material["fund_code"], window["base_nav_date"]
    uri = urlparse(document.get("url", ""))
    # 当前只实现已经核实过字段定义的易方达官网格式；新管理人须有自己的真实适配。
    if (
        profile.get("fund_code") != code
        or profile.get("manager_name") != "易方达基金管理有限公司"
        or uri.scheme != "https"
        or uri.hostname != "cdn.efunds.com.cn"
        or uri.path != f"/market/2.0/{code}_1y.js"
        or uri.query
        or uri.username
        or uri.password
        or document.get("kind") != "FUND_NAV_DATA"
        or document.get("codes") != [code]
        or document.get("body_scope") != "FULL_TEXT"
        or document.get("receipt_http") != 200
    ):
        raise ValueError("COMPREHENSIVE_NAV_PUBLISHER_MISMATCH")
    try:
        raw = Path(document["raw_path"]).read_bytes()
        if (
            len(raw) > 1_000_000
            or hashlib.sha256(raw).hexdigest() != document["raw_sha256"]
            or raw.decode("utf-8-sig").strip() != document["body"].strip()
            or digest(document["body"]) != document["body_hash"]
        ):
            raise ValueError("COMPREHENSIVE_NAV_ORIGINAL_MISMATCH")
        body = re.fullmatch(rf'\s*(?:var\s+)?mk_{code}_1y\s*=\s*"([^"]+)";?\s*', document["body"])
        if not body:
            raise ValueError("COMPREHENSIVE_NAV_FORMAT_INVALID")
        fields = body[1].split(";")
        # 官网前五列分别为日期、累计收益、单位净值、累计净值、日增长率。
        if fields[0] != "0_1_2_3_4":
            raise ValueError("COMPREHENSIVE_NAV_FORMAT_INVALID")
        rows = {}
        for value in fields[1:]:
            if not value:
                continue
            day, _return, unit, cumulative, _change = value.split("_")
            day = date.fromisoformat(day).isoformat()
            unit, cumulative = Decimal(unit), Decimal(cumulative)
            if day in rows or not unit.is_finite() or unit <= 0 or not cumulative.is_finite() or cumulative <= 0:
                raise ValueError("COMPREHENSIVE_NAV_VALUE_INVALID")
            rows[day] = (unit, cumulative)
        available = datetime.fromisoformat(document["available_at"])
        if available.tzinfo is None or base not in rows or date.fromisoformat(base) > available.date():
            raise ValueError("COMPREHENSIVE_NAV_BASE_REQUIRED")
        if any(date.fromisoformat(day) > available.date() for day in rows):
            raise ValueError("COMPREHENSIVE_NAV_FUTURE_ROW")
        unit, cumulative = rows[base]
    except (KeyError, TypeError, OSError, UnicodeError, InvalidOperation) as error:
        raise ValueError("COMPREHENSIVE_NAV_SUPPLEMENT_INVALID") from error
    current = [row for row in observed if row["nav_date"] == base]
    if any(Decimal(row["unit_nav"]) != unit for row in current):
        raise ValueError("COMPREHENSIVE_NAV_SOURCE_CONFLICT")
    proof = {
        "policy": POLICY,
        "source_code": "EFUNDS_OFFICIAL_PUBLIC_DOCUMENT",
        "document_ref": document["id"],
        "url": document["url"],
        "body_hash": document["body_hash"],
        "raw_sha256": document["raw_sha256"],
        "observed_at": document["available_at"],
        "nav_date": base,
        "unit_nav": str(unit),
        "accum_nav": str(cumulative),
        "provider_same_day_value_present": bool(current),
        "nav_daily_unchanged": True,
    }
    if current:
        return observed, proof
    addition = {
        "nav_date": base,
        "unit_nav": str(unit),
        "accum_nav": str(cumulative),
        "ann_date": None,
        # 不伪造 source_registry 外键、供应商版本编号或首次公告日。
        "source_id": None,
        "source_code": proof["source_code"],
        "source_ref": document["id"],
        "upstream_hash": document["body_hash"],
        "version_id": "material:" + digest(proof),
        "content_hash": digest(proof),
        "observed_at": document["available_at"],
        "expires_at": window["deadline_at"],
    }
    return sorted([*observed, addition], key=lambda row: row["nav_date"]), proof


def _distributor_nav(profile: dict, material: dict, documents: dict, observed: list, window: dict) -> tuple[list, dict]:
    """核验已使用的销售机构公开净值，不因官网暂时不可达而冒用供应商身份。

    必须有本基金现行招募说明书允许销售机构披露净值的条款、本基金报告与实际
    份额身份、销售机构自己的基金页面及未改写的净值响应。页面与响应的当前日、
    单位及累计值须一致，历史重叠日期也须与已核验供应商逐日一致。仅补当前日，
    分红折算等特殊情形不在此适配范围。该函数不向原净值库写入任何记录。
    """
    evidence, code = material["nav_supplement"], material["fund_code"]
    source = documents.get(evidence.get("document_ref"), {})
    identity = documents.get(evidence.get("identity_ref"), {})
    prospectus = documents.get(evidence.get("prospectus_ref"), {})
    report = documents.get(evidence.get("report_ref"), {})
    uri, site = urlparse(source.get("url", "")), urlparse(identity.get("url", ""))
    query = parse_qs(uri.query, keep_blank_values=True)
    clause, title = evidence.get("disclosure_quote", ""), evidence.get("fund_title", "")
    if (
        profile.get("fund_code") != code
        or profile.get("fund_type") not in {"STOCK", "MIXED", "BOND"}
        or not profile.get("manager_name")
        or re.sub(r"\s+", "", profile["manager_name"]) not in re.sub(r"\s+", "", prospectus.get("body", ""))
        or len(title) < 10
        or re.sub(r"\s+", "", title) not in re.sub(r"\s+", "", prospectus.get("body", ""))
        or re.sub(r"\s+", "", title) not in re.sub(r"\s+", "", report.get("body", ""))
        or code not in report.get("body", "")
        or not clause
        or clause not in prospectus.get("body", "")
        or not all(term in clause for term in ["基金销售机构网站", "基金份额净值", "基金份额累计净值"])
        or prospectus.get("kind") != "PROSPECTUS"
        or report.get("kind") != "REPORT"
        or code not in prospectus.get("codes", [])
        or code not in report.get("codes", [])
        or urlparse(prospectus.get("url", "")).hostname != "np-cnotice-fund.eastmoney.com"
        or identity.get("kind") != "FUND_DISTRIBUTOR_IDENTITY"
        or identity.get("codes") != [code]
        or site.scheme != "https"
        or site.hostname != "fund.eastmoney.com"
        or site.path != f"/{code}.html"
        or site.query
        or site.fragment
        or site.username
        or site.password
        or not all(term in identity.get("body", "") for term in [code, "上海天天基金销售有限公司", "证监会批准"])
        or source.get("kind") != "FUND_NAV_DATA"
        or source.get("codes") != [code]
        or source.get("body_scope") != "FULL_TEXT"
        or source.get("receipt_http") != 200
        or uri.scheme != "https"
        or uri.hostname != "api.fund.eastmoney.com"
        or uri.path != "/f10/lsjz"
        or uri.username
        or uri.password
        or uri.fragment
        or query.get("fundCode") != [code]
        or query.get("pageIndex") != ["1"]
        or not set(query) <= {"fundCode", "pageIndex", "pageSize", "startDate", "endDate"}
    ):
        raise ValueError("COMPREHENSIVE_NAV_DISTRIBUTOR_IDENTITY_INVALID")
    try:
        raw = Path(source["raw_path"]).read_bytes()
        if (
            len(raw) > 1_000_000
            or hashlib.sha256(raw).hexdigest() != source["raw_sha256"]
            or raw.decode("utf-8-sig").strip() != source["body"].strip()
            or digest(source["body"]) != source["body_hash"]
        ):
            raise ValueError("COMPREHENSIVE_NAV_ORIGINAL_MISMATCH")
        response = json.loads(source["body"])
        values, rows = response["Data"]["LSJZList"], {}
        if response["ErrCode"] != 0 or response["PageIndex"] != 1 or not 1 <= len(values) <= 100:
            raise ValueError("COMPREHENSIVE_NAV_FORMAT_INVALID")
        available, base = datetime.fromisoformat(source["available_at"]), window["base_nav_date"]
        for item in values:
            day = date.fromisoformat(item["FSRQ"]).isoformat()
            unit, cumulative = Decimal(item["DWJZ"]), Decimal(item["LJJZ"])
            if (
                day in rows
                or day > available.date().isoformat()
                or not unit.is_finite()
                or unit <= 0
                or not cumulative.is_finite()
                or cumulative <= 0
                or item.get("NAVTYPE") != "1"
                or any(item.get(key) for key in ["FHFCBZ", "FHFCZ", "FHFCZ10", "FHSP"])
            ):
                raise ValueError("COMPREHENSIVE_NAV_VALUE_OR_ACTION_INVALID")
            rows[day] = (unit, cumulative)
        if available.tzinfo is None or base not in rows:
            raise ValueError("COMPREHENSIVE_NAV_BASE_REQUIRED")
        unit, cumulative = rows[base]
        header = re.search(r"单位净值\s*[（(]\s*(\d{4}-\d{2}-\d{2})\s*[）)]\s*([0-9]+(?:\.[0-9]+)?)", identity["body"])
        total = re.search(r"累计净值\s*([0-9]+(?:\.[0-9]+)?)", identity["body"])
        if (
            not header
            or header[1] != base
            or Decimal(header[2]) != unit
            or not total
            or Decimal(total[1]) != cumulative
        ):
            raise ValueError("COMPREHENSIVE_NAV_PAGE_CONFLICT")
        previous = [row for row in observed if row["nav_date"] < base]
        if not previous or max(row["nav_date"] for row in previous) not in rows:
            raise ValueError("COMPREHENSIVE_NAV_PROVIDER_OVERLAP_REQUIRED")
        overlap = []
        for row in observed:
            if row["nav_date"] not in rows:
                continue
            if Decimal(row["unit_nav"]) != rows[row["nav_date"]][0]:
                raise ValueError("COMPREHENSIVE_NAV_SOURCE_CONFLICT")
            overlap.append({key: row.get(key) for key in ["nav_date", "unit_nav", "source_id", "content_hash"]})
    except (KeyError, TypeError, OSError, UnicodeError, InvalidOperation) as error:
        raise ValueError("COMPREHENSIVE_NAV_SUPPLEMENT_INVALID") from error
    current = [row for row in observed if row["nav_date"] == base]
    proof = {
        "policy": DISTRIBUTOR_POLICY,
        "source_code": "TTFUND_PUBLIC_DISCLOSED_NAV_DOCUMENT",
        "document_ref": source["id"],
        "url": source["url"],
        "body_hash": source["body_hash"],
        "raw_sha256": source["raw_sha256"],
        "observed_at": source["available_at"],
        "nav_date": base,
        "unit_nav": str(unit),
        "accum_nav": str(cumulative),
        "provider_same_day_value_present": bool(current),
        "nav_daily_unchanged": True,
        "identity_ref": identity["id"],
        "identity_hash": identity["body_hash"],
        "identity_url": identity["url"],
        "prospectus_ref": prospectus["id"],
        "prospectus_hash": prospectus["body_hash"],
        "disclosure_quote": clause,
        "report_ref": report["id"],
        "report_hash": report["body_hash"],
        "provider_overlap": overlap,
    }
    if current:
        return observed, proof
    addition = {
        "nav_date": base,
        "unit_nav": str(unit),
        "accum_nav": str(cumulative),
        "ann_date": None,
        "source_id": None,
        "source_code": proof["source_code"],
        "source_ref": source["id"],
        "upstream_hash": source["body_hash"],
        "version_id": "material:" + digest(proof),
        "content_hash": digest(proof),
        "observed_at": source["available_at"],
        "expires_at": window["deadline_at"],
    }
    return sorted([*observed, addition], key=lambda row: row["nav_date"]), proof
