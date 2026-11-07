"""从显式不可变原件重放仓位和地区事实，不接受派生账本直接作为原件证明。

沿用2026-10-19行业仓位提取器：原行百分比Decimal求和、每行0.005pp舍入界；
地区上下界为历史提取器的float汇总后Decimal传播，避免静默改动冻结准入边界。
所有文件经调用方的规范来源读取器逐次校验摘要，本模块不依赖研究目录。
"""

import re
from copy import deepcopy
from decimal import Decimal as D
from decimal import localcontext

from bs4 import BeautifulSoup

VERSION = "R5_EXPOSURE_ORIGINAL_REPLAY_V1"


def _same(observed, expected, label):
    if not D(observed).is_finite() or not D(expected).is_finite() or abs(D(observed) - D(expected)) > D("1e-24"):
        raise ValueError("ACTIVE_EXPOSURE_DERIVED_FACT_CONFLICT:" + label)


def _primary(raw, code, day):
    if raw.get("success") != 1 or code not in {r["stock"] for r in raw.get("data", {}).get("security", [])}:
        raise ValueError("ACTIVE_EXPOSURE_PRIMARY_IDENTITY")
    text = raw["data"].get("notice_content", "")
    start = text.find("§5")
    part = text[start:]
    if start < 0:
        raise ValueError("ACTIVE_EXPOSURE_PRIMARY_PORTFOLIO_BOUNDARY")
    if code == "007481":
        stop = re.search(r"5\.2\s*华夏逸享健康灵活配置", part)
        if stop:
            part = part[: stop.start()]
        if not re.search(r"2024\s*年\s*6\s*月\s*30\s*日", part[:500]):
            raise ValueError("ACTIVE_EXPOSURE_TRANSFORMATION_PERIOD")
    match = re.search(r"权益投资\s+([\d,]+\.\d{2})\s+([\d.]+)", part)
    if not match:
        raise ValueError("ACTIVE_EXPOSURE_PRIMARY_EQUITY_AMOUNT")
    stock = D(match[1].replace(",", ""))
    head = re.search("按行业分类的股票投资组合", part)
    if not head:
        raise ValueError("ACTIVE_EXPOSURE_PRIMARY_INDUSTRY_BOUNDARY")
    sector = part[head.start() :]
    stop = re.search(r"(?:期末按公允价值|报告期末按公允价值)", sector)
    if not stop:
        raise ValueError("ACTIVE_EXPOSURE_PRIMARY_INDUSTRY_END")
    totals = list(re.finditer(r"合\s*计\s+([\d,]+\.\d{2})\s+(\d+\.\d{2})", sector[: stop.start()]))
    if not 1 <= len(totals) <= 2:
        raise ValueError("ACTIVE_EXPOSURE_PRIMARY_TOTAL_COUNT")
    amounts = [D(m[1].replace(",", "")) for m in totals]
    if sum(amounts) != stock or stock <= 0:
        raise ValueError("ACTIVE_EXPOSURE_PRIMARY_EQUITY_RECONCILIATION")
    point = sum(D(m[2]) for m in totals)
    ratio = amounts[0] / stock * 100
    return point, D(".005") * len(totals), ratio, ratio, len(totals) == 1


def replay_exposures(inputs, root, read_source, read_bytes):
    """逐期重算并精确核对派生行；返回经原件复算的事实及审计计数。"""
    original = read_source(inputs, root, inputs["exposure_ref"])
    result = deepcopy(original)
    admission = {r["code"]: r for r in read_source(inputs, root, inputs["admission_ref"])}
    cache = {}
    hashes = set()
    quarters = 0
    primary_count = 0
    asset_points = {}
    for code, reference in inputs.get("asset_exposure_refs", {}).items():
        text = read_bytes(inputs, root, reference["source_ref"]).decode("utf-8", errors="strict")
        if reference["fund_code"] != code or code not in text:
            raise ValueError("ACTIVE_EXPOSURE_ASSET_IDENTITY")
        table = next(
            (t for t in BeautifulSoup(text, "html.parser").find_all("table") if "股票占净比" in t.get_text()), None
        )
        if table is None:
            raise ValueError("ACTIVE_EXPOSURE_ASSET_HEADER")
        points = {}
        for tr in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all("td")]
            if len(cells) >= 2 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[0]):
                if re.fullmatch(r"\d+\.\d+%", cells[1]):
                    points[cells[0]] = D(cells[1][:-1])
        asset_points[code] = points

    def source(ref):
        digest = ref["sha256"]
        if digest not in cache:
            status = ref.get("http_status", ref.get("status"))
            if status != 200:
                raise ValueError("ACTIVE_EXPOSURE_HTTP_NOT_SUCCESS")
            cache[digest] = read_source(inputs, root, digest)
            hashes.add(digest)
        return cache[digest]

    with localcontext() as context:
        # 冻结台账的地区边界由默认28位上下文形成；这里逐算式复现，非降低披露精度。
        context.prec = 28
        for item in result:
            code = item["code"]
            hk_free = {}
            seen = set()
            for quarter in item["quarters"]:
                day = quarter["date"]
                if day in seen:
                    raise ValueError("ACTIVE_EXPOSURE_DUPLICATE_QUARTER")
                seen.add(day)
                refs = quarter["source"]
                if quarter["basis"] == "PRIMARY_QUARTER_ORIGINAL_ALL_EQUITY":
                    point, radius, lo, hi, no_hk = _primary(source(refs), code, day)
                    primary_count += 1
                else:
                    if quarter["basis"] != "SAVED_INDUSTRY_LINES_ALL_EQUITY_SECONDARY_DISCLOSURE" or not isinstance(
                        refs, list
                    ):
                        raise ValueError("ACTIVE_EXPOSURE_UNKNOWN_DERIVATION")
                    matches = []
                    alias = item["portfolio_evidence_share_code"]
                    if alias not in admission.get(code, {}).get("share_codes", []):
                        raise ValueError("ACTIVE_EXPOSURE_PORTFOLIO_SHARE_IDENTITY")
                    for ref in refs:
                        raw = source(ref)
                        data = raw.get("Data")
                        if raw.get("ErrCode") != 0 or not isinstance(data, dict) or data.get("FundCode") != alias:
                            raise ValueError("ACTIVE_EXPOSURE_HYPZ_IDENTITY_OR_SUCCESS")
                        if str(ref.get("params", {}).get("fundCode")) != alias:
                            raise ValueError("ACTIVE_EXPOSURE_HYPZ_REQUEST_BINDING")
                        if raw.get("PageSize") not in [0, None] or raw.get("PageIndex") not in [0, None]:
                            raise ValueError("ACTIVE_EXPOSURE_HYPZ_PAGINATION_UNPROVEN")
                        if not isinstance(data.get("QuarterInfos"), list):
                            raise ValueError("ACTIVE_EXPOSURE_HYPZ_QUARTERS_REQUIRED")
                        matches.extend(q for q in data["QuarterInfos"] if q.get("JZRQ", "")[:10] == day)
                    if len(matches) != 1:
                        raise ValueError("ACTIVE_EXPOSURE_ORIGINAL_QUARTER_NOT_UNIQUE")
                    lines = matches[0]["HYPZInfo"]
                    if not lines or any(
                        r.get("BZDM") != alias or r.get("FSRQ", "")[:10] != day or r.get("ZJZBL") in [None, ""]
                        for r in lines
                    ):
                        raise ValueError("ACTIVE_EXPOSURE_HYPZ_LINE_IDENTITY_OR_PERCENT")
                    domestic = [r for r in lines if re.fullmatch("[A-S]", str(r["HYDM"]))]
                    hk = [r for r in lines if re.fullmatch(r"\d+", str(r["HYDM"]))]
                    if len(domestic) + len(hk) != len(lines):
                        raise ValueError("ACTIVE_EXPOSURE_UNKNOWN_INDUSTRY_SYSTEM")
                    point = sum(D(str(r["ZJZBL"])) for r in lines)
                    radius = D(".005") * len(lines)
                    if point <= radius:
                        raise ValueError("ACTIVE_EXPOSURE_NONPOSITIVE_DENOMINATOR")
                    d = D(str(sum(float(r["ZJZBL"]) for r in domestic)))
                    dr = D(".005") * len(domestic)
                    lo = max(D(0), d - dr) / (point + radius) * 100
                    hi = min(D(100), (d + dr) / (point - radius) * 100)
                    no_hk = not hk
                    stock_pct = asset_points.get(alias, {}).get(day)
                    if stock_pct is None:
                        if item["region_status_by_cutoff"]["100"].startswith("REGION_PASS"):
                            raise ValueError("ACTIVE_EXPOSURE_ALL_STOCK_RECONCILIATION_REQUIRED:" + code + ":" + day)
                    elif abs(point - stock_pct) > (D(len(lines)) + 1) * D(".005") + D(".00001"):
                        if not any(
                            g.get("day") == day and g.get("reason") == "ASSET_INDUSTRY_CONFLICT_WITHOUT_PRIMARY_REPAIR"
                            for g in item["gaps"]
                        ):
                            raise ValueError("ACTIVE_EXPOSURE_UNRECORDED_ASSET_INDUSTRY_CONFLICT:" + code + ":" + day)
                for label, value in [
                    ("equity_nav_pct", point),
                    ("rounding_radius_pp", radius),
                    ("domestic_ratio_lower_pct", lo),
                    ("domestic_ratio_upper_pct", hi),
                ]:
                    _same(value, quarter[label], code + ":" + day + ":" + label)
                    quarter[label] = str(value)
                hk_free[day] = no_hk
                quarters += 1
            for cutoff in [75, 80, 90, 100]:
                qs = item["quarters"]
                if any(D(q["domestic_ratio_upper_pct"]) < cutoff for q in qs):
                    state = "REGION_SEPARATE"
                elif len(qs) == 12 and not item["gaps"] and all(D(q["domestic_ratio_lower_pct"]) >= cutoff for q in qs):
                    state = "REGION_PASS"
                elif (
                    cutoff == 100
                    and len(qs) == 12
                    and not item["gaps"]
                    and all(
                        q["basis"].startswith("PRIMARY")
                        and D(q["domestic_ratio_lower_pct"]) == 100
                        or q["basis"].startswith("SAVED")
                        and D(q["domestic_ratio_upper_pct"]) == 100
                        and hk_free[q["date"]]
                        for q in qs
                    )
                ):
                    state = "REGION_PASS_NO_HK_ROWS_AND_TOTAL_RECONCILED"
                else:
                    state = "REGION_PENDING"
                if state != item["region_status_by_cutoff"][str(cutoff)]:
                    raise ValueError("ACTIVE_EXPOSURE_REGION_STATE_CONFLICT:" + code + ":" + str(cutoff))
                item["region_status_by_cutoff"][str(cutoff)] = state
    return result, {
        "version": VERSION,
        "products": len(result),
        "quarters": quarters,
        "original_source_count": len(hashes),
        "primary_quarters": primary_count,
        "original_sha256": sorted(hashes),
        "derived_rows_recomputed_and_compared": True,
    }
