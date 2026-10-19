"""从已留存的公开定期报告补齐市场持仓；离线核验后发布，不改评级或预测输入。"""

import hashlib
import json
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

from bs4 import BeautifulSoup

from app.core.config import get_settings
from app.integrations.public_fund_reports import report_period
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_build import safe_url
from app.services.fund_materials_store import versioned_save
from app.services.fund_rating_public_market import balance, matches_name, public_document, publication, stock_pages
from app.services.fund_rating_public_stock_tables import parse_disclosed


def _fund_name_matches(text: str, name: str) -> bool:
    """目录简称与报告全称仅做确定的术语展开；管理人前缀须在报告中再次核对。"""

    def normalize(value):
        value = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))
        for term in (
            "交易型开放式指数证券投资基金",
            "交易型开放式证券投资基金",
            "ETF",
            "发起式",
            "证券投资基金",
            "基金",
            "(QDII)",
        ):
            value = value.replace(term, "")
        return re.sub(r"(股票|混合|债券|指数)型", r"\1", value)

    stem = normalize(name.split("-")[0])
    body = normalize(text)
    return stem in body or (
        len(stem) > 6
        and stem[2:] in body
        and bool(re.search(r"基金管理人[：:]" + re.escape(name[:2]) + "基金", re.sub(r"\s+", "", text)))
    )


def _section(pages: list[str], heading: str) -> str | None:
    """只取投资组合正文，不把目录页、买卖明细或附注中的数字混入期末持仓。"""
    text = "\n".join(stock_pages(pages))
    starts = list(re.finditer(r"(?m)^\s*7\.(\d+)\s+([^\n]+)", text))
    for i in range(len(starts) - 1, -1, -1):
        match = starts[i]
        if re.search(heading, match[2]) and not re.search(r"\.{3}|…", match[2]):
            stop = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            return text[match.end() : stop]
    return None


def _number(value: str) -> Decimal:
    result = Decimal(value.replace(",", ""))
    if not result.is_finite() or result < 0:
        raise ValueError("HOLDINGS_AMOUNT_INVALID")
    return result


def _security_rows(body: str, net: Decimal) -> list[dict]:
    """读取标准债券/基金表；跨行名称保留，金额和占比必须由原文明示并相互核对。"""
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    start = re.compile(r"^(\d{1,4})\s+(\d{5,9})\s+(.+)$")
    starts = [i for i, line in enumerate(lines) if start.match(line)]
    rows = []
    for ordinal, i in enumerate(starts):
        stop = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(lines)
        match = start.match(lines[i])
        if int(match[1]) != ordinal + 1:
            raise ValueError("HOLDINGS_ROW_SEQUENCE")
        block = " ".join([match[3], *lines[i + 1 : stop]])
        # 表尾注释不是名称的一部分；数量列可有可无，市值必须精确到分。
        block = re.split(r"注[：:]|7\.\d+", block)[0]
        values = list(re.finditer(r"(?<![\d.,])([\d,]+\.\d{2})\s+([\d.]+)\s*(?:%|$)", block))
        if len(values) != 1:
            raise ValueError("HOLDINGS_ROW_CELLS")
        tail = values[0]
        value, weight = map(_number, tail.groups())
        if abs(value / net * 100 - weight) > Decimal(".006"):
            raise ValueError("HOLDINGS_ROW_WEIGHT")
        name = re.sub(r"\s+[\d,]+(?:\.\d+)?\s*$", "", block[: tail.start()]).strip()
        if not name:
            raise ValueError("HOLDINGS_NAME_EMPTY")
        rows.append({"stockCode": match[2], "stockName": name, "weightPct": float(weight), "marketValue": float(value)})
    if not rows or len({r["stockCode"] for r in rows}) != len(rows):
        raise ValueError("HOLDINGS_TABLE_EMPTY_OR_DUPLICATE")
    return rows


def _public_table(
    root: Path, ref: dict, code: str, end: str, expected: Decimal | None, net: Decimal | None
) -> list[dict]:
    """转载明细按报告期核对；万元精度只允许原表逐行舍入误差，不补缺行或猜币种。"""
    path = (root / ref["file"]).resolve()
    if not path.is_relative_to(root) or path.stat().st_size > 8_000_000:
        raise ValueError("HOLDINGS_TABLE_PATH")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"] or ref.get("fund_code") != code:
        raise ValueError("HOLDINGS_TABLE_HASH_OR_IDENTITY")
    match = re.search(r'content:("(?:\\.|[^"\\])*")', raw.decode("utf-8"))
    if not match:
        raise ValueError("HOLDINGS_TABLE_RESPONSE")
    soup = BeautifulSoup(json.loads(match[1]), "html.parser")
    for box in soup.select(".boxitem"):
        if not any(re.search(r"/" + code + r"\.html(?:$|\?)", link.get("href", "")) for link in box.select("h4 a")):
            continue
        if not any(node.get_text(strip=True) == end for node in box.select(".t font")):
            continue
        headers = [h.get_text("", strip=True) for h in box.select("th")]
        if not headers or "万元" not in headers[-1] or any(unit in headers[-1] for unit in ("美元", "港元", "日元")):
            raise ValueError("HOLDINGS_TABLE_CURRENCY")
        rows = []
        for tr in box.select("tbody tr"):
            cells = [td.get_text("", strip=True) for td in tr.select("td")]
            if not cells:
                continue
            weight = next((c for c in cells[3:] if c.endswith("%")), None)
            if int(cells[0]) != len(rows) + 1 or not weight:
                raise ValueError("HOLDINGS_TABLE_ROW")
            value, ratio = _number(cells[-1]) * 10000, _number(weight.rstrip("%"))
            if ratio > 100 or (
                net is not None and abs(value / net * 100 - ratio) > Decimal(".006") + Decimal(50) / net * 100
            ):
                raise ValueError("HOLDINGS_TABLE_WEIGHT")
            rows.append(
                {"stockCode": cells[1], "stockName": cells[2], "weightPct": float(ratio), "marketValue": float(value)}
            )
        if not rows or len({h["stockCode"] for h in rows}) != len(rows):
            raise ValueError("HOLDINGS_TABLE_ROWS")
        amount = sum(Decimal(str(r["marketValue"])) for r in rows)
        tolerance = Decimal(50) * len(rows)
        if expected is not None and ref.get("kind") == "stocks" and abs(amount - expected) > tolerance:
            raise ValueError("HOLDINGS_TABLE_INCOMPLETE")
        if expected is not None and amount > expected + tolerance:
            raise ValueError("HOLDINGS_TABLE_EXCEEDS_BALANCE")
        return rows
    raise ValueError("HOLDINGS_TABLE_PERIOD_OR_FUND")


def report_from_original(code: str, name: str, root: Path, ref: dict) -> tuple[dict, list[dict]]:
    """核对原件哈希、产品身份、披露日期及金额；缺少某类明细时保留准确资产结构和缺口。"""
    root = root.resolve()
    path = (root / ref["file"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 25_000_000:
        raise ValueError("HOLDINGS_ORIGINAL_PATH_OR_SIZE")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"] or not safe_url(ref["url"]):
        raise ValueError("HOLDINGS_ORIGINAL_HASH_OR_URL")
    if ref.get("table_only"):
        # 公告目录仍在，但正文暂不可读：仅发布独立公开表的实际行，完整性与股票仓位保持未知。
        metadata = json.loads(raw)
        entry = metadata.get("data") or {}
        title, published = entry.get("notice_title", ""), entry.get("notice_date", "")[:10]
        period = report_period(title)
        if (
            metadata.get("success") != 1
            or code not in {s.get("stock") for s in entry.get("security", [])}
            or not _fund_name_matches(title, name)
            or not period
            or not period[0] <= published <= str(now().date())
        ):
            raise ValueError("HOLDINGS_UNAVAILABLE_REPORT_IDENTITY")
        rows = _public_table(root, ref["tables"][0], code, period[0], None, None)
        return {
            "id": ref["sha256"],
            "title": title,
            "endDate": period[0],
            "publishedDate": published,
            "sourceUrl": safe_url(entry.get("attach_url")),
            "fullDisclosure": False,
            "stockWeightPct": None,
            "disclosedWeightPct": sum(h["weightPct"] for h in rows),
            "holdings": rows,
            "assets": [],
            "industries": [],
            "investmentGroups": [],
            "holdingsNotice": "按公开来源列示持仓明细；报告原文暂不可读，"
            "持仓完整性与资产配置尚待核对。不代表实时持仓。",
            "holdingsPublicationVersion": 1,
        }, [{"part": "report", "reason": "HOLDINGS_ORIGINAL_BODY_UNAVAILABLE"}]
    pages, reprint = public_document(raw, ref, code, "报告")
    text = "\n".join(pages)
    if code not in text or not (matches_name(text, name.split("-")[0]) or _fund_name_matches(text, name)):
        raise ValueError("HOLDINGS_FUND_IDENTITY")
    # 目录中的“摘要”二字不能使完整报告失效；只识别首页报告标题。
    title = re.search(r"[0-9〇零一二三四五六七八九]{4}年(?:中期|半年度|年度)报告", re.sub(r"\s+", "", text[:1500]))
    period = report_period(title[0]) if title else None
    published = publication(pages)
    if not period or period[1] not in {"HALF", "ANNUAL"} or not period[0] <= published <= str(now().date()):
        raise ValueError("HOLDINGS_REPORT_DATE")
    if reprint and reprint.get("notice_date", "")[:10] != published:
        raise ValueError("HOLDINGS_PUBLICATION_CONFLICT")
    b = balance(pages, require_stock=False)
    net, total, stock = map(Decimal, (b["net_assets"], b["total_assets"], b["stock_amount"]))
    holdings, industries, issues = [], [], []
    stock_valuation_adjustment = "0"
    if stock > 0:
        # 只规范报告标题中的“所有/全部”，不修改证券行或原始文件。
        normalized = [re.sub(r"(?:所有|全部)股票\s*投资明细", "股票投资明细", page) for page in pages]
        try:
            table = parse_disclosed(normalized, str(stock), str(net))
            holdings, industries = table["holdings"], table["reported_industries"]
            stock_valuation_adjustment = table["substitution_increment_cny"]
        except ValueError:
            # 海外表采用不同证券代码，独立解析器仍要求完整股票金额和行业表对账。
            from app.services.fund_rating_public_global import positions as global_positions
            from app.services.fund_rating_public_japan import positions as japan_positions

            for reader in (global_positions, japan_positions):
                try:
                    holdings, industries = reader(pages, str(stock))
                    break
                except ValueError:
                    continue
            if not holdings:
                issues.append({"part": "stocks", "reason": "HOLDINGS_STOCK_TABLE_NOT_VERIFIED"})
    rows = []
    for h in holdings:
        value = Decimal(h["market_value_cny"])
        rows.append(
            {
                "stockCode": h["stock_code"],
                "stockName": h["stock_name"],
                "weightPct": float(value / net * 100),
                "marketValue": float(value),
            }
        )
    rows.sort(key=lambda h: h["marketValue"], reverse=True)
    for supplement in ref.get("tables", []):
        if supplement["kind"] == "stocks" and not rows and stock > 0:
            rows = _public_table(root, supplement, code, period[0], stock, net)
            issues = [i for i in issues if i["part"] != "stocks"]
    groups = []
    for key, label, heading in (
        ("bond_amount", "债券持仓", "前五名债券投资明细"),
        ("fund_amount", "基金持仓", "基金投资明细|持有.*基金.*明细|投资.*基金.*明细"),
    ):
        expected = Decimal(b[key])
        if expected == 0:
            continue
        body = _section(pages, heading)
        try:
            positions = _security_rows(body or "", net)
            if sum(Decimal(str(p["marketValue"])) for p in positions) > expected + Decimal(".02"):
                raise ValueError("HOLDINGS_GROUP_AMOUNT")
            groups.append({"name": label, "notice": "按定期报告列示，可能仅含主要持仓。", "holdings": positions})
        except ValueError:
            supplemental = next((t for t in ref.get("tables", []) if t["kind"] == key), None)
            if supplemental:
                positions = _public_table(root, supplemental, code, period[0], expected, net)
                groups.append(
                    {"name": label, "notice": "报告主要持仓，市值按来源万元精度列示。", "holdings": positions}
                )
            else:
                issues.append({"part": key, "reason": "HOLDINGS_GROUP_NOT_VERIFIED"})
    # 人工核对的跨列基金表与原报告片段精确绑定；金额与同报告基金资产按分对账。
    reviewed = ref.get("reviewed_fund_table")
    if reviewed:
        body = _section(pages, "基金投资明细|投资目标基金明细|投资基金情况")
        if not body or hashlib.sha256(body.encode("utf-8")).hexdigest() != reviewed["section_sha256"]:
            raise ValueError("HOLDINGS_REVIEWED_SECTION_CHANGED")
        positions = reviewed["holdings"]
        amount = sum(Decimal(str(h["marketValue"])) for h in positions)
        if not positions or abs(amount - Decimal(b["fund_amount"])) > Decimal(".02"):
            raise ValueError("HOLDINGS_REVIEWED_FUND_TOTAL")
        for h in positions:
            if not h["stockName"] or abs(
                Decimal(str(h["marketValue"])) / net * 100 - Decimal(str(h["weightPct"]))
            ) > Decimal(".006"):
                raise ValueError("HOLDINGS_REVIEWED_FUND_WEIGHT")
        groups = [g for g in groups if g["name"] != "基金持仓"]
        groups.append({"name": "基金持仓", "notice": "本基金直接持有的基金，未穿透至底层证券。", "holdings": positions})
        issues = [i for i in issues if i["part"] != "fund_amount"]
    allocations = [
        ("股票", stock),
        ("基金", Decimal(b["fund_amount"])),
        ("债券", Decimal(b["bond_amount"])),
        ("现金及结算资金", sum(Decimal(b["assets"][k]) for k in ("货币资金", "结算备付金", "存出保证金"))),
    ]
    remainder = total - sum(value for _, value in allocations)
    if remainder < 0:
        raise ValueError("HOLDINGS_ASSET_RECONCILIATION")
    allocations.append(("其他资产", remainder))
    notice = "报告披露持仓，不代表实时持仓。"
    if not rows and stock == 0:
        notice += "本报告期末未直接持有股票。"
    if any(i["part"] == "stocks" for i in issues):
        notice += "股票明细暂未核齐，股票仓位与资产配置按报告保留。"
    if any(i["part"] != "stocks" for i in issues):
        notice += "部分债券或基金明细暂未核齐，相关投资已计入资产配置。"
    report = {
        "id": ref["sha256"],
        "title": ref.get("title") or name + "定期报告",
        "endDate": period[0],
        "publishedDate": published,
        "sourceUrl": safe_url((reprint or {}).get("attach_url") or ref["url"]),
        "fullDisclosure": bool(rows) and not any(i["part"] == "stocks" for i in issues),
        "stockWeightPct": float(stock / net * 100),
        "disclosedWeightPct": sum(h["weightPct"] for h in rows),
        "holdings": rows,
        "assets": [
            {"name": label, "weightPct": float(value / total * 100), "value": float(value)}
            for label, value in allocations
            if value
        ],
        "industries": [
            {"name": h["name"], "weightPct": float(Decimal(h["value_cny"]) / net * 100)} for h in industries
        ],
        "investmentGroups": groups,
        "holdingsNotice": notice,
        "holdingsPublicationVersion": 1,
    }
    if Decimal(stock_valuation_adjustment) != 0:
        # 差额属于股票资产列报中的非证券估值项目，不并入任何一只证券的市值。
        report["stockValuationAdjustment"] = {
            "name": "可退替代款估值差额",
            "valueCny": stock_valuation_adjustment,
            "basis": "资产负债表股票金额减去已独立对账的全部证券金额",
        }
    return report, issues


def publish_report(code: str, report: dict, *, directory: Path, dry_run: bool = False) -> dict:
    """所有自动/离线入口共用历史合并；同一原件解析变窄时不降低已验证披露覆盖。"""
    if not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("HOLDINGS_FUND_CODE")
    path = directory / (code + ".json")
    previous = read(path) if path.exists() else {}
    if previous and previous.get("fundCode") != code:
        raise ValueError("HOLDINGS_SNAPSHOT_IDENTITY")
    reports = {r["id"]: r for r in previous.get("reports", [])}
    old = reports.get(report["id"])
    if old is None:
        reports[report["id"]] = report
    elif old.get("holdingsPublicationVersion") == 1:

        def identities(value):
            rows = value.get("holdings", []) + [h for g in value.get("investmentGroups") or [] for h in g["holdings"]]
            return {(h.get("stockCode"), h.get("stockName")) for h in rows}

        if identities(old) <= identities(report) and not (old["fullDisclosure"] and not report["fullDisclosure"]):
            reports[report["id"]] = report
    ordered = sorted(
        reports.values(), key=lambda r: (r["endDate"], r["publishedDate"], r["fullDisclosure"]), reverse=True
    )
    if old is not None and ordered == previous.get("reports"):
        # 原件与解析结果未变时复用已有快照；检查时间由维护状态单独记录。
        return previous
    latest_codes = {h["stockCode"] for h in ordered[0].get("holdings", []) if h.get("stockCode")}
    value = {
        **previous,
        "fundCode": code,
        "reports": ordered,
        "asOfDate": max(previous.get("asOfDate") or "", report["publishedDate"]),
        "collectedAt": now().isoformat(),
        "companies": [
            {**c, "latestHeld": c["stockCode"] in latest_codes} if "latestHeld" in c else c
            for c in previous.get("companies", [])
        ],
        "documents": previous.get("documents", []),
        "notice": previous.get("notice") or "持仓来自公开定期报告，不代表实时持仓。",
    }
    json.dumps(value, allow_nan=False)
    if not dry_run:
        versioned_save(path, value)
    return value


def publish(manifest: dict, *, directory: Path | None = None, dry_run: bool = False) -> dict:
    """逐基金隔离失败；旧快照只追加报告，已有公司资料和历史报告始终保留。"""
    directory = directory or Path(get_settings().fund_material_directory)
    results = []
    for item in manifest["funds"]:
        code = item["fund_code"]
        try:
            if not re.fullmatch(r"[0-9]{6}", code):
                raise ValueError("HOLDINGS_FUND_CODE")
            report, issues = report_from_original(code, item["fund_name"], Path(item["source_root"]), item["source"])
            publish_report(code, report, directory=directory, dry_run=dry_run)
            results.append(
                {
                    "fund_code": code,
                    "status": "PARTIAL_SUCCESS" if issues else "SUCCEEDED",
                    "holdings": len(report["holdings"]),
                    "groups": len(report["investmentGroups"]),
                    "report_date": report["endDate"],
                    "issues": issues,
                }
            )
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError) as exc:
            results.append({"fund_code": code, "status": "FAILED", "reason": str(exc)})
    return {"dry_run": dry_run, "funds": results}
