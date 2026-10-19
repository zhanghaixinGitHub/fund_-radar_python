"""新报告持仓适配；季报只展示本期披露部分，不用旧中报补成当期完整组合。"""

import re
import unicodedata
from decimal import Decimal
from pathlib import Path

from app.integrations.fund_report_layout_v3 import parse_text
from app.integrations.market_fund_reports import read_source
from app.services.fund_holdings_publication import _fund_name_matches, _public_table, report_from_original
from app.services.fund_materials_build import safe_url
from app.services.fund_rating_public_market import public_document, publication


def disclosed_stocks(rows: list[dict], text: str) -> list[dict]:
    """同为6月/12月的公开表可能已经更新为中报/年报，只取原季报明确列出的证券。"""
    sections = []
    heading = r"前十名(?:股票(?:及存托凭证)?|权益)\s*投资\s*明细"
    for match in re.finditer(heading, text):
        following = text[match.end() :]
        stop = re.search(r"(?m)^\s*[5-8]\.\d+(?:\.\d+)?\s+[\u4e00-\u9fff]", following)
        sections.append(following[: stop.start()] if stop else following[:8000])
    selected = []
    for row in rows:
        code = row["stockCode"].removesuffix(".SH").removesuffix(".SZ").removesuffix(".HK")
        if any(re.search(r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])", s) for s in sections):
            selected.append(row)
    if not selected or len(selected) > 20:
        raise ValueError("MARKET_QUARTER_DISCLOSED_STOCK_IDENTITY")
    return selected


def disclosed_single_fund(text: str, title: str) -> list[dict]:
    """解析仅一只基金的直接投资表；跨列金额只拼接明确数值列，不推算、不借用旧期。"""
    found = list(re.finditer(r"(?m)^\s*5\.\d+(?:\.\d+)?\s+[^\n]*(?:基金投资明细|投资目标基金明细)", text))
    if not found:
        raise ValueError("MARKET_QUARTER_FUND_SECTION")
    following = text[found[-1].end() :]
    stop = re.search(r"(?m)^\s*5\.\d+\s+[\u4e00-\u9fff]", following)
    body = following[: stop.start()] if stop else following[:8000]
    lines = body.splitlines()
    ranked = [(i, line) for i, line in enumerate(lines) if re.match(r"^\s*\d{1,2}\s+", line)]
    if len(ranked) != 1 or not re.match(r"^\s*1\s+", ranked[0][1]):
        raise ValueError("MARKET_QUARTER_FUND_MULTIPLE_OR_EMPTY")
    index, line = ranked[0]
    numbers = list(re.finditer(r"(?<![A-Za-z0-9])\d[\d,.]*(?![A-Za-z0-9])", line))
    if not numbers or not re.fullmatch(r"\d+\.\d{2}", numbers[-1][0]):
        raise ValueError("MARKET_QUARTER_FUND_WEIGHT")
    weight = numbers[-1]
    lower = max(15, weight.start() - 35)
    pieces = [
        m[0]
        for i, current in enumerate(lines)
        for m in re.finditer(r"(?<![A-Za-z0-9])(?:\d[\d,.]*|,[\d,.]+)(?![A-Za-z0-9])", current)
        if m.start() >= lower
        and not (i == index and m.start() == weight.start())
        and not current.lstrip().startswith("注")
    ]
    amount = "".join(pieces)
    if not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}", amount):
        raise ValueError("MARKET_QUARTER_FUND_AMOUNT_AMBIGUOUS")
    value, ratio = Decimal(amount.replace(",", "")), Decimal(weight[0])
    if value <= 0 or not 0 < ratio <= 100:
        raise ValueError("MARKET_QUARTER_FUND_RANGE")
    if "联接" in title:
        name = title.split("联接", 1)[0].removesuffix("发起式")
        core = re.sub(r"交易型开放式.*|ETF.*", "", name)
        # 原表一行被类型、管理人等列打断时，按原文顺序逐字核对目标名称核心。
        # 只允许已绑定报告身份、且表中明确只有一只基金的联接产品使用此路径。
        remaining = iter(re.sub(r"\s+", "", body))
        if len(core) < 6 or not all(any(char == candidate for candidate in remaining) for char in core):
            raise ValueError("MARKET_QUARTER_TARGET_FUND_NAME")
    else:
        kind = re.search(r"\bETF\b", line)
        if not kind:
            raise ValueError("MARKET_QUARTER_FUND_NAME")
        name = " ".join(
            current[6 : kind.start()].strip()
            for current in lines
            if current[6 : kind.start()].strip()
            and not any(
                word in current
                for word in ("基金名称", "序号", "金额单位", "基金类型", "公允价值", "净值比例", "人民币", "（%）")
            )
        )
    return [{"stockCode": "", "stockName": name, "weightPct": float(ratio), "marketValue": float(value)}]


def parse_report(client, root: Path, fund: dict, entry: dict, source: dict) -> tuple[dict, list[dict]]:
    """正文身份和报告期先核验；完整报告沿用资产负债表对账，季报独立适配。"""
    code = fund["fund_code"]
    if entry["kind"] != "QUARTER":
        report, issues = report_from_original(code, fund["fund_name"], root, source)
        # 仅有缺项的新报告才查同报告期补充表；已完整收录的报告不会走到这里。
        if any(i["part"] in {"stocks", "bond_amount"} for i in issues):
            tables = []
            for issue in issues:
                if issue["part"] in {"stocks", "bond_amount"}:
                    tables.append(client.table(code, entry["end"], issue["part"]))
            source = {**source, "tables": tables}
            report, issues = report_from_original(code, fund["fund_name"], root, source)
        report["sourceArticleId"] = entry["id"]
        return report, issues

    pages, reprint = public_document(read_source(root, source), source, code, "季度报告")
    text = "\n".join(pages)
    if code not in text or not _fund_name_matches(text, fund["fund_name"]):
        raise ValueError("MARKET_QUARTER_IDENTITY")
    published = publication(pages)
    if published != entry["published"] or (reprint and reprint["notice_date"][:10] != published):
        raise ValueError("MARKET_QUARTER_PUBLICATION")
    title = re.sub(r"\s+", "", unicodedata.normalize("NFKC", entry["title"]))
    if title not in re.sub(r"\s+", "", unicodedata.normalize("NFKC", text[:4000])):
        raise ValueError("MARKET_QUARTER_TITLE")
    report = {
        "id": source["sha256"],
        "sourceArticleId": entry["id"],
        "title": entry["title"],
        "endDate": entry["end"],
        "publishedDate": published,
        "sourceUrl": safe_url((reprint or {}).get("attach_url") or source["url"]),
        "fullDisclosure": False,
        "stockWeightPct": None,
        "disclosedWeightPct": 0,
        "holdings": [],
        "assets": [],
        "industries": [],
        "investmentGroups": [],
        "holdingsPublicationVersion": 1,
    }
    issues = []
    try:
        # 名称取目录已绑定的报告标题前缀，避免把ETF简称误当另一只基金。
        name = re.split(r"[0-9〇零一二三四五六七八九]{4}\s*年", entry["title"])[0]
        parsed = parse_text(pages, entry["title"], fund_code=code, fund_name=name, master_code=code)
        if parsed["report_end"] != entry["end"]:
            raise ValueError("MARKET_QUARTER_PERIOD")
        report["holdings"] = [
            {
                "stockCode": h["stock_code"],
                "stockName": h["stock_name"],
                "weightPct": float(h["nav_weight_pct"]),
                "marketValue": float(h["market_value_cny"]),
            }
            for h in parsed["holdings"]
        ]
        report["stockWeightPct"] = float(parsed["stock_nav_pct"])
        report["assets"] = [
            {"name": k, "weightPct": float(v["total_asset_pct"]), "value": float(v["value_cny"])}
            for k, v in parsed["assets"].items()
        ]
        report["industries"] = [
            {"name": v["name"], "weightPct": float(v["nav_weight_pct"])} for v in parsed["reported_industries"]
        ]
    except ValueError:
        # 公开转载表仍要核对基金、报告期、人民币单位和原表精度；不能按最新行情冒充持仓。
        table = client.table(code, entry["end"], "stocks")
        try:
            report["holdings"] = disclosed_stocks(_public_table(root, table, code, entry["end"], None, None), text)
        except ValueError:
            # 明确披露未直接持股才可接受空表；抓取失败不能自动解释为零持仓。
            if not re.search(r"本基金本报告期末未持有股票", text):
                issues.append({"part": "stocks", "reason": "MARKET_QUARTER_STOCK_TABLE_UNAVAILABLE"})
    # 季报的债券明细本来就可能只有前五名，不能用其合计推断完整债券敞口。
    bonds = client.table(code, entry["end"], "bond_amount")
    try:
        rows = _public_table(root, bonds, code, entry["end"], None, None)
        report["investmentGroups"].append(
            {"name": "债券持仓", "notice": "本期报告主要持仓，可能仅含前五名。", "holdings": rows}
        )
    except ValueError:
        # 来源无表不代表没有债券；只有正文明确显示持有而解析缺失时记录待补。
        if re.search(r"债券投资\s+[\d,]+\.\d{2}\s+[\d.]+", text):
            issues.append({"part": "bonds", "reason": "MARKET_QUARTER_BOND_TABLE_UNAVAILABLE"})
    if "联接" in fund["fund_name"] or "FOF" in fund["fund_name"] or "基金投资明细" in text:
        try:
            rows = disclosed_single_fund(text, entry["title"])
            report["investmentGroups"].append(
                {"name": "基金持仓", "notice": "本基金直接持有的基金，未穿透底层证券。", "holdings": rows}
            )
        except ValueError:
            if "联接" in fund["fund_name"] or "FOF" in fund["fund_name"]:
                issues.append({"part": "funds", "reason": "MARKET_QUARTER_FUND_TABLE_NOT_VERIFIED"})
    report["disclosedWeightPct"] = sum(h["weightPct"] for h in report["holdings"])
    report["holdingsNotice"] = "季度报告披露的部分持仓，不代表完整或实时持仓。"
    if issues:
        report["holdingsNotice"] += "部分投资明细暂未核齐，相关缺口保留。"
    if not report["holdings"] and not report["investmentGroups"] and not report["assets"]:
        raise ValueError("MARKET_QUARTER_NO_VERIFIED_HOLDINGS")
    return report, issues
