"""境内主动权益基金的公开原件适配；完整八维通过后才允许试算。

沿用已有绝对区间，另建不可变口径，不改变 002112、016008 的首个规则和原件。
净值使用已授权来源的复权单位净值，不能用累计净值冒充再投资总回报。
"""

import hashlib
import io
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pypdf import PdfReader

from app.core.config import get_settings
from app.integrations.fund_report_layout_v3 import parse_text
from app.services.fund_rating_disclosures import compact
from app.services.fund_rating_metrics import history_metrics, standard_cost
from app.services.fund_rating_rules import decimal

BASIS = "PUBLIC_DISCLOSURE_DOMESTIC_EQUITY_V1"
SCOPE = {"basis": BASIS, "asset": "DOMESTIC_ACTIVE_EQUITY", "currency": "CNY"}
HOSTS = {
    "www.scfund.com.cn",
    "www.zofund.com",
    "www.fullgoal.com.cn",
    "wap.fullgoal.com.cn",
    "www.cmfchina.com",
    "static.cmfchina.com",
    "fund.chinaamc.com",
    "www.chinaamc.com",
    "www.dcfund.com.cn",
    "www.wjasset.com",
    "www.cxfund.com.cn",
    "cxfund.com.cn",
    "www.bdfund.cn",
    "bdfund.cn",
    "www.99fund.com",
    "www.hsfund.com",
    "www.essencefund.com",
    "www.efunds.com.cn",
    "www.founderff.com",
    "founderff.com",
    "www.dfham.com",
    "www.maxwealthfund.com",
    "owa.maxwealthfund.com",
    "www.fsfund.com",
    "api.fsfund.com",
    # 摩根公开产品及本份额净值；仍需逐个原件哈希、产品身份与日期核验。
    "www.cifm.com",
    "ecmob.cifm.com",
    "www.yhfund.com.cn",
    "www.gtfund.com",
    "st.gtfund.com",
    "www.thfund.com.cn",
    "ilcapp.tianhongjijin.com.cn",
    "api.fund.eastmoney.com",
    "www.gffunds.com.cn",
    "tushare.pro",
    "np-cnotice-fund.eastmoney.com",
    "pdf.dfcfw.com",
}

ATTACHMENT_HOST = "pdf.dfcfw.com"
REPRINT_HOST = "np-cnotice-fund.eastmoney.com"


def attachment_identity(refs: dict, root: Path, label: str, code: str, required_title: str | None = None) -> dict:
    """dfcfw 法定披露附件按东财公告目录条目逐件绑定，不整域信任。

    附件 URL 的公告号必须与同包归档的东财公告记录一致：目录条目存在、
    公告号与基金代码匹配、附件路径与目录条目登记一致；缺少绑定记录即拒绝。
    """
    ref = refs[label]
    names = set(re.findall(r"AN\d{10,}", urlsplit(ref["url"]).path))
    if len(names) != 1:
        raise ValueError("RATING_MARKET_ATTACHMENT_IDENTITY")
    article = names.pop()
    record_ref = refs.get(label + "_identity")
    if record_ref is None or urlsplit(record_ref["url"]).hostname != REPRINT_HOST:
        raise ValueError("RATING_MARKET_ATTACHMENT_IDENTITY")
    raw = (root / record_ref["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record_ref["sha256"]:
        raise ValueError("RATING_MARKET_ORIGINAL_HASH")
    record = json.loads(raw)
    data = record.get("data") or {}
    attach = urlsplit(data.get("attach_url") or "")
    if (
        record.get("success") != 1
        or record_ref.get("article_id") != article
        or data.get("art_code") != article
        or code not in {s.get("stock") for s in data.get("security", [])}
        or attach.hostname != ATTACHMENT_HOST
        or attach.path != urlsplit(ref["url"]).path
        or not data.get("notice_date")
        or required_title is not None
        and required_title not in data.get("notice_title", "")
    ):
        raise ValueError("RATING_MARKET_ATTACHMENT_IDENTITY")
    return data


@lru_cache(maxsize=24)
def _pdf_pages(raw: bytes) -> tuple[str, ...]:
    """同一批原件按字节复用文本；不同字节不会命中旧缓存。"""
    return tuple(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)


def public_document(raw: bytes, ref: dict, code: str, required_title: str) -> tuple[list[str], dict | None]:
    """保留发行人 PDF 与完整公开转载的区别，转载必须核对正文标识及基金身份。"""
    if raw.startswith(b"%PDF"):
        return list(_pdf_pages(raw)), None
    if raw.startswith(b"PK\x03\x04"):
        # 部分管理人只发布 DOCX。只读取标准正文 XML，不执行宏、关系或外部链接。
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if sum(info.file_size for info in archive.infolist()) > 50_000_000:
                raise ValueError("RATING_MARKET_DOCUMENT_EXPANDED_SIZE")
            xml = archive.read("word/document.xml")
        if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
            raise ValueError("RATING_MARKET_DOCUMENT_XML_ENTITY")
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        body = ET.fromstring(xml).find("w:body", ns)
        if body is None:
            raise ValueError("RATING_MARKET_DOCUMENT_BODY")
        lines = []
        for child in body:
            if child.tag.endswith("}tbl"):
                lines.extend(
                    " ".join("".join(cell.itertext()) for cell in row.findall("w:tc", ns))
                    for row in child.findall("w:tr", ns)
                )
            else:
                lines.append("".join(child.itertext()))
        return ["\n".join(lines)], None
    reprint = json.loads(raw)
    data = reprint.get("data") or {}
    if (
        reprint.get("success") != 1
        or code not in {s.get("stock") for s in data.get("security", [])}
        or required_title not in data.get("notice_title", "")
        or len(data.get("notice_content", "")) < 1500
        or ref.get("article_id") != data.get("art_code")
    ):
        raise ValueError("RATING_MARKET_REPRINT_IDENTITY_OR_COMPLETENESS")
    return [data["notice_content"]], data


def publication(pages: list[str]) -> str:
    """只识别报告首页明确送出日期，不把信息截至日、抓取日或正文任免日当公布日。"""
    text = compact("\n".join(pages))[:4000]
    found = re.search(r"送出日期[：:]?(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not found:
        found = re.search(r"送出日期[：:]?(\d{4})-(\d{2})-(\d{2})", text)
    if found:
        return date(*map(int, found.groups())).isoformat()
    chinese = re.search(
        r"送出日期[：:]?([〇零0一二三四五六七八九]{4})年([一二三四五六七八九十]+)月([一二三四五六七八九十]+)日", text
    )
    if not chinese:
        raise ValueError("RATING_MARKET_PUBLICATION_DATE_MISSING")
    digits = dict(zip("零一二三四五六七八九", range(10), strict=True))

    def number(s):
        if "十" in s:
            left, right = s.split("十")
            return (digits[left] if left else 1) * 10 + (digits[right] if right else 0)
        return digits[s]

    year, month, day = chinese.groups()
    return date(
        int(year.translate(str.maketrans("〇零一二三四五六七八九", "00123456789"))), number(month), number(day)
    ).isoformat()


def matches_name(text: str, name: str) -> bool:
    """统一全半角括号与可选上市标识；基金核心名称及份额代码仍必须核对。"""

    def normalize(value):
        return compact(unicodedata.normalize("NFKC", value)).replace("(LOF)", "")

    return normalize(name) in normalize(text)


def market_policy() -> dict:
    from app.services.fund_rating_public_policy import policy

    value = policy()
    value.update(basis=BASIS, revision="public-domestic-equity-2026-10-14-v1", public_evidence_version=1)
    value["scope"] = SCOPE
    value["input_contract"] = {
        "history": "AUTHORIZED_PROVIDER_ADJUSTED_UNIT_NAV_ALL_DOMESTIC_SESSIONS",
        "holdings": "FULL_REPORT_STOCKS_AND_REPORTED_INDUSTRIES_RECONCILED_TO_BALANCE",
        "minimum_report_stock_ratio": "0.60",
        "strategy": "EXPLICIT_NO_MATERIAL_STRATEGY_CHANGE",
        "liquidity": "RESTRICTED_SECURITIES_CASH_LARGE_HOLDER_DISCLOSURE_AND_PRODUCT_ASSETS",
        "unknowns": "REJECT_NO_FILL_NO_WEIGHT_REDISTRIBUTION",
    }
    return value


def section(text: str, start: str, end: str) -> str:
    matches = list(re.finditer(start, text))
    if not matches:
        raise ValueError("RATING_MARKET_SECTION_MISSING:" + start)
    a = matches[-1].start()
    stop = re.search(end, text[matches[-1].end() :])
    if not stop:
        raise ValueError("RATING_MARKET_SECTION_END_MISSING:" + end)
    return text[a : matches[-1].end() + stop.start()]


def amount(text: str, label: str, *, first: bool = False) -> str:
    lines = [s for s in text.splitlines() if re.match(r"^\s*" + re.escape(label) + r"\s", s)]
    if not lines or (not first and len(lines) != 1):
        raise ValueError("RATING_MARKET_AMOUNT_ROW:" + label)
    tail = re.sub(r"^\s*" + re.escape(label) + r"\s+", "", lines[0])
    tail = re.sub(r"^[56]\.4\.\d+(?:\.\d+)?\s+", "", tail)
    m = re.match(r"([\d,]+\.\d{2}|[-－—])(?:\s|$)", tail)
    if not m:
        raise ValueError("RATING_MARKET_AMOUNT_VALUE:" + label)
    return "0" if m[1] in "-－—" else m[1].replace(",", "")


def balance(pages: list[str], *, require_stock: bool = True) -> dict:
    from app.services.fund_report_sections import numbered_pages

    pages = numbered_pages(pages)
    text = section("\n".join(pages), r"[56]\.1\s*资产负债表", r"[56]\.2\s*利润表")
    if "单位：人民币元" not in compact(text):
        raise ValueError("RATING_MARKET_BALANCE_UNIT")
    core = (
        "货币资金",
        "结算备付金",
        "存出保证金",
        "交易性金融资产",
        "衍生金融资产",
        "买入返售金融资产",
        "应收清算款",
        "应收股利",
        "应收申购款",
        "递延所得税资产",
        "其他资产",
    )
    assets = {k: amount(text, k) for k in core}
    for name in ("债权投资", "其他债权投资", "其他权益工具投资"):
        if re.search(r"(?m)^\s*" + name + r"\s", text):
            assets[name] = amount(text, name)
    result = {
        "assets": assets,
        "total_assets": amount(text, "资产总计"),
        "liabilities": amount(text, "负债合计"),
        "net_assets": amount(text, "净资产合计"),
        "stock_amount": amount(text, "其中：股票投资"),
        "fund_amount": amount(text, "基金投资", first=True),
        "bond_amount": amount(text, "债券投资", first=True),
    }
    if abs(sum(map(decimal, assets.values())) - decimal(result["total_assets"])) > Decimal(".02") or abs(
        decimal(result["total_assets"]) - decimal(result["liabilities"]) - decimal(result["net_assets"])
    ) > Decimal(".02"):
        raise ValueError("RATING_MARKET_BALANCE_RECONCILIATION")
    if decimal(result["net_assets"]) <= 0:
        raise ValueError("RATING_MARKET_NONPOSITIVE_NET_ASSETS")
    if require_stock and (
        decimal(result["fund_amount"]) != 0
        or decimal(result["stock_amount"]) / decimal(result["net_assets"]) < Decimal(".60")
    ):
        raise ValueError("RATING_MARKET_REQUIRES_OTHER_ASSET_POLICY")
    return result


def disclosures(pages: list[str], positions: list[dict], *, include_holder: bool = True) -> dict:
    text = "\n".join(pages)
    governance = compact(section(text, r"(?:9|10)\.[67]\.1\s*管理人受调查或处罚", r"(?:9|10)\.[67]\.2"))
    if "受到的具体措施类型" in governance and "受到调查或处罚等措施的原因" in governance:
        # 结构化监管事项优先于节内其他否定句，避免“无其他事项”掩盖已披露措施。
        governance_state = (
            1 if re.search(r"(?:完成整改工作|已完成整改|措施完成整改|整改结果已经监管部门验收通过)", governance) else 0
        )
    elif re.search(
        r"(?:无受调查或处罚|不存在受调查或处罚|未受到(?:相关监管部门)?调查或处罚|未发生管理人及其高级管理人员受稽查或处罚|未发生基金管理人受调查或处罚|等情况无[。.]?)",
        governance,
    ):
        governance_state = 2
    else:
        raise ValueError("RATING_MARKET_GOVERNANCE_REVIEW_REQUIRED")
    strategy = compact(section(text, r"(?:9|10)\.4\s*基金投资策略", r"(?:9|10)\.5"))
    if not re.search(
        r"(?:投资策略(?:在本报告期内)?(?:未发生重大改变|未发生改变|未改变|无改变|未发生变更|没有重大改变)|(?:未发生|无)基金投资策略的改变|基金投资策略[^。]{0,120}未发生显著的改变)",
        strategy,
    ):
        raise ValueError("RATING_MARKET_STRATEGY_UNVERIFIED")
    holder_matches = list(re.finditer(r"\d+\.\d+\s*报告期内单一投资者持有", text))
    if include_holder and not holder_matches:
        raise ValueError("RATING_MARKET_HOLDER_DISCLOSURE_MISSING")
    holder = text[holder_matches[-1].start() :] if holder_matches else ""
    holder = (
        holder[: re.search(r"\n\s*\d+\.\d+\s*影响投资者", holder).start()]
        if re.search(r"\n\s*\d+\.\d+\s*影响投资者", holder)
        else holder[:2500]
    )
    h = compact(holder)
    band = None
    if not include_holder:
        pass
    elif re.search(
        r"(?:未出现|无单一投资者|不存在达到|未发生|未有|不存在单一投资者|不存在持有).*?(?:20%|百分之二十)", h
    ) or re.search(r"达到或超过20%的情况无[。.]?$", h):
        band = 0
    else:
        # 只读取持有人表格；其后的通用风险文字中的50%不是本基金实际持有比例。
        holder_table = holder.split("产品特有风险", 1)[0]
        percentages = [decimal(v) for v in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)%", holder_table)]
        if not percentages or max(percentages) < 20:
            raise ValueError("RATING_MARKET_HOLDER_TABLE_UNVERIFIED")
        if max(percentages) >= 50:
            band = 2
        else:
            # 表中“份额占比”是期末值，不能证明整个报告期最高值低于 50%。
            raise ValueError("RATING_MARKET_HOLDER_PEAK_RANGE_UNKNOWN")
    # “无”只在对应正文节内有效；其他章节的否定陈述不能拿来填缺项。
    suspended = section(text, r"[56]\.4\.12\.2\s*期末持有的暂时停牌", r"[56]\.4\.12\.3")
    lending = section(text, r"[56]\.4\.12\.4\s*期末参与转融通", r"[56]\.4\.13")
    if not re.search(
        r"(?:未持有(?:期末)?参与|无参与).*?转融通证券出借业务的(?:融出)?证券|无因参与转融通证券出借业务而出借的证券|^.*?期末参与转融通证券出借业务的证券(?:注[：:])?无[。.]?$",
        compact(lending),
    ):
        raise ValueError("RATING_MARKET_LENDING_UNVERIFIED")
    pledged = section(text, r"[56]\.4\.12\.3\s*期末债券正回购", r"[56]\.4\.12\.4")
    for market in ("银行间市场债券正回购", "交易所市场债券正回购"):
        clause = compact(pledged).split(market, 1)[-1].split("6.4.12.3.2", 1)[0].split("5.4.12.3.2", 1)[0]
        if not re.search(
            r"(?:未持有(?:因)?.*?作为抵押的债券|无因.*?债券正回购交易而抵押的债券|无(?:因)?从事.*?形成的卖出回购(?:金融资产|证券)款余额|无(?:因)?从事.*?形成的卖出回购(?:金融资产|证券)款，无抵押债券|无(?:银行间|交易所)市场债券正回购，因此没有.*?作为抵押的债券|^无[。.]?$)",
            clause,
        ):
            raise ValueError("RATING_MARKET_PLEDGED_BONDS_UNVERIFIED")
    restricted = section(text, r"[56]\.4\.12\.1\s*因认购", r"[56]\.4\.12\.2")
    blocks = re.split(r"(?m)(?=^[ \t]*\d{5,6}(?:[ \t]+|\r?$))", restricted)[1:]
    total, rows = Decimal(0), []
    by_code = {p["stock_code"].split(".")[0]: p for p in positions}
    for block in blocks:
        prefix = re.match(r"\s*(\d{5,6})", block)[1]
        known = [k for k in by_code if k.startswith(prefix)]
        if len(prefix) == 5:
            immediate = re.match(r"\s*\d{5}[ \t]*\r?\n[ \t]*([0-9])(?:\s|$)", block)
            if immediate and prefix + immediate[1] in known:
                known = [prefix + immediate[1]]
            tails = re.findall(r"(?m)^\s*([0-9])(?:[ \t]+[\u4e00-\u9fff]|[ \t]*$)", block)
            candidates = [prefix + digit for digit in tails if prefix + digit in known]
            if len(known) > 1 and len(set(candidates)) == 1:
                known = list(set(candidates))
            if len(known) > 1:
                # 跨页末尾一位代码不能从年份取值；再用该行已披露的股票名称独立定位。
                names = [
                    k
                    for k in known
                    if re.search(".*?".join(map(re.escape, compact(by_code[k]["stock_name"]))), block, re.S)
                ]
                if len(names) == 1:
                    known = names
        if len(known) != 1:
            raise ValueError("RATING_MARKET_RESTRICTED_IDENTITY:" + prefix)
        # 仅恢复数值小数点后明确缺位的折行，保持原行与恢复后的文本用于核验。
        numeric = re.sub(r"(?<=\d)(?=[\r\n])\s+(?=\.\d{2}\b)", "", block)
        numeric = re.sub(r"(\d+\.\d)\s*\r?\n\s*(\d)(?=\s|$)", r"\1\2", numeric)
        numeric = re.sub(r"(\d+\.)\s*\r?\n\s*(\d{2})(?=\s|$)", r"\1\2", numeric)
        # 部分版式把千分位整数最后一位与小数部分拆到两行；只恢复金额形态的明确折行。
        numeric = re.sub(r"(\d[\d,]*)\s*\r?\n\s*(\d\.\d{2})(?=\s|$)", r"\1\2", numeric)
        # 部分转载把两位单价和千分位股数粘连；数量×单价须同时与成本、估值金额一致。
        glued = re.search(
            r"([\d,]+\.\d{2})\s+([\d,]+\.\d{2})([1-9]\d{0,2}(?:,\d{3})+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*[-－—]",
            numeric,
        )
        if glued:
            issue, price, shares, cost_amount, fair_amount = (decimal(v.replace(",", "")) for v in glued.groups())
            if issue * shares != cost_amount or price * shares != fair_amount:
                raise ValueError("RATING_MARKET_GLUED_PRICE_QUANTITY_CONFLICT")
            numeric = (
                numeric[: glued.start()]
                + glued[0].replace(glued[2] + glued[3], glued[2] + " " + glued[3], 1)
                + numeric[glued.end() :]
            )
        values = re.findall(
            r"(?<![\d,.])([\d,]+(?:\.\d{2})?)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*[-－—](?:\s|$)", numeric
        )
        if len(values) != 1:
            raise ValueError("RATING_MARKET_RESTRICTED_VALUE:" + prefix)
        qty, cost, fair = (decimal(v.replace(",", "")) for v in values[0])
        if qty > decimal(by_code[known[0]]["quantity_shares"]) or fair > decimal(by_code[known[0]]["market_value_cny"]):
            raise ValueError(
                "RATING_MARKET_RESTRICTED_RANGE:"
                + known[0]
                + ":"
                + str(qty)
                + ":"
                + by_code[known[0]]["quantity_shares"]
                + ":"
                + str(fair)
                + ":"
                + by_code[known[0]]["market_value_cny"]
            )
        rows.append({"code": known[0], "quantity": str(qty), "cost": str(cost), "amount": str(fair)})
        total += fair
    if not re.search(
        r"(?:未持有|无)暂时停牌等流通受限股票|期末持有的暂时停牌等流通受限股票(?:注[：:])?无[。.]?$", compact(suspended)
    ):
        halt_rows = re.findall(r"(?m)^\s*(\d{6})\s+(.+?)(?=^\s*\d{6}\s|\Z)", suspended, re.S)
        if not halt_rows:
            raise ValueError("RATING_MARKET_SUSPENDED_TABLE_UNVERIFIED")
        for code, block in halt_rows:
            value = re.findall(r"([\d,]+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*[-－—](?:\s|$)", block)
            if code not in by_code or len(value) != 1:
                raise ValueError("RATING_MARKET_SUSPENDED_ROW_UNVERIFIED")
            qty, cost, fair = (decimal(v.replace(",", "")) for v in value[0])
            rows.append({"code": code, "quantity": str(qty), "cost": str(cost), "amount": str(fair)})
            total += fair
    # 同一发行人可能有多批限售股份；逐批合法仍不代表合计不会超过完整持仓。
    for code, position in by_code.items():
        issuer_rows = [r for r in rows if r["code"] == code]
        if sum(decimal(r["quantity"]) for r in issuer_rows) > decimal(position["quantity_shares"]) or sum(
            decimal(r["amount"]) for r in issuer_rows
        ) > decimal(position["market_value_cny"]):
            raise ValueError("RATING_MARKET_RESTRICTED_ISSUER_SUM")
    if not blocks and not re.search(
        r"(?:未持有|无).*?(?:认购新发|流通受限证券)|流通受限证券(?:注[：:])?无[。.]?$", compact(restricted)
    ):
        raise ValueError("RATING_MARKET_RESTRICTED_TABLE_MISSING")
    # 股票之外受限或质押资产不能静默漏掉，需另行取得全部记录。
    bond_section = compact(restricted.split("受限证券类别：债券")[-1]) if "受限证券类别：债券" in restricted else ""
    if bond_section and re.search(r"\d+\.\d{2}", bond_section):
        raise ValueError("RATING_MARKET_RESTRICTED_BONDS_REVIEW_REQUIRED")
    result = {
        "strategy_continuity": 1,
        "governance_state": governance_state,
        "restricted_stock_amount": str(total),
        "restricted_rows": rows,
        "excerpts": {"governance": governance, "strategy": strategy, "holder": holder},
    }
    if include_holder:
        result["holder_risk_band"] = band
    return result


def a_share_subscription(fee_text: str, subscription_text: str) -> str:
    """A 类份额费用场景：前端申购费首档费率 + 明示不收取销售服务费。

    与 C 类相反的结构不能互相套用。标准情景（一万元）只适用申购费表
    最低金额档（M 小于上限）的费率：档位必须唯一、上限不低于一万元，
    且区域内有前端收费标识；中间档（下限≤M<上限）与固定金额档不算首档。
    销售服务费为零不得由费率表缺行推断，须由概要或招募说明书原文对
    A 类份额明示“不收取销售服务费”。
    """
    region = fee_text
    sales = re.search(r"基金销售相关费用(.*?)基金运作相关费用", fee_text)
    if sales:
        region = sales[1]
    tiers = re.findall(r"(?<![<≤≦=＜〈])M[<〈＜](\d[\d,]*)万(?:元)?([\d.]+)%", region)
    if (
        len(tiers) != 1
        or not re.search(r"申购费[(（]前收费[)）]|前端", region)
        or decimal(tiers[0][0].replace(",", "")) < 1
    ):
        raise ValueError("RATING_MARKET_SUBSCRIPTION_SCENARIO")
    if not re.search(r"A类(?:和[A-Z]类)?(?:基金)?份额[^，。；、]{0,12}不收取销售服务费", subscription_text):
        raise ValueError("RATING_MARKET_SALES_FEE_SCENARIO")
    return str(decimal(tiers[0][1]) / 100)


def key_information(
    pages: list[str],
    code: str,
    name: str,
    prospectus_pages: list[str] | None = None,
    current_names: list[str] | None = None,
    *,
    index: bool = False,
    profile_only: bool = False,
) -> dict:
    text = compact("\n".join(pages))
    if (
        code not in text
        or not matches_name(text, name)
        or not re.search(r"交易币种人民币", text)
        or not profile_only
        and any(
            v in text.split("编制日期")[0].split("送出日期")[0][:200]
            for v in (("QDII", "ETF", "黄金") if index else ("QDII", "ETF", "指数证券", "指数型", "指数增强", "黄金"))
        )
    ):
        raise ValueError("RATING_MARKET_DOCUMENT_SCOPE")
    if not profile_only and index and "指数" not in text.split("编制日期")[0].split("送出日期")[0][:200]:
        raise ValueError("RATING_INDEX_DOCUMENT_SCOPE")
    if not profile_only and not re.search(r"基金类型(?:股票型|混合型)", text):
        raise ValueError("RATING_MARKET_FUND_TYPE_UNVERIFIED")
    share_identity = re.search(
        r"(?:下属基金简称|基金简称C)((?:(?!下属基金|基金代码).){1,80}?C)(?:下属基金(?:交易)?代码|基金代码C)" + code,
        text,
    )
    individual_share = re.search(r"基金简称((?:(?!基金代码).){1,80}?C)基金代码" + code, text[:800])
    c_share = bool(share_identity or individual_share or re.search(r"C(?:类)?(?:基金)?份额", text[:600]))
    # A 类份额身份与 C 类镜像核验，严格程度一致：下属简称以 A 结尾并紧邻本份额代码，
    # 或单份额文件标题标明 A 类。A 类费率结构与 C 类相反（前端收取申购费、不计提销售
    # 服务费），两类场景不能互相套用；身份无法归入任一类同样拒绝。
    share_identity_a = re.search(
        r"(?:下属基金简称|基金简称A)((?:(?!下属基金|基金代码).){1,80}?A)(?:下属基金(?:交易)?代码|基金代码A)" + code,
        text,
    )
    individual_share_a = re.search(r"基金简称((?:(?!基金代码).){1,80}?A)基金代码" + code, text[:800])
    a_share = bool(share_identity_a or individual_share_a or re.search(r"A(?:类)?(?:基金)?份额", text[:600]))
    if not c_share and not a_share:
        raise ValueError("RATING_MARKET_SHARE_CLASS_FEE_IDENTITY")
    master = re.search(r"(?<!属)基金(?:主)?代码(\d{6})", text)
    sent = re.search(r"送出日期[：:]?(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not sent:
        sent = re.search(r"送出日期[：:]?(\d{4})-(\d{2})-(\d{2})", text)
    if not sent:
        # “信息截至”是编制基准日，不能冒充前面已明确公布的文件日期。
        sent = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日[（(]信息截至", text)
    if not master or not sent:
        raise ValueError("RATING_MARKET_KEY_DATE_OR_PRODUCT")
    current = re.findall(r"基金经理([\u4e00-\u9fff]{2,5})(?:开始担任|任职日期)", text)
    if not current and current_names:
        profile = text.split("一、产品概况", 1)[-1].split("二、基金投资", 1)[0]
        if "基金经理" in profile and all(name in profile for name in current_names):
            # 概要表格的姓名列与任职日期列可能换序；只核对已授权任职记录中的确切姓名。
            current = current_names
    if not current:
        raise ValueError("RATING_MARKET_CURRENT_TEAM_MISSING")
    # 费用表可同时列出 A/C/E；按本份额类别标题界定场景，禁止拿其他类别的费率评分。
    start = text.find("三、投资本基金涉及的费用")
    if start < 0:
        start = text.find("三、投资本基金涉及的费用")
    fee_text = text[start:]
    operating = re.findall(
        r"(?:基金运作综合费率（年化）|基金运作综合费率\(年化\))(?:持有期间|持有期)?[-－—]?([\d.]+)%", fee_text
    )
    if not operating:
        raise ValueError("RATING_MARKET_OPERATING_COST_MISSING")
    if len(operating) == 1:
        cost = decimal(operating[0]) / 100
    else:
        # 多份额并列时按本份额标签界定综合费率分段；A 类标签与 C 类标签同样只取本份额行。
        share = share_identity or share_identity_a
        if not share:
            raise ValueError("RATING_MARKET_C_SHARE_LABEL")
        segment = re.search(re.escape(share[1]) + r"基金运作综合费率（年化）(?:持有期间|持有期)?([\d.]+)%", fee_text)
        if not segment:
            raise ValueError("RATING_MARKET_C_COST_AMBIGUOUS")
        cost = decimal(segment[1]) / 100
    subscription_text = fee_text
    if prospectus_pages:
        prospectus = compact("\n".join(prospectus_pages))
        # 招募说明书须出现本份额类别的明文类别标记，防止拿其他份额类别的文件评分。
        class_marker = "C类基金份额" if c_share else "A类基金份额"
        if not matches_name(prospectus, name) or class_marker not in prospectus:
            raise ValueError("RATING_MARKET_PROSPECTUS_IDENTITY")
        subscription_text += prospectus
    if c_share:
        explicit_c = re.search(
            r"C类(?:基金)?份额(?:不收取认购[、/]申购费用|不收取认购费/申购费|不收取认/申购费|不收取申购费|不收取认购费和申购费)",
            subscription_text,
        )
        own_summary = len(operating) == 1 and re.search(r"不收取申购费", fee_text)
        if not explicit_c and not own_summary:
            raise ValueError("RATING_MARKET_SUBSCRIPTION_SCENARIO")
        subscription = "0"
    else:
        subscription = a_share_subscription(fee_text, subscription_text)
    # 365 天赎回费必须由明确适用的区间或文字核实，不能因份额类别推断为零。
    if c_share:
        redemption_verified = re.search(
            r"(?:[NYT][≥≧](?:7|30|180|365)日|(?:7|30|180|365)天≤[NYT]|(?:7|30|180|365)日≤[NYT]|[NYT][≥≧](?:7|30|180|365)天)[-－—]?0(?:\.0+)?%?(?![\d.])",
            fee_text,
        ) or re.search(r"(?:持有期|持有期限|持有时间)(?:在)?(?:7|30)日(?:以上|及以上).*?不收取赎回费", fee_text)
    else:
        # A 类兼容 ASCII 区间写法（如 N>=365天0.00%）；设锁定期且原文明示赎回不收费
        # 的产品按明示处理，同样不接受由类别推断。
        redemption_verified = (
            re.search(
                r"(?:[NYT](?:[≥≧]|>=)(?:7|30|180|365)[日天]|(?:7|30|180|365)[日天](?:[≤≦]|<=)[NYT])[-－—]?0(?:\.0+)?%?(?![\d.])",
                fee_text,
            )
            or re.search(r"(?:持有期|持有期限|持有时间)(?:在)?(?:7|30)日(?:以上|及以上).*?不收取赎回费", fee_text)
            or re.search(r"赎回时?不收取赎回费|(?:持有|锁定|期满)[^。；]{0,20}不收取赎回费", fee_text)
        )
    if not redemption_verified:
        raise ValueError("RATING_MARKET_REDEMPTION_SCENARIO")
    found = date(*map(int, sent.groups())).isoformat()
    fees = {
        "management": "0",
        "custody": "0",
        "sales": "0",
        "subscription": subscription,
        "redemption": "0",
        "subscription_mode": "EXTERNAL",
        "operating_cost": str(cost),
    }
    # 分项费率不使用零占位；本口径只接受已披露的综合运作费与明确交易费。
    fees.pop("management")
    fees.pop("custody")
    fees.pop("sales")
    # cost_1y 与 metrics 的标准情景口径一致：A 类含已核验的前端申购费，C 类为零。
    total_cost = standard_cost(0, 0, 0, subscription, "0", mode="EXTERNAL", operating_cost=str(cost))
    return {
        "fund_code": code,
        "product_id": "PUBLIC:" + master[1],
        "currency": "CNY",
        "published_on": found,
        "current_managers": current,
        "fees": fees,
        "cost_1y": str(total_cost),
    }


def provider_history(snapshot: dict, as_of: date) -> tuple[dict, dict]:
    from app.services.direction_1d_protocol import calendar

    h = snapshot["history"]
    sessions, calendar_hash = calendar()
    days = list(map(date.fromisoformat, h["dates"]))
    if not days or calendar_hash != h["calendar_hash"] or days != [d for d in sessions if days[0] <= d <= as_of]:
        raise ValueError("RATING_MARKET_CALENDAR_GAP")
    if len(h["month_ends"]) != 37 or len({r["nav_date"] for r in h["rows"]}) != len(h["rows"]):
        raise ValueError("RATING_MARKET_MONTH_ENDS_OR_DUPLICATE")
    raw = {r["nav_date"]: r for r in h["rows"]}
    if not days or any(str(d) not in raw or raw[str(d)]["adjusted_nav"] is None for d in days):
        raise ValueError("RATING_MARKET_THREE_YEAR_HISTORY_INCOMPLETE")
    values = [decimal(raw[str(d)]["adjusted_nav"]) for d in days]
    if any(v <= 0 for v in values) or any(not raw[str(d)]["content_hash"] for d in days):
        raise ValueError("RATING_MARKET_NAV_SOURCE_INVALID")
    metrics = history_metrics(
        days, values, [], list(map(date.fromisoformat, h["month_ends"])), 252, require_sharpe=False
    )
    metrics.pop("sharpe")
    ends = [values[h["dates"].index(d)] for d in h["month_ends"]]
    metrics["rolling_positive"] = Decimal(sum(v > 0 for v in metrics["rolling"])) / len(metrics["rolling"])
    metrics["monthly_positive"] = Decimal(sum(b > a for a, b in zip(ends, ends[1:], strict=False))) / (len(ends) - 1)
    metrics["calmar"] = (
        metrics["return_3y"] / metrics["drawdown"]
        if metrics["drawdown"]
        else Decimal(2 if metrics["return_3y"] > 0 else 0 if metrics["return_3y"] == 0 else -1)
    )
    return metrics, {**h, "adjusted_nav": list(map(str, values)), "unit_nav": [raw[str(d)]["unit_nav"] for d in days]}


def provider_team(snapshot: dict, info: dict, start: date, end: date) -> dict:
    records = snapshot["manager_records"]
    dates = sorted(
        {r["begin_date"] for r in records if r["begin_date"]} | {r["end_date"] for r in records if r["end_date"]}
    )
    teams = []
    for day in dates:
        if day > str(end):
            continue
        members = sorted(
            {
                r["manager_name"]
                for r in records
                if r["begin_date"] and r["begin_date"] <= day and (not r["end_date"] or r["end_date"] > day)
            }
        )
        if not members:
            if day >= str(start):
                raise ValueError("RATING_MARKET_TEAM_HISTORY_GAP")
            continue
        if not teams or teams[-1]["members"] != members:
            teams.append({"effective_date": day, "members": members})
    if (
        not teams
        or teams[0]["effective_date"] > str(start)
        or set(teams[-1]["members"]) != set(info["current_managers"])
    ):
        raise ValueError("RATING_MARKET_TEAM_ORIGINAL_CONFLICT")
    return {
        "records": records,
        "teams": teams,
        "team_changes": sum(str(start) < t["effective_date"] <= str(end) for t in teams[1:]),
        "team_months": str(
            min(Decimal(36), Decimal((end - date.fromisoformat(teams[-1]["effective_date"])).days) / Decimal("30.4375"))
        ),
    }


def stock_pages(pages: list[str]) -> list[str]:
    """只适配新版投资组合正文的章节号，证券代码和财务数值保持原文。

    2026 个别报告目录仍列第七节，而正文已改为第六节；本适配不修改旧研究解析器。
    """
    text = "\n".join(pages)
    matches = list(re.finditer(r"(?m)^6\.1\s*(?:报告)?期末基金资产组合情况", text))
    if not matches:
        return pages
    start = matches[-1].start()
    end = re.search(r"(?m)^§\s*7\s|^7\.1\s*期末基金份额持有人", text[start:])
    if not end:
        raise ValueError("RATING_MARKET_INVESTMENT_SECTION_END")
    stop = start + end.start()
    body = re.sub(r"(?m)^6(\.\d+(?:\.\d+)*)\s+(?=[\u4e00-\u9fff])", r"7\1 ", text[start:stop])
    return [text[:start] + body + text[stop:]]


def read_facts(refs: dict, root: Path, code: str, name: str, as_of: date) -> dict:
    def read(label):
        ref = refs[label]
        raw = (root / ref["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("RATING_MARKET_ORIGINAL_HASH")
        return raw

    def pages(label, required_title):
        ref = refs[label]
        if urlsplit(ref["url"]).hostname == ATTACHMENT_HOST:
            identity = attachment_identity(refs, root, label, code, required_title)
            return public_document(read(label), ref, code, required_title)[0], identity
        return public_document(read(label), refs[label], code, required_title)

    report_pages, report_reprint = pages("half", "中期报告")
    summary_pages, summary_reprint = pages("summary", "产品资料概要")
    prospectus_pages = pages("prospectus", "招募说明书")[0] if "prospectus" in refs else None
    snapshot = json.loads(read("provider"))
    if (
        snapshot["fund_code"] != code
        or not snapshot["source"]["enabled"]
        or not snapshot["source"]["authorization_verified_at"]
    ):
        raise ValueError("RATING_MARKET_PROVIDER_SCOPE")
    current_names = sorted(
        {
            r["manager_name"]
            for r in snapshot["manager_records"]
            if r["begin_date"] and r["begin_date"] <= str(as_of) and (not r["end_date"] or r["end_date"] > str(as_of))
        }
    )
    info = key_information(summary_pages, code, name, prospectus_pages, current_names)
    if date.fromisoformat(info["published_on"]) > as_of:
        raise ValueError("RATING_MARKET_SUMMARY_FUTURE")
    if summary_reprint and info["published_on"] != summary_reprint["notice_date"][:10]:
        raise ValueError("RATING_MARKET_REPRINT_PUBLICATION_CONFLICT")
    title = re.sub(r"\s", "", report_pages[0])[:350]
    try:
        holdings = parse_text(
            stock_pages(report_pages),
            title,
            fund_code=code,
            fund_name=name,
            master_code=info["product_id"].split(":")[1],
        )
    except ValueError as error:
        if not str(error).startswith(
            (
                "REPORT_LAYOUT_",
                "REPORT_HOLDINGS_RANK_OR_DUPLICATE",
                "REPORT_INDUSTRY_",
                "REPORT_FULL_HOLDING_TOTAL_MISMATCH",
            )
        ):
            raise
        from app.services.fund_rating_public_stock_tables import parse

        if not matches_name("\n".join(report_pages), name) or code not in "\n".join(report_pages):
            raise ValueError("RATING_MARKET_REPORT_IDENTITY") from error
        report_balance = balance(report_pages)
        holdings = {
            **parse(report_pages, report_balance["stock_amount"], report_balance["net_assets"]),
            "report_end": "2026-06-30",
            "published_date": publication(report_pages),
        }
    if holdings["report_end"] != "2026-06-30" or date.fromisoformat(holdings["published_date"]) > as_of:
        raise ValueError("RATING_MARKET_REPORT_DATE")
    if report_reprint and holdings["published_date"] != report_reprint["notice_date"][:10]:
        raise ValueError("RATING_MARKET_REPRINT_PUBLICATION_CONFLICT")
    b = balance(report_pages)
    if abs(decimal(holdings["stock_value_cny"]) - decimal(b["stock_amount"])) > Decimal(".02"):
        raise ValueError("RATING_MARKET_HOLDINGS_BALANCE_CONFLICT")
    d = disclosures(report_pages, holdings["holdings"])
    _, h = provider_history(snapshot, as_of)
    manager = provider_team(snapshot, info, date.fromisoformat(h["dates"][0]), as_of)
    return {
        "key_information": info,
        "holdings": {
            "report_date": holdings["report_end"],
            "published_on": holdings["published_date"],
            "positions": holdings["holdings"],
            "industries": holdings["reported_industries"],
            "balance": b,
            "net_assets": b["net_assets"],
        },
        "disclosure": d,
        "history": h,
        "manager": manager,
    }


def metrics(bundle: dict, as_of: date, frozen: datetime) -> tuple[dict, dict, datetime]:
    from app.services.fund_rating_public_inputs import public_valid_until

    if (
        bundle["basis"] != BASIS
        or bundle["family"] != "ACTIVE_EQUITY"
        or bundle["currency"] != "CNY"
        or bundle["as_of_date"] != str(as_of)
    ):
        raise ValueError("RATING_MARKET_SCOPE")
    result, _ = provider_history({"history": bundle["history"]}, as_of)
    h, d, manager = (bundle[k] for k in ("holdings", "disclosure", "manager"))
    total = sum(decimal(p["market_value_cny"]) for p in h["positions"])
    if abs(total - decimal(h["balance"]["stock_amount"])) > Decimal(".02") or abs(
        sum(decimal(p["value_cny"]) for p in h["industries"]) - total
    ) > Decimal(".02"):
        raise ValueError("RATING_MARKET_STOCK_OR_INDUSTRY_TOTAL")
    result["issuer_hhi"] = sum((decimal(p["market_value_cny"]) / total) ** 2 for p in h["positions"])
    result["industry_hhi"] = sum((decimal(p["value_cny"]) / total) ** 2 for p in h["industries"])
    net = decimal(h["net_assets"])
    result.update(
        net_assets=net,
        cash_ratio=decimal(h["balance"]["assets"]["货币资金"]) / net,
        restricted_ratio=decimal(d["restricted_stock_amount"]) / net,
    )
    result.update({k: decimal(d[k]) for k in ("holder_risk_band", "strategy_continuity", "governance_state")})
    result.update({k: decimal(manager[k]) for k in ("team_months", "team_changes")})
    fees = bundle["key_information"]["fees"]
    result["cost_1y"] = standard_cost(
        0,
        0,
        0,
        fees["subscription"],
        fees["redemption"],
        mode=fees["subscription_mode"],
        operating_cost=fees["operating_cost"],
    )
    # 运作综合费包含已公布的分项及其他运作费用；此处不再次叠加分项费用。
    report = date.fromisoformat(h["report_date"])
    checked = datetime.fromisoformat(bundle["documents_checked_at"])
    expiry = public_valid_until(as_of, report, checked, frozen)
    return (
        result,
        {
            "nav": str(as_of),
            "holdings": str(report),
            "manager": checked.date().isoformat(),
            "fees": checked.date().isoformat(),
        },
        expiry,
    )


def verify_originals(bundle: dict) -> None:
    refs = {r["label"]: r for r in bundle["raw_archive"]}
    original = read_facts(
        refs,
        Path(get_settings().fund_insight_directory) / "ratings",
        bundle["fund_code"],
        bundle["fund_name_stem"],
        date.fromisoformat(bundle["as_of_date"]),
    )
    if any(original[k] != bundle[k] for k in original):
        raise ValueError("RATING_MARKET_ORIGINAL_FACT_CONFLICT")


def independent_metrics(bundle: dict, expected: dict) -> dict:
    """独立浮点复算原始净值、完整持仓及标准费用，不调用生产指标函数。"""
    import math

    h = bundle["history"]
    rows = {r["nav_date"]: r for r in h["rows"]}
    values = [float(rows[d]["adjusted_nav"]) for d in h["dates"]]
    monthly = [values[h["dates"].index(d)] for d in h["month_ends"]]
    returns = [b / a - 1 for a, b in zip(values, values[1:], strict=False)]
    peak, drawdown = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, 1 - value / peak)
    rolling = [monthly[i] / monthly[i - 12] - 1 for i in range(12, len(monthly))]
    start, end = map(date.fromisoformat, (h["dates"][0], h["dates"][-1]))
    actual = {
        "return_1y": monthly[-1] / monthly[-13] - 1,
        "return_3y": (values[-1] / values[0]) ** (365.25 / (end - start).days) - 1,
        "drawdown": drawdown,
        "downside": math.sqrt(252 * sum(min(r, 0) ** 2 for r in returns) / len(returns)),
        "rolling_positive": sum(r > 0 for r in rolling) / len(rolling),
        "monthly_positive": sum(b > a for a, b in zip(monthly, monthly[1:], strict=False)) / (len(monthly) - 1),
    }
    actual["calmar"] = (
        actual["return_3y"] / drawdown
        if drawdown
        else 2
        if actual["return_3y"] > 0
        else 0
        if actual["return_3y"] == 0
        else -1
    )
    holding, disclosure = bundle["holdings"], bundle["disclosure"]
    total = sum(float(p["market_value_cny"]) for p in holding["positions"])
    actual.update(
        issuer_hhi=sum((float(p["market_value_cny"]) / total) ** 2 for p in holding["positions"]),
        industry_hhi=sum((float(p["value_cny"]) / total) ** 2 for p in holding["industries"]),
        net_assets=float(holding["net_assets"]),
        cash_ratio=float(holding["balance"]["assets"]["货币资金"]) / float(holding["net_assets"]),
        restricted_ratio=sum(float(r["amount"]) for r in disclosure["restricted_rows"]) / float(holding["net_assets"]),
        **{k: float(disclosure[k]) for k in ("holder_risk_band", "strategy_continuity", "governance_state")},
    )
    teams = bundle["manager"]["teams"]
    actual.update(
        team_months=min(36, (end - date.fromisoformat(teams[-1]["effective_date"])).days / 30.4375),
        team_changes=sum(start < date.fromisoformat(t["effective_date"]) <= end for t in teams[1:]),
    )
    fees = bundle["key_information"]["fees"]
    buy = float(fees["subscription"]) / (1 + float(fees["subscription"]))
    actual["cost_1y"] = buy + (1 - buy) * float(fees["redemption"]) + float(fees["operating_cost"])
    if any(not math.isclose(v, float(expected[k]), rel_tol=1e-12, abs_tol=1e-12) for k, v in actual.items()):
        raise ValueError("RATING_MARKET_INDEPENDENT_MISMATCH")
    return {
        "metrics": actual,
        "rolling_one_year": [{"end": d, "return": r} for d, r in zip(h["month_ends"][12:], rolling, strict=True)],
        "note": "仅历史收益窗口，不回填历史综合评级",
    }
