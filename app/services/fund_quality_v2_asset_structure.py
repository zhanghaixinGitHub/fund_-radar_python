"""由完整财报、指数响应及协会原件重建候选全资产结构，不发布评级。

当前适配限于境内直接持股、非股票投资全部为国债的现金证券组合。使用
资产总额作非负权重分母，负债另列；这衡量资产结构对合同业绩基准的偏离，
不是合同合规结论、净敞口、现金目标区间或发行人身份核验。
"""

import json
import re
from datetime import date
from decimal import localcontext
from urllib.parse import urlsplit

from app.services.fund_quality_v2_balance_proofs import identify_nonnegative_parts, strict_balance_values
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_index_reference import rebuild_index_reference
from app.services.fund_quality_v2_industry_reference import equity_industry_diagnostic, rebuild_capco_reference
from app.services.fund_rating_public_market import compact, section
from app.services.fund_rating_public_stock_tables import parse
from app.services.fund_rating_rules import decimal

STRUCTURE_POLICY = {
    "version": "QUALITY_GROSS_ASSET_INDUSTRY_20261017_1",
    "denominator": "资产负债表资产总计；负债单列，不抵扣货币资金或制造负行业权重",
    "equity": "完整股票金额按同日CAPCO门类A-S；基金完整行业表逐项金额与全股票金额对账",
    "non_equity": "国债、货币资金、结算备付金、存出保证金及各项应收分别成类；其他资产未知即拒收",
    "target": "合同业绩基准明示的指数权重乘同期完整指数行业权重，活期存款权重记货币资金",
    "rounding": "公开指数舍入残差保留，不归一；半L1原值不截断",
    "arithmetic": "十进制40位有效数字；类别代码排序后逐项求和，不依赖哈希种子或调用方精度",
    "zero": "完整账簿与完整目标的并集外权重按已明确不存在记0；缺行、缺来源不可记0",
    "meaning": "行业及非权益资产类别结构参照偏离；不是合同强制仓位、发行人复制或合规判断",
    "scope": "境内直接股票及国债；基金、信用债、衍生品、回购和其他未穿透投资暂不适配",
}
REFERENCE_ROLES = {"index_metadata", "index_weights", "industry_article", "industry_pdf"}


def capco_publication(article: bytes, *, attachment_url: str) -> dict:
    """仅把协会JS中的JSON作为数据读取，核正文标题、日期及附件关联，不执行脚本。"""
    text = article.decode("utf-8-sig").strip()
    prefix = "export const allData ="
    if not text.startswith(prefix):
        raise ValueError("QUALITY_STRUCTURE_INDUSTRY_ARTICLE_SCHEMA")
    data = json.loads(text[len(prefix) :].strip().removesuffix(";").rstrip())
    title = re.fullmatch(r"(20\d{2})年(上|下)半年上市公司行业分类结果", data.get("title", ""))
    if not title or data.get("source") != "中国上市公司协会":
        raise ValueError("QUALITY_STRUCTURE_INDUSTRY_ARTICLE_IDENTITY")
    # 关联的是实际href及附件名称，不能只相信登记中的observation_period或布尔值。
    links = re.findall(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', data.get("text", ""), re.S)
    expected_name = data["title"] + "（按股票代码排序）.pdf"
    if not any(url.strip() == attachment_url and compact(name) == expected_name for url, name in links):
        raise ValueError("QUALITY_STRUCTURE_INDUSTRY_ATTACHMENT_NOT_LINKED")
    period = f"{title[1]}-06-30" if title[2] == "上" else f"{title[1]}-12-31"
    published = date.fromisoformat(data["date"])
    if str(published) < period:
        raise ValueError("QUALITY_STRUCTURE_INDUSTRY_ARTICLE_DATE")
    return {"period_end": period, "published_on": str(published), "title": data["title"]}


def gross_asset_book(pages: list[str]) -> dict:
    """固定十进制精度后重建账簿，避免调用方上下文改变同一原件的权重。"""
    with localcontext() as context:
        context.prec = 40
        return _gross_asset_book(pages)


def _gross_asset_book(pages: list[str]) -> dict:
    """逐项对账全资产，再拆出股票行业和非权益类别；未知投资不放入残差桶。

    国债品种表的所有类别必须齐全，只有国家债券允许非零；前五债券仅当逐行
    合计恰好覆盖全部债券余额时才能认作完整债券表。表内“－”保留未知，
    仅由完整、互斥非负科目精确守恒唯一识别的余额另留证明；缺行不作零。
    所有金额单位为人民币元。
    """
    balance_proof = strict_balance_values(pages, observed_on=str(report_period("\n".join(pages))[1]))
    b = balance_proof["point_book"]
    if b is None:
        raise ValueError("QUALITY_STRUCTURE_EXACT_BALANCE_POINT_NOT_IDENTIFIED")
    text = "\n".join(pages)
    stock = parse(pages, b["stock_amount"], b["net_assets"], require_industry=True)
    assets = {key: decimal(value) for key, value in b["assets"].items()}
    if any(v < 0 for v in assets.values()) or decimal(b["fund_amount"]) != 0:
        raise ValueError("QUALITY_STRUCTURE_ASSET_VALUE_OR_FUND_UNSUPPORTED")
    if assets["交易性金融资产"] != decimal(b["stock_amount"]) + decimal(b["bond_amount"]):
        raise ValueError("QUALITY_STRUCTURE_TRADING_ASSETS_NOT_FULLY_DECOMPOSED")
    unsupported = {
        "衍生金融资产",
        "买入返售金融资产",
        "递延所得税资产",
        "其他资产",
        "债权投资",
        "其他债权投资",
        "其他权益工具投资",
    }
    if any(assets.get(key, 0) != 0 for key in unsupported):
        raise ValueError("QUALITY_STRUCTURE_OTHER_ASSETS_NOT_LOOKED_THROUGH")
    if "本报告期末本基金未持有衍生金融资产/负债" not in compact(text):
        raise ValueError("QUALITY_STRUCTURE_DERIVATIVE_LIABILITIES_UNVERIFIED")
    bond_section = section(text, r"(?m)^7\.5\s+期末按债券品种", r"(?m)^7\.6\s")
    rows = {}
    money = r"(?:\d[\d,]*\.\d{2}|[-－—])"
    for m in re.finditer(rf"(?m)^\s*(\d+)\s+([^\n\d]+?)\s+({money})\s+({money})\s*$", bond_section):
        rank, name, amount, _ = m.groups()
        if int(rank) in rows:
            raise ValueError("QUALITY_STRUCTURE_DUPLICATE_BOND_CATEGORY")
        rows[int(rank)] = (compact(name), None if amount in "-－—" else decimal(amount.replace(",", "")))
    labels = [
        "国家债券",
        "央行票据",
        "金融债券",
        "企业债券",
        "企业短期融资券",
        "中期票据",
        "可转债（可交换债）",
        "同业存单",
        "其他",
        "合计",
    ]
    if set(rows) != set(range(1, 11)) or [rows[i][0] for i in range(1, 11)] != labels:
        raise ValueError("QUALITY_STRUCTURE_BOND_CATEGORIES_INCOMPLETE")
    bond_total = decimal(b["bond_amount"])
    bond_proof = identify_nonnegative_parts(str(bond_total), {
        str(i): str(rows[i][1]) if rows[i][1] is not None else None for i in range(1, 10)
    })
    if bond_proof["state"] != "POINT_IDENTIFIED":
        raise ValueError("QUALITY_STRUCTURE_BOND_CATEGORIES_POINT_UNIDENTIFIED")
    identified_bonds = {int(key): decimal(value) for key, value in bond_proof["point_values"].items()}
    if (identified_bonds[1] != bond_total or rows[10][1] is not None and rows[10][1] != bond_total
            or any(identified_bonds[i] != 0 for i in range(2, 10))):
        raise ValueError("QUALITY_STRUCTURE_NON_GOVERNMENT_BOND_UNSUPPORTED")
    details = section(text, r"(?m)^7\.6\s+期末按公允价值", r"(?m)^7\.7\s")
    bonds = []
    for m in re.finditer(r"(?m)^\s*(\d+)\s+(\d{6})\s+(.*?)\s+([\d,]+)\s+([\d,]+\.\d{2})\s+([\d.]+)\s*$", details):
        rank, code, name, qty, amount, weight = m.groups()
        bonds.append(
            {
                "rank": int(rank),
                "security_code": code,
                "name": compact(name),
                "quantity_bonds": qty.replace(",", ""),
                "value_cny": amount.replace(",", ""),
                "nav_percent": weight,
                "excerpt": m[0].strip(),
            }
        )
    if (
        len({r["security_code"] for r in bonds}) != len(bonds)
        or sum(decimal(r["value_cny"]) for r in bonds) != bond_total
    ):
        raise ValueError("QUALITY_STRUCTURE_BOND_DETAIL_DOES_NOT_COVER_FULL_AMOUNT")
    industries = {}
    for row in stock["reported_industries"]:
        key = "EQUITY:" + row["industry_code"]
        industries[key] = industries.get(key, 0) + decimal(row["value_cny"])
    amounts = {**industries, "GOVERNMENT_BONDS": bond_total}
    for key, value in assets.items():
        if key != "交易性金融资产":
            amounts["ASSET:" + key] = value
    total = decimal(b["total_assets"])
    if total <= 0 or sum(amounts.values()) != total:
        raise ValueError("QUALITY_STRUCTURE_GROSS_ASSET_LEDGER_NOT_RECONCILED")
    return {
        "balance": b,
        "balance_proof": balance_proof,
        "bond_category_proof": bond_proof | {
            "printed_total_cell": str(rows[10][1]) if rows[10][1] is not None else None,
            "parent_bond_total_cny": str(bond_total), "original_excerpt": bond_section,
        },
        "positions": stock["holdings"],
        "disclosed_industries": stock["reported_industries"],
        "bonds": bonds,
        "bond_category_excerpt": bond_section,
        "amounts_cny": {k: str(v) for k, v in sorted(amounts.items())},
        "weights": {k: str(v / total) for k, v in sorted(amounts.items())},
        "denominator_cny": str(total),
        "liabilities_cny": b["liabilities"],
        "policy": STRUCTURE_POLICY,
    }


def half_l1_structure(actual: dict, target: dict) -> tuple[str, dict]:
    """固定精度及类别排序，返回半L1总值和逐类贡献；未知类别须在上游拒收。"""
    with localcontext() as context:
        context.prec = 40
        keys = sorted(actual.keys() | target.keys())
        contributions = {key: abs(actual.get(key, 0) - target.get(key, 0)) / 2 for key in keys}
        return str(sum(contributions.values())), {key: str(value) for key, value in contributions.items()}


def rebuild_industry_structure(
    pages: list[str], *, benchmark: str, reference: dict, raw: dict, observed_on: str, as_of: str
) -> dict:
    """整个重建窗口固定40位精度；正文和原字节仍由内层逐件验证。"""
    with localcontext() as context:
        context.prec = 40
        return _rebuild_industry_structure(
            pages, benchmark=benchmark, reference=reference, raw=raw, observed_on=observed_on, as_of=as_of
        )


def _rebuild_industry_structure(
    pages: list[str], *, benchmark: str, reference: dict, raw: dict, observed_on: str, as_of: str
) -> dict:
    """参考附件已逐件核字节后重建行业指标；输入元数据不能直接成为来源证明。

    raw是按四个固定角色读取的实际字节；reference为对应来源回执。指数身份
    必须与已核验概要中的完整基准公式一致，行业发布正文必须关联该附件。
    """
    if set(raw) != REFERENCE_ROLES or set(reference) != REFERENCE_ROLES:
        raise ValueError("QUALITY_STRUCTURE_REFERENCE_ROLES")
    for role, ref in reference.items():
        url = urlsplit(ref["url"])
        host = {
            "index_metadata": "api.tushare.pro",
            "index_weights": "api.tushare.pro",
            "industry_article": "www.capco.org.cn",
            "industry_pdf": "sp.capco.org.cn",
        }[role]
        if url.scheme != "https" or url.hostname != host or url.username or url.password or url.query or url.fragment:
            raise ValueError("QUALITY_STRUCTURE_SOURCE_HOST")
    metadata = json.loads(raw["index_metadata"])
    weights = json.loads(raw["index_weights"])
    index_rows = metadata.get("data", {}).get("items", [])
    fields = metadata.get("data", {}).get("fields", [])
    if len(index_rows) != 1 or "ts_code" not in fields or len(index_rows[0]) != len(fields):
        raise ValueError("QUALITY_STRUCTURE_INDEX_IDENTITY")
    index_code = index_rows[0][fields.index("ts_code")]
    target = rebuild_index_reference(metadata, weights, index_code=index_code, observed_on=observed_on)
    # 窄范围适配明确写出的两项基准；不从名称猜目标，不把95%说成合同持仓下限。
    expression = re.fullmatch(
        r"(\d+(?:\.\d+)?)%×"
        + re.escape(target["index_name"])
        + r"指数收益率＋(\d+(?:\.\d+)?)%×银行人民币活期存款利率（税后）",
        benchmark or "",
    )
    if not expression:
        raise ValueError("QUALITY_STRUCTURE_BENCHMARK_INDEX_NOT_MATCHED")
    equity, cash = (decimal(v) / 100 for v in expression.groups())
    if min(equity, cash) < 0 or equity + cash != 1:
        raise ValueError("QUALITY_STRUCTURE_BENCHMARK_WEIGHTS")
    publication = capco_publication(raw["industry_article"], attachment_url=reference["industry_pdf"]["url"])
    if publication["period_end"] != observed_on or publication["published_on"] > as_of or observed_on > as_of:
        raise ValueError("QUALITY_STRUCTURE_CLASSIFICATION_TIME")
    classification = rebuild_capco_reference(
        raw["industry_pdf"],
        expected_sha256=reference["industry_pdf"]["sha256"],
        period_end=observed_on,
        published_on=publication["published_on"],
    )
    book = gross_asset_book(pages)
    diagnostic = equity_industry_diagnostic(
        book["positions"],
        target,
        classification,
        expected_stock_amount=book["balance"]["stock_amount"],
        disclosed_industries=book["disclosed_industries"],
    )
    actual = {k: decimal(v) for k, v in book["weights"].items()}
    target_weights = {"EQUITY:" + k: equity * decimal(v) for k, v in diagnostic["target_industries"].items()}
    target_weights["ASSET:货币资金"] = cash
    value, contributions = half_l1_structure(actual, target_weights)
    return {
        "value": value,
        "observed_on": observed_on,
        "scope": "FULL_GROSS_ASSET_INDUSTRY_AND_NON_EQUITY_CATEGORIES",
        "asset_book": book,
        "target_index": target,
        "classification_publication": publication,
        "target_weights": {k: str(v) for k, v in sorted(target_weights.items())},
        "target_weight_sum": str(sum(target_weights.values())),
        "rounding_residual": str(1 - sum(target_weights.values())),
        "half_l1_contributions": contributions,
        "unmapped_actual_securities": diagnostic["unmapped_actual_securities"],
        "unmapped_rule": "实际基金行业表完整对账；不把剩余金额倒推为未映射证券分类",
        "acceptance": "CANDIDATE_RAW_METRIC_NOT_CALIBRATED_OR_INDEPENDENTLY_ACCEPTED",
    }
