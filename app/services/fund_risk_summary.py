"""从已核验披露快照生成公共风险事实；不推定实时仓位或买卖动作。"""

import re
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine
from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import read, save
from app.services.fund_materials import snapshot
from app.services.fund_materials_build import safe_url


def number(value) -> Decimal | None:
    """接受有限十进制数，空值与布尔值不转换成零。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        value = Decimal(str(value))
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def valid_day(value, cutoff: date) -> str | None:
    try:
        day = date.fromisoformat(str(value))
        return str(day) if day <= cutoff else None
    except ValueError:
        return None


def unavailable(code: str, reason: str) -> dict:
    return {
        "fundCode": code,
        "available": False,
        "asOfDate": None,
        "reportDate": None,
        "publishedDate": None,
        "sourceUrl": None,
        "facts": [],
        "holdings": [],
        "industries": [],
        "assets": [],
        "crossCheck": None,
        "limitations": [reason],
        "disclosedWeightPct": None,
        "topTenWeightPct": None,
        "quoteCoverageWeightPct": None,
    }


def calculate(code: str, data: dict | None, *, today: date) -> dict:
    """所有占比保留披露分母；历史权重乘当前涨跌只称静态估算。

    总资产配置与净资产持仓分别返回；不同日期的行情不混合求和。缺少行业行情
    时明确保留缺口，不能把公司行业名称当行业涨跌，也不能用个股估算解释预测。
    """
    if not data:
        return unavailable(code, "暂缺可核验的披露持仓，暂时无法说明底层风险。")
    if data.get("fundCode") != code:
        raise ValueError("FUND_RISK_SCOPE_MISMATCH")
    as_of = valid_day(data.get("asOfDate"), today)
    if not as_of:
        return unavailable(code, "资料日期尚未核实，暂时无法说明底层风险。")
    cutoff = date.fromisoformat(as_of)
    reports = [
        r
        for r in data.get("reports", [])
        if valid_day(r.get("endDate"), cutoff)
        and valid_day(r.get("publishedDate"), cutoff)
        and r["endDate"] <= r["publishedDate"]
    ]
    if not reports:
        return unavailable(code, "持仓报告日期或公开日期尚未核实。")
    report = max(reports, key=lambda r: (r["endDate"], r["publishedDate"], r.get("fullDisclosure", False)))
    holdings = report.get("holdings", [])
    if not holdings or len(holdings) > 1000:
        return unavailable(code, "持仓明细暂缺或范围尚未核实。")
    codes = [h.get("stockCode") for h in holdings]
    weights = [number(h.get("weightPct")) for h in holdings]
    disclosed = number(report.get("disclosedWeightPct"))
    if (
        any(not re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", str(c)) for c in codes)
        or len(set(codes)) != len(codes)
        or disclosed is None
        or disclosed < 0
        or disclosed > 100
        or any(w is None or w < 0 or w > 100 for w in weights)
    ):
        return unavailable(code, "披露持仓的对象或占比有待核对，暂不合并计算。")
    total = sum(weights, Decimal(0))
    # 原披露百分比保留两位；累计舍入误差按明细数计算，不能容许任意缺失权重。
    tolerance = Decimal("0.005") * (len(weights) + 1)
    if abs(total - disclosed) > tolerance or total > 100 + tolerance:
        return unavailable(code, "持仓明细合计与披露占比不一致，暂不合并计算。")
    top_ten = sum(sorted(weights, reverse=True)[:10], Decimal(0))
    result = unavailable(code, "")
    result.update(
        available=True,
        asOfDate=as_of,
        reportDate=report["endDate"],
        publishedDate=report["publishedDate"],
        sourceUrl=safe_url(report.get("sourceUrl")),
        disclosedWeightPct=float(disclosed),
        topTenWeightPct=float(top_ten),
        holdings=[
            {"stockCode": h["stockCode"], "name": h["stockName"], "weightPct": float(w)}
            for h, w in zip(holdings, weights, strict=True)
        ],
    )
    result["facts"] = [
        f"前十大股票合计占基金净资产 {top_ten:.2f}%。",
        f"本次股票明细覆盖基金净资产 {disclosed:.2f}%，未将已披露部分重新放大到全部资产。",
    ]
    result["limitations"] = [
        "持仓来自定期披露，可能已经变化，不代表今日实时仓位。",
        "尚无同一时点的独立行业涨跌资料，暂不判断行业与重仓股是否共同走弱。",
        "这些是披露和行情事实，不代表已计入走势预测，也不直接形成买卖建议。",
    ]
    for key, denominator in (("industries", "基金净资产"), ("assets", "基金总资产")):
        for row in report.get(key, []):
            weight = number(row.get("weightPct"))
            if weight is not None and 0 <= weight <= 100:
                result[key].append({"name": row["name"], "weightPct": float(weight), "denominator": denominator})
    if result["industries"]:
        major = max(result["industries"], key=lambda r: r["weightPct"])
        result["facts"].append(f"披露行业中，{major['name']}占基金净资产 {major['weightPct']:.2f}%。")
    company_rows = data.get("companies", [])
    duplicates = Counter(c.get("stockCode") for c in company_rows)
    # 同一公司出现冲突记录时不随意选择最后一条；该公司行情保持未知。
    companies = {c["stockCode"]: c for c in company_rows if duplicates[c.get("stockCode")] == 1}
    quotes = []
    for holding, weight in zip(holdings, weights, strict=True):
        quote = companies.get(holding["stockCode"], {}).get("quote", {})
        day = valid_day(quote.get("date"), date.fromisoformat(as_of))
        change = number(quote.get("changePct"))
        if day and change is not None and change >= -100 and weight > 0:
            quotes.append((day, weight, change))
    dates = Counter(q[0] for q in quotes)
    if len(dates) == 1:
        coverage = sum((q[1] for q in quotes), Decimal(0))
        contribution = sum((q[1] * q[2] / 100 for q in quotes), Decimal(0))
        result["quoteCoverageWeightPct"] = float(coverage)
        result["crossCheck"] = {
            "date": quotes[0][0],
            "coverageWeightPct": float(coverage),
            "staticContributionPctPoints": float(contribution),
            "summary": f"按披露权重静态估算，已覆盖股票当日变动约影响 {contribution:+.2f} 个百分点。",
            "limitation": ("只计算有同日行情的已披露股票，未覆盖部分保持未知；不是实际净值涨跌归因。"),
        }
    else:
        result["limitations"].append("重仓股行情日期不一致或暂缺，暂不计算合计影响。")
    return result


def publish(code: str, *, root: Path | None = None) -> dict:
    """显式任务构建并留存不可变结果，页面读取不会触发构建或来源请求。"""
    if not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("INVALID_FUND_CODE")
    # 同一基金的发布顺序由数据库事务锁串行化；排队任务取得锁后才读取最新资料。
    with get_engine().begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(20260929, :code)"), {"code": int(code)})
        return _publish(code, root=root)


def _publish(code: str, *, root: Path | None) -> dict:
    data = snapshot(code)  # 原资料读取会校验内容哈希；不修改历史研究或资格。
    value = calculate(code, data, today=datetime.now(ZONE).date())
    # 保存实际参与计算的原始输入，以后资料更新也能复算；公告全文不参与本项计算。
    inputs = None if data is None else {key: data.get(key) for key in ("fundCode", "asOfDate", "reports", "companies")}
    evidence = {"rule": "DISCLOSED_FUND_RISK_V1", "sourceHash": digest(data), "inputs": inputs, "result": value}
    version = digest(evidence)
    folder = (root or Path(get_settings().fund_insight_directory)) / code
    artifact = folder / "risk" / f"{version}.json"
    if artifact.exists():
        if read(artifact) != evidence:
            raise ValueError("FUND_RISK_ARCHIVE_MISMATCH")
    else:
        try:
            save(artifact, evidence)
        except FileExistsError:
            if read(artifact) != evidence:
                raise ValueError("FUND_RISK_ARCHIVE_MISMATCH") from None
    save(folder / "risk-current.json", {"version": version}, replace=True)
    return {"fundCode": code, "version": version, "available": value["available"]}


def current(code: str) -> dict:
    """只读已发布结果；保留其原始资料日期，不把读取时间作为资料日期。"""
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("INVALID_FUND_CODE")
    folder = Path(get_settings().fund_insight_directory) / code
    pointer = folder / "risk-current.json"
    if not pointer.exists():
        return unavailable(code, "风险资料尚未完成核对，暂不提供底层风险结论。")
    version = read(pointer)["version"]
    if not re.fullmatch(r"[a-f0-9]{64}", version):
        raise ValueError("FUND_RISK_VERSION_INVALID")
    evidence = read(folder / "risk" / f"{version}.json")
    if digest(evidence) != version or evidence["result"]["fundCode"] != code:
        raise ValueError("FUND_RISK_SCOPE_MISMATCH")
    return evidence["result"]
