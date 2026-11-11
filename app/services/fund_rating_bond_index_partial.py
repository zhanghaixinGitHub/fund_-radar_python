"""债券指数已验证分项候选；纯重放、不联网、不写库、不计算总分。

完整类型未验收。缺失基准不能拖空净值风险等独立分项，也不能凭本模块
变成完整评分；抽样持仓只披露事实，不套股票集中度或未验收类别公式。
"""

import re
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services import fund_rating_bond_disclosure as base
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_market_review import annual_fee_score, fee_profile, normalized_text

VERSION = "BOND_INDEX_PARTIAL_PUBLIC_EVIDENCE_20261111_V1"
KEYS = ("return", "risk", "efficiency", "stability", "management", "holdings", "liquidity", "cost")
PARTIAL_ENABLED = True
COMPLETE_ENABLED = False


def calculate(inputs, *, root: Path):
    base.require(inputs.get("version") == VERSION and inputs.get("profile") == "BOND_INDEX", "PARTIAL_VERSION_PROFILE")
    scope = {
        "fund_code": "001021",
        "legal_name": "亚债中国债券指数基金",
        "manager_name": "华夏基金管理有限公司",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "primary_ref": "58fd1c2e700ed5ba99b68e12314bc2542494701f66996d27a4055879d7b65447",
    }
    base.require(all(inputs.get(k) == v for k, v in scope.items()), "PARTIAL_VERIFIED_SCOPE")
    base.require(inputs.get("currency") == "CNY" and inputs.get("share_class") == "A", "PARTIAL_CURRENCY_SHARE")
    source = Originals(inputs, Path(root), lazy=True)
    base.require(inputs["refs"]["half"] == inputs["primary_ref"], "PARTIAL_REPORT_BINDING")
    pages = source.pages(inputs["primary_ref"])
    text = "\n".join(pages)
    report = normalized_text(text)
    base.require(
        inputs["fund_code"] in report
        and normalized_text(inputs["legal_name"]) in report
        and inputs["manager_name"] in report,
        "PARTIAL_LEGAL_IDENTITY",
    )
    base.require(str(report_period(text)[1]) == inputs["holding_date"], "PARTIAL_REPORT_DATE")
    scores = {k: None for k in KEYS}
    facts = {
        "partial_only": True,
        "complete_type_validated": False,
        "evidence_errors": {},
        "unscored_dimensions": {
            "return": "同截止一年准确基准日线缺失；报告取整值不作精确点",
            "efficiency": "缺准确基准日线，跟踪误差未知",
            "stability": "缺准确基准月份值，胜率未知",
            "holdings": "抽样债券类别事实尚未独立验收为点分公式",
        },
    }
    with localcontext() as ctx:
        ctx.prec = 65
        try:
            days, ends, values = base._nav(inputs, source)
            base.require(ends[0] == "2025-09-30" and ends[-1] == "2026-09-30", "PARTIAL_VERIFIED_YEAR")
            peak = values[0]
            drawdown = D(0)
            for value in values:
                peak = max(peak, value)
                drawdown = max(drawdown, 1 - value / peak)
            scores["risk"] = 100 * (1 - drawdown)
            facts["window"] = {"from": ends[0], "through": ends[-1], "month_ends": ends, "valuation_points": len(days)}
            facts["risk"] = {
                "maximum_drawdown": str(drawdown),
                "first_nav": str(values[0]),
                "last_nav": str(values[-1]),
                "fund_total_return": str(values[-1] / values[0] - 1),
                "scope": "本A同一年真实复权净值的已实现最大回撤；不代表未来风险、久期或信用风险已全面通过。",
                "provider_source_hash": inputs["refs"]["provider"],
            }
        except (ValueError, KeyError) as exc:
            facts["evidence_errors"]["risk"] = str(exc)
        try:
            provider = source.json(inputs["refs"]["provider"])
            scores["management"] = base._management(inputs, source, provider, report)
            facts["management"] = {
                "current_names": inputs["manager_review"]["current_names"],
                "rule": "已核实际职责50×40%＋全窗连续75×20%＋有原件且明确整改50×40%；不引用历史点分。",
                "claims": inputs["manager_review"]["claims"],
            }
        except (ValueError, KeyError) as exc:
            facts["evidence_errors"]["management"] = str(exc)
        try:
            review = inputs["holder_review"]
            for name in ["mechanism", "execution", "actual_operation"]:
                source.claim(review[name])
            source.claim(review["holder_concentration"])
            concentration = normalized_text(review["holder_concentration"]["quote"])
            base.require(
                review["holder_concentration"]["source_hash"] == inputs["primary_ref"]
                and "5,009,431,504.3552.60%" in concentration,
                "PARTIAL_VERIFIED_HOLDER_ROW",
            )
            mechanism = normalized_text(review["mechanism"]["quote"])
            trustee = normalized_text(review["execution"]["quote"])
            operation = normalized_text(review["actual_operation"]["quote"])
            base.require("巨额" in mechanism and "赎回" in mechanism, "PARTIAL_REDEMPTION_MECHANISM")
            base.require(
                all(v in trustee for v in ["托管人", "申购赎回价格", "真实", "准确", "完整"])
                and ("核查" in trustee or "复核" in trustee)
                and "未发现损害基金" in trustee,
                "PARTIAL_TRUSTEE_ACTUAL_REVIEW",
            )
            base.require(
                "遵守" in operation and "基金合同" in operation and "没有损害基金" in operation,
                "PARTIAL_ACTUAL_OPERATION",
            )
            flows, _ = base._flows(pages, allow_reversed=True, allow_interleaved=True, allow_period_variant=True)
            own = next(r for r in flows if r["share"] == inputs["share_class"])
            base.require(review["actual_effect_review_complete"] is True, "PARTIAL_HOLDER_REVIEW_OPEN")
            scores["liquidity"] = None if review["confirmed_pressure_effects"] else D(75)
            facts["liquidity"] = {
                "reported_redemptions": own["redeemed"],
                "largest_holder_percent": "52.60",
                "holder_concentration_original": review["holder_concentration"],
                "meaning": (
                    "有公开机制及托管人实际核查、期内实际申赎记录；"
                    "固定原件行绑定的集中比例及条件性压力提示继续披露，"
                    "75为公开执行观察档，不代表没有赎回压力。"
                ),
            }
        except (ValueError, KeyError, StopIteration) as exc:
            facts["evidence_errors"]["liquidity"] = str(exc)
        try:
            fees = fee_profile(
                "\n".join(source.pages(inputs["refs"]["summary"])),
                code=inputs["fund_code"],
                share_class=inputs["share_class"],
                legal_name=inputs["legal_name"],
                supporting_text="\n".join(source.pages(inputs["refs"]["prospectus"])),
                published_on=inputs["sources"][inputs["refs"]["summary"]]["published_on"],
            )
            cost = annual_fee_score(fees["fees"])
            scores["cost"] = D(cost["score"])
            facts["cost"] = cost
        except (ValueError, KeyError) as exc:
            facts["evidence_errors"]["cost"] = str(exc)
    # 原表类别守恒仅为事实，暂不变成未验收的抽样持仓点分。
    sections = list(re.finditer(r"7\.5\s*期末按债券品种分类的债券投资组合(.*?)7\.6", text, re.S))
    section = sections[-1] if sections else None
    if section:
        body = section[1]
        gov = re.search(r"1\s+国家债券\s+([\d,]+\.\d{2})", body)
        policy = re.search(r"其中[：:]政策性金融债\s+([\d,]+\.\d{2})", body)
        total = re.search(r"10\s+合计\s+([\d,]+\.\d{2})", body)
        if gov and policy and total:
            values = [D(m[1].replace(",", "")) for m in [gov, policy, total]]
            facts["public_bond_categories"] = {
                "source_hash": inputs["primary_ref"],
                "quote": normalized_text(body),
                "government_cny": str(values[0]),
                "policy_bank_cny": str(values[1]),
                "all_bonds_cny": str(values[2]),
                "sums_reconcile": values[0] + values[1] == values[2],
                "score": None,
                "scope": "发行人类别匹配事实；不代表逐券资格、精确复制权重或评分类型验收。",
            }
    events = inputs.get("reviewed_events", [])
    blocked = (
        any(e.get("status") == "UNRESOLVED" and e.get("severity") in ["MATERIAL", "SEVERE"] for e in events)
        or bool(inputs.get("holder_review", {}).get("confirmed_pressure_effects"))
        or inputs.get("manager_review", {}).get("governance_state") == "MATERIAL_UNRESOLVED"
    )
    if inputs.get("event_review"):
        try:
            event = inputs["event_review"]
            review = source.json(event["source_hash"])
            base.require(
                event.get("complete") is True
                and review["fund_code"] == inputs["fund_code"]
                and review["as_of_date"] == inputs["as_of_date"]
                and review["complete"] is True
                and review["pending_events"] == []
                and review["conclusion"] == "NO_UNRESOLVED_MATERIAL_EVENT_FOR_AS_OF",
                "PARTIAL_EVENT_REVIEW",
            )
            base.require(
                isinstance(review["reviews"], list)
                and len(review["reviews"]) == len(review["candidate_source_hashes"])
                and set(review["candidate_source_hashes"]) == {r["event"]["source_hash"] for r in review["reviews"]}
                and all(
                    r.get("decision") == "NO_RATING_IMPACT"
                    and r.get("reviewed_at")
                    and r.get("note")
                    and r.get("event", {}).get("fund_code") == inputs["fund_code"]
                    for r in review["reviews"]
                ),
                "PARTIAL_EVENT_RECEIPTS",
            )
            for ref in review["receipt_sources"]:
                receipt = source.json(ref["source_hash"])
                base.require(
                    receipt
                    == next(r for r in review["reviews"] if r["event"]["source_hash"] == ref["candidate_source_hash"]),
                    "PARTIAL_EVENT_RECEIPT_ORIGINAL",
                )
            base.require(len(review["receipt_sources"]) == len(review["reviews"]), "PARTIAL_EVENT_RECEIPT_COUNT")
            facts["event_review"] = {
                "source_hash": event["source_hash"],
                "checked_at": review["checked_at"],
                "conclusion": review["conclusion"],
                "receipt_count": len(review["reviews"]),
                "coverage": review["news_coverage"],
                "scope": "事后已采集事项回执保护，不把10-11月消息倒算入9月评分窗口；正式发布仍实时核门控。",
            }
        except (ValueError, KeyError) as exc:
            blocked = True
            facts["evidence_errors"]["event_review"] = str(exc)
    else:
        blocked = True
        facts["evidence_errors"]["event_review"] = "PARTIAL_EVENT_REVIEW_MISSING"
    facts["source_sha256"] = sorted(source.raw)
    return {
        "dimensions": {k: str(v) if v is not None else None for k, v in scores.items()},
        "facts": facts,
        "blocked": bool(blocked),
    }
