"""资料任务中编译评价事实；沿用同版本受控读取，不接受人工填写评价数字。"""

import re
from copy import deepcopy
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path

from app.services.fund_rating_narrative import number


def _drawdown(values, days):
    peak, peak_day, maximum, pair = values[0], days[0], Decimal(0), (days[0], days[0])
    for day, value in zip(days, values, strict=True):
        if value > peak:
            peak, peak_day = value, day
        if 1 - value / peak > maximum:
            maximum, pair = 1 - value / peak, (peak_day, day)
    return maximum, pair


def _published_source(bundle, ref, title):
    url = ref.get("url") or ""
    # 日常页面只给公开资料链接；采集接口留在原始来源记录中。
    if "/api/" in url or "/v2/" in url or url.startswith("https://api."):
        url = ""
    return {"title": title, "url": url, "publishedOn": ref.get("published_on")}


def _summary_source(inputs, source):
    """旧发布包有摘要锚却遗漏来源描述时，只定位该摘要原件，不扫描或借其他版本。"""
    digest = inputs["target_refs"].get("summary")
    if not digest:
        return None
    if digest not in source.sources:
        from app.services.material_content import source_exists

        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("NARRATIVE_SUMMARY_HASH")
        for suffix, kind in (("json", "json"), ("pdf", "pdf")):
            relative = f"sources/pipeline/{digest}.{suffix}"
            if source_exists(source.root / relative):
                source.sources[digest] = {"file": relative, "sha256": digest, "kind": kind}
                break
        else:
            return None
    # Originals对实际原件执行路径、摘要与大小检查，缺原件时继续走混合读取。
    pages = source.pages(digest)
    text = re.sub(r"\s+", "", "\n".join(pages))
    if inputs["fund_code"] not in text[:16000] or (inputs["share_class"] == "C" and "C" not in text[:16000]):
        raise ValueError("NARRATIVE_SUMMARY_SHARE")
    published = source.sources[digest].get("published_on")
    url = source.sources[digest].get("url")
    if source.sources[digest]["kind"] == "json":
        raw = source.json(digest).get("data", {})
        published = str(raw.get("notice_date", published or ""))[:10]
        article = raw.get("art_code")
        if article and re.fullmatch(r"AN\d+", article):
            url = f"https://fund.eastmoney.com/gonggao/{inputs['fund_code']},{article}.html"
    if not published:
        raise ValueError("NARRATIVE_SUMMARY_PUBLICATION_DATE_MISSING")
    if date.fromisoformat(published) > date.fromisoformat(inputs["as_of_date"]):
        raise ValueError("NARRATIVE_SUMMARY_PUBLICATION_DATE")
    return text, {**source.sources[digest], "url": url, "published_on": published}


def compile_facts(bundle: dict, dimensions: dict, *, root: Path) -> dict:
    """正式输入生成解释事实；旧公式只用于一致性校验，输出中不生成新分数。"""
    from app.services.fund_rating_direct_index import FEEDER_VERSION, Originals, _history
    from app.services.fund_rating_evidence_preview import preview
    from app.services.fund_rating_narrative_adapters import base_facts, money_facts, r5_facts, typed_facts
    from app.services.fund_rating_reviewed_snapshot import KEYS

    facts = base_facts(bundle)
    packet = bundle.get("evidence_v1_calculation", {})
    if not packet or bundle["family"] not in {"INDEX", "ACTIVE_EQUITY"}:
        if bundle.get("typed_disclosure_calculation"):
            return typed_facts(bundle, dimensions, root=root, facts=facts)
        if bundle.get("r5_calculation"):
            return r5_facts(bundle, facts=facts)
        if bundle["family"] == "MONEY":
            return money_facts(bundle, facts=facts)
        return facts
    replay = preview(bundle, root, review_required=False)
    for public, internal in KEYS.items():
        actual, expected = number(replay["dimensions"].get(internal)), number(dimensions.get(internal))
        if (actual is None) != (expected is None) or actual is not None and f"{actual:.2f}" != f"{expected:.2f}":
            raise ValueError("NARRATIVE_SCORE_REPLAY_MISMATCH:" + public)
    if replay["status"] == "EVENT_REVIEW":
        raise ValueError("NARRATIVE_EVENT_REVIEW")
    inputs = deepcopy(packet["inputs"])
    source = Originals(inputs, root, lazy=True)
    refs = inputs.get("target_refs", {})
    observed = replay["facts"]
    report_ref = {**inputs["sources"][refs["half"]], "published_on": bundle["report_published_on"]}
    if report_ref.get("kind") == "json":
        report_metadata = source.json(refs["half"]).get("data", {})
        article = report_metadata.get("art_code", "")
        if re.fullmatch(r"AN\d+", article):
            report_ref["url"] = f"https://fund.eastmoney.com/gonggao/{bundle['fund_code']},{article}.html"
    own_report = _published_source(bundle, report_ref, bundle["report_title"])
    for key in KEYS.values():
        facts["sources"][key] = [own_report]

    history = None
    with localcontext() as context:
        context.prec = 60
        if "fund_return" in observed:
            history = _history(source, inputs) if bundle["family"] == "INDEX" else inputs.get("history")
            if history:
                days = history["dates"]
                start = days.index(observed["from"])
                days = days[start:]
                nav = [number(v) for v in history["adjusted_nav"][start:]]
                reference = [number(v) for v in history["benchmark_index"][start:]]
                if any(v is None or v <= 0 for v in nav + reference):
                    raise ValueError("NARRATIVE_PERFORMANCE_VALUES")
                dd, pair = _drawdown(nav, days)
                reference_dd, _ = _drawdown(reference, days)
                facts["performance"] = {
                    **{
                        k: observed[k]
                        for k in ("fund_return", "reference_return", "drawdown", "tracking_error", "months_at_or_above")
                        if k in observed
                    },
                    "period": f"{observed['from']} 至 {observed['through']}",
                    "month_count": 12,
                    "reference_drawdown": str(reference_dd),
                    "drawdown_peak": pair[0],
                    "drawdown_trough": pair[1],
                }
                if abs(dd - number(observed["drawdown"])) > Decimal("1e-20"):
                    raise ValueError("NARRATIVE_DRAWDOWN_MISMATCH")
                for key in ("return", "risk", "stability", "efficiency"):
                    facts["sources"][key] = [
                        _published_source(
                            bundle,
                            inputs["sources"].get(refs.get(name), {}),
                            "本份额历史净值" if name == "provider" else "基金公司历史合同基准",
                        )
                        for name in ("provider", "benchmark")
                    ]

        report_text = re.sub(r"\s+", "", "\n".join(source.pages(refs["half"])))
        execution = {}
        if bundle["family"] == "INDEX":
            limit = re.search(r"年跟踪误差控制在([\d.]+)%以内", report_text)
            if limit:
                execution.update(
                    tracking_error_limit=str(Decimal(limit[1]) / 100), contract_date=bundle["holding_date"]
                )
            transition = re.search(r"自(20\d{2})年(\d{1,2})月(\d{1,2})日起[^。]{0,100}转型为本基金", report_text)
            if transition and history:
                changed = date(*map(int, transition.groups())).isoformat()
                if observed["from"] < changed <= observed["through"]:
                    execution["transition_date"] = changed
                    indices = [i for i, day in enumerate(days) if day >= changed]
                    if len(indices) >= 3:
                        from app.services.fund_rating_specialized import tracking

                        first = indices[0]
                        execution["post_transition_tracking_error"] = str(
                            tracking(nav[first:], reference[first:], 252)["tracking_error"]
                        )
            facts["sources"]["efficiency"].append(own_report)
        else:
            execution["description"] = bundle.get("public_explanations", {}).get("execution", "")
            execution.update(
                {k: observed[k] for k in ("stock_total_ratio", "stock_nav_ratio", "cash_nav_ratio") if k in observed}
            )
            execution.update(
                {
                    k: inputs["execution"][k]
                    for k in ("stock_min", "stock_max", "cash_min")
                    if k in inputs.get("execution", {})
                }
            )
        facts["execution"] = execution

    management = inputs.get("management_inputs")
    if management:
        from app.services.fund_rating_management_inputs import calculate as calculate_management

        computed = calculate_management(
            management,
            root=root,
            manager_name=inputs["manager_name"],
            as_of=inputs["as_of_date"],
            fund_code=inputs["fund_code"],
            legal_name=inputs["legal_name"],
            primary_report_hash=inputs.get("management_primary_report_hash", refs["half"]),
        )
        continuity = management.get("continuity") or {}
        spans = continuity.get("intervals", [])
        paragraphs = []
        names = {s["name"] for s in spans}
        if spans and len(names) == 1:
            first = min(s["start"] for s in spans)
            paragraphs.append(f"{next(iter(names))}自{first}起管理本产品；已核职责时间线覆盖至{bundle['as_of_date']}。")
            if facts["execution"].get("transition_date"):
                paragraphs.append("转型前后由同一经理继续管理，产品转型不等于更换经理。")
        elif names:
            paragraphs.append("已核任职记录涉及" + "、".join(sorted(names)) + "，经理职责按各自任职期间区分。")
        if computed["score"] is not None:
            paragraphs.append(
                "本项已核基本任职职责、责任接续和公开治理及托管履职。"
                "基本任职职责已证实，不因履历较长就直接认定投资能力优秀。"
            )
        governance = management.get("governance", {})
        if governance.get("state") == "NORMAL_REVIEWED":
            last = max(s["period_end"] for s in governance["statements"])
            paragraphs.append(
                f"已核多期管理人声明及本基金实际托管履职，最近所引报告期截至{last}；"
                "这些资料支持对应期间的正常履职判断。"
            )
        if paragraphs:
            facts["management"] = {
                "conclusion": "经理职责接续已有记录，管理分须按证据含义理解。",
                "paragraphs": paragraphs,
            }
    elif number(inputs.get("reviewed_management", {}).get("score")) is not None:
        facts["management"] = {
            "paragraphs": ["本批次保留了与所引报告绑定的管理分，尚未提供可展开的任职与分项事实，不从分数推断经理能力。"]
        }

    facts["holdings"] = {
        k: observed[k]
        for k in (
            "target_etf_nav_fraction",
            "major_holdings_count",
            "matched_major_holdings_count",
            "major_holdings_nav_fraction",
        )
        if k in observed
    }
    facts["index_name"] = inputs.get("index", {}).get("name")
    review = inputs.get("holder_review", {})
    mechanism = re.sub(r"\s+", "", review.get("mechanism", {}).get("quote", ""))
    facts["liquidity"] = {
        **{k: observed[k] for k in ("reported_redemption_alert", "reported_redemption_shares") if k in observed},
        "review_complete": review.get("actual_effect_review_complete") is True,
        "deferred_redemption": "巨额赎回" in mechanism and "延期" in mechanism,
    }
    if bundle["family"] == "INDEX" and review.get("actual_effect_review_complete"):
        from app.services.fund_rating_passive_portfolio import portfolio

        verified = None
        if inputs.get("security_disclosures"):
            from app.services.fund_rating_security_disclosures import verified_securities

            verified = verified_securities(source, inputs)
        own_book = portfolio(
            source.pages(refs["half"]), holding_date=bundle["holding_date"], verified_securities=verified
        )
        facts["liquidity"]["total_net_assets"] = own_book["balance"]["net_assets"]
    if refs.get("prospectus") in inputs["sources"]:
        facts["sources"]["liquidity"].append(
            _published_source(bundle, inputs["sources"][refs["prospectus"]], "本基金招募说明书及赎回安排")
        )
    if "annual_cost_per_10000" in observed:
        facts["cost"] = {"annual_cost_per_10000": observed["annual_cost_per_10000"]}
        try:
            summary = _summary_source(inputs, source)
        except ValueError as error:
            if str(error) != "NARRATIVE_SUMMARY_PUBLICATION_DATE_MISSING":
                raise
            summary = None
            facts["cost"]["detail_gap"] = "费用明细来源的公开日期尚未核准，本项仅展示正式评分已核的一年费用情景。"
        if summary:
            plain, ref = summary
            facts["fee_date"] = ref["published_on"]
            cost = facts["cost"]
            total = re.search(r"基金运作综合费率[（(]年化[）)]持有期([\d.]+)%", plain)
            if total:
                cost["annual_total_rate"] = str(Decimal(total[1]) / 100)
            cost["no_subscription_fee"] = "本基金份额类别不收取申购费" in plain
            if re.search(r"赎回费N[＜<]7日1\.50%7日[≤≦]N0\.00%", plain):
                cost["short_redemption"] = "持有不足7天的赎回费为1.50%，满7天为0%；按赎回金额一万元计算，前者约150元。"
            cost["feeder_fee_waiver"] = (
                inputs.get("version") == FEEDER_VERSION and "投资于目标ETF的部分不收取管理费、托管费" in plain
            )
            facts["sources"]["cost"] = [_published_source(bundle, ref, "本份额基金产品资料概要")]
    facts["source_sha256"] = sorted(set(observed.get("source_sha256", [])) | set(source.raw))
    return facts
