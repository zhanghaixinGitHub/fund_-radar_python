"""从已归档的份额原件重算一年口径的八维候选分。"""

from contextlib import contextmanager
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import (
    FEEDER_VERSION,
    VERSION,
    Originals,
    _flows,
    _history,
    _table,
    holder_execution_evidence,
)
from app.services.fund_rating_management_inputs import calculate as management_calculate
from app.services.fund_rating_market_review import annual_fee_score, fee_profile, normalized_text
from app.services.fund_rating_passive_holdings import _fund_holding, _identity
from app.services.fund_rating_passive_portfolio import portfolio
from app.services.fund_rating_r5 import PROTOCOL, number, signed_gap_score, tracking_score
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, PASSIVE_BASIS, policy
from app.services.fund_rating_specialized import tracking
from app.services.material_content import material_read_cache

REVISION = "EQUITY_DISCLOSURE_ONE_YEAR_20261110_V1"


def performance(history: dict) -> tuple[dict, dict]:
    """使用连续 13 个已核月末和首尾之间实际公布的日净值。"""
    days, ends = history["dates"], history["month_ends"][-13:]
    months = [int(day[:4]) * 12 + int(day[5:7]) for day in ends]
    if len(months) != 13 or months != list(range(months[0], months[0] + 13)) or ends[-1] != days[-1]:
        raise ValueError("EVIDENCE_PREVIEW_YEAR_WINDOW")
    start = days.index(ends[0])
    nav = list(map(number, history["adjusted_nav"][start:]))
    bench = list(map(number, history["benchmark_index"][start:]))
    if min(nav + bench) <= 0 or len(nav) != len(days[start:]) or len(bench) != len(nav):
        raise ValueError("EVIDENCE_PREVIEW_INVALID_HISTORY")
    positions = [days[start:].index(end) for end in ends]
    excess = nav[-1] / nav[0] - bench[-1] / bench[0]
    peak, drawdown = nav[0], Decimal(0)
    for value in nav:
        peak = max(peak, value)
        drawdown = max(drawdown, 1 - value / peak)
    wins = sum(nav[b] / nav[a] >= bench[b] / bench[a] for a, b in zip(positions[:-1], positions[1:], strict=True))
    error = tracking(nav, bench, 252)["tracking_error"]
    points = {
        "return": signed_gap_score(excess, PROTOCOL["parameters"]["annual_return_gap_unit"]),
        "risk": 100 * (1 - drawdown),
        "efficiency": tracking_score(error, PROTOCOL["parameters"]["tracking_economic_anchor_ratio"]),
        "stability": Decimal(wins * 100) / 12,
    }
    facts = {
        "from": ends[0],
        "through": ends[-1],
        "valuation_points": len(nav),
        "months_at_or_above": wins,
        "fund_return": str(nav[-1] / nav[0] - 1),
        "reference_return": str(bench[-1] / bench[0] - 1),
        "drawdown": str(drawdown),
        "tracking_error": str(error),
    }
    return points, facts


def management(inputs: dict, root: Path, report_hash: str):
    if not inputs.get("management_inputs"):
        reviewed = inputs.get("reviewed_management")
        if reviewed and reviewed.get("source_hash") == report_hash:
            return number(reviewed["score"])
        raise ValueError("EVIDENCE_PREVIEW_MANAGEMENT_INPUTS_REQUIRED")
    from app.services.fund_rating_management_anchor import verify_management_anchor

    verify_management_anchor(inputs, root, report_hash)
    value = management_calculate(
        inputs["management_inputs"],
        root=root,
        manager_name=inputs["manager_name"],
        as_of=inputs["as_of_date"],
        fund_code=inputs["fund_code"],
        legal_name=inputs["legal_name"],
        primary_report_hash=inputs.get("management_primary_report_hash", report_hash),
    )["score"]
    return number(value) if value is not None else None


def top_positions(positions: list[dict]) -> list[dict]:
    if not positions:
        raise ValueError("EVIDENCE_PREVIEW_NO_DISCLOSED_STOCKS")
    return sorted(positions, key=lambda row: (-number(row["market_value_cny"]), row["stock_code"]))[:10]


@contextmanager
def evidence_stage(errors: dict, group: str):
    """保留可独立核算的维度，缺失或无效的资料按组返回原因。"""
    try:
        yield
    except FileNotFoundError:
        errors[group] = "EVIDENCE_SOURCE_FILE_MISSING"
    except (KeyError, ValueError) as error:
        errors[group] = str(error)


def passive(inputs: dict, root: Path) -> tuple[dict, dict]:
    refs = inputs["target_refs"]
    source = Originals(inputs, root, lazy=True)
    holding_date = inputs["holding_date"]
    scores, facts, errors = {}, {}, {}

    def report_pages() -> list[str]:
        pages = _identity(source, refs["half"], inputs["fund_code"], inputs["legal_name"])
        if str(report_period("\n".join(pages))[1]) != holding_date:
            raise ValueError("EVIDENCE_PREVIEW_REPORT_PERIOD")
        return pages

    with evidence_stage(errors, "performance"):
        history = _history(source, inputs)
        if history["dates"][-1] != inputs["as_of_date"]:
            raise ValueError("EVIDENCE_PREVIEW_HISTORY_CUTOFF")
        scores, facts = performance(history)
        facts["benchmark"] = inputs["index"]["name"]

    with evidence_stage(errors, "holdings"):
        if not inputs.get("index", {}).get("weights"):
            # 此字段为同日成员表的来源摘要，不要求实际使用完整精确权重。
            raise ValueError("EVIDENCE_PREVIEW_INDEX_CONSTITUENTS_REQUIRED")
        verified = None
        if inputs.get("security_disclosures"):
            from app.services.fund_rating_security_disclosures import verified_securities

            verified = verified_securities(source, inputs)
        own = portfolio(report_pages(), holding_date=holding_date, verified_securities=verified)
        book = own
        if inputs["version"] == FEEDER_VERSION:
            target = inputs["etf_refs"][inputs["fund_code"]]
            relation = _fund_holding(source, refs["half"], target, own["balance"]["fund_amount"])
            target_pages = _identity(source, target["source_hash"], target["fund_code"], target["legal_name"])
            if str(report_period("\n".join(target_pages))[1]) != holding_date:
                raise ValueError("EVIDENCE_PREVIEW_TARGET_REPORT_PERIOD")
            book = portfolio(target_pages, holding_date=holding_date, verified_securities=verified)
            facts["target_etf"] = target["fund_code"]
            facts["target_etf_nav_fraction"] = str(
                number(relation["amount_cny"]) / number(own["balance"]["net_assets"])
            )
        members = set()
        for row in _table(source.json(inputs["index"]["weights"])):
            if (
                row["index_code"] != inputs["index"]["code"]
                or str(date.fromisoformat(row["trade_date"])) != holding_date
                or row["con_code"] in members
            ):
                raise ValueError("EVIDENCE_PREVIEW_INDEX_CONSTITUENT_SCOPE")
            members.add(row["con_code"])
        stocks = top_positions(book["positions"])
        amount = sum(number(row["market_value_cny"]) for row in stocks)
        matched = sum(number(row["market_value_cny"]) for row in stocks if row["stock_code"] in members)
        if not members or amount <= 0:
            raise ValueError("EVIDENCE_PREVIEW_INDEX_HOLDINGS_EMPTY")
        scores["holdings"] = 100 * matched / amount
        facts["major_holdings_count"] = len(stocks)
        facts["matched_major_holdings_count"] = sum(row["stock_code"] in members for row in stocks)

    with evidence_stage(errors, "management"):
        scores["management"] = management(inputs, root, refs["half"])

    with evidence_stage(errors, "liquidity"):
        if inputs.get("holder_structure_condition_protocol"):
            from app.services.fund_rating_cmf_liquidity import PROTOCOL, foundation

            if inputs["holder_structure_condition_protocol"] != PROTOCOL:
                raise ValueError("EVIDENCE_PREVIEW_HOLDER_PROTOCOL")
            node, proof = foundation(source, inputs)
            flows = proof["share_flows"]["rows"]
            facts["reported_redemption_alert"] = bool(node["confirmed_pressure_effects"])
            eligible = True
        else:
            review = inputs["holder_review"]
            facts["reported_redemption_alert"] = bool(review["confirmed_pressure_effects"])
            source.claim(review["mechanism"])
            mechanism = normalized_text(review["mechanism"]["quote"])
            holder_execution_evidence(source, review, refs["half"])
            flows, _ = _flows(
                report_pages(),
                allow_reversed=True,
                allow_interleaved=True,
                additional_shares=inputs.get("additional_shares", ()),
            )
            eligible = (
                all(term in mechanism for term in ("赎回", "巨额"))
                and review["actual_effect_review_complete"]
                and "未发现" in normalized_text(review["execution"]["quote"])
            )
        facts["reported_redemption_shares"] = next(
            row["redeemed"] for row in flows if row["share"] == inputs["share_class"]
        )
        scores["liquidity"] = Decimal(75) if eligible and not facts["reported_redemption_alert"] else None

    with evidence_stage(errors, "cost"):
        reviewed = inputs.get("reviewed_cost")
        if reviewed:
            if reviewed.get("source_hash") != refs["summary"]:
                raise ValueError("EVIDENCE_PREVIEW_REVIEWED_COST_SOURCE")
            if reviewed.get("scenario"):
                from app.services.fund_rating_locked_holding_cost import replay as replay_locked_cost

                actual_cost = replay_locked_cost(inputs, source)
                scores["cost"] = number(actual_cost["score"])
                facts["annual_cost_per_10000"] = actual_cost["annual_cost_per_10000"]
                facts["cost_scenario"] = actual_cost
            else:
                scores["cost"] = number(reviewed["score"])
                facts["annual_cost_per_10000"] = reviewed["annual_cost_per_10000"]
        else:
            fees = fee_profile(
                "\n".join(source.pages(refs["summary"])),
                code=inputs["fund_code"],
                share_class=inputs["share_class"],
                legal_name=inputs["legal_name"],
                supporting_text="\n".join(source.pages(refs["prospectus"])),
                published_on=source.sources[refs["summary"]]["published_on"],
            )
            cost = annual_fee_score(fees["fees"])
            scores["cost"], facts["annual_cost_per_10000"] = number(cost["score"]), cost["cost_cny"]

    facts["source_sha256"] = sorted(set(source.raw) | inputs.get("management_inputs", {}).get("sources", {}).keys())
    if errors:
        facts["evidence_errors"] = errors
    return scores, facts


def active_partial(bundle: dict, root: Path) -> tuple[dict, dict]:
    from app.services.fund_rating_health_stock_liquidity_interval import (
        check_binding,
        originals,
        plain,
        public_redemption,
        stock_book,
        text,
    )

    archive = bundle.get("health_stock_liquidity_interval_calculation")
    if archive is None:
        return {}, {}
    inputs = archive["inputs"]
    check_binding(bundle, inputs)
    source = originals(inputs, root)
    book, balance, _, _ = stock_book(source, inputs["primary_ref"], bundle["fund_code"])
    top = top_positions(book["holdings"])
    ratio = sum(number(row["market_value_cny"]) for row in top) / number(balance["net_assets"])
    if not 0 <= ratio <= 1:
        raise ValueError("EVIDENCE_PREVIEW_ACTIVE_HOLDINGS_RANGE")
    report_hash, legal_hash = inputs["primary_ref"], inputs["payment_legal"]
    report_text = text(source, report_hash)
    public_redemption(report_text, plain(text(source, legal_hash)))
    flows, _ = _flows([report_text])
    redeemed = next(row["redeemed"] for row in flows if row["share"] == bundle["share_class"])
    scores = {"holdings": 100 * (1 - ratio), "liquidity": Decimal(75)}
    used = {report_hash, legal_hash}
    managed = bundle.get("management_calculation")
    if managed:
        scores["management"] = management({**inputs, "management_inputs": managed["inputs"]}, root, report_hash)
        used.update(managed["inputs"]["sources"])
    return scores, {
        "major_holdings_nav_fraction": str(ratio),
        "major_holdings_count": len(top),
        "holding_date": inputs["holding_date"],
        "reported_redemption_shares": redeemed,
        "source_sha256": sorted(used),
    }


def active(inputs: dict, root: Path) -> tuple[dict, dict]:
    refs = inputs["target_refs"]
    source = Originals(inputs, root, lazy=True)
    scores, facts, errors = {}, {}, {}

    with evidence_stage(errors, "performance"):
        scores, facts = performance(inputs["history"])
        execution = inputs["execution"]
        stock_total = number(execution["stock_amount"]) / number(execution["total_assets"])
        stock_nav = number(execution["stock_amount"]) / number(execution["net_assets"])
        cash_nav = number(execution["cash_amount"]) / number(execution["net_assets"])
        source.claim(execution["strategy"])
        scores["efficiency"] = (
            Decimal(75)
            if number(execution["stock_min"]) <= stock_total <= number(execution["stock_max"])
            and number(execution["stock_min"]) <= stock_nav <= number(execution["stock_max"])
            and cash_nav >= number(execution["cash_min"])
            and execution["actual_review_complete"]
            else None
        )
        if inputs.get("execution_public_observations"):
            scores["efficiency"] = None
            from app.services.fund_rating_public_execution_core import calculate as public_execution_calculate

            execution_review = public_execution_calculate(inputs, source)
            scores["efficiency"] = number(execution_review["score"])
            facts["execution_public_review"] = execution_review
        facts.update(
            stock_total_ratio=str(stock_total),
            stock_nav_ratio=str(stock_nav),
            cash_nav_ratio=str(cash_nav),
        )

    with evidence_stage(errors, "holdings"):
        top = top_positions(inputs["positions"])
        ratio = sum(number(row["market_value_cny"]) for row in top) / number(inputs["net_assets"])
        if not 0 <= ratio <= 1:
            raise ValueError("EVIDENCE_PREVIEW_ACTIVE_HOLDINGS_RANGE")
        scores["holdings"] = 100 * (1 - ratio)
        facts["major_holdings_nav_fraction"] = str(ratio)
        facts["major_holdings_count"] = len(top)

    with evidence_stage(errors, "management"):
        scores["management"] = management(inputs, root, refs["half"])

    with evidence_stage(errors, "liquidity"):
        review = inputs["holder_review"]
        facts["reported_redemption_alert"] = bool(review["confirmed_pressure_effects"])
        source.claim(review["mechanism"])
        mechanism = normalized_text(review["mechanism"]["quote"])
        holder_execution_evidence(source, review, refs["half"])
        flows, _ = _flows(
            source.pages(refs["half"]),
            allow_reversed=True,
            allow_interleaved=True,
            additional_shares=inputs.get("additional_shares", ()),
            allow_period_variant=True,
        )
        facts["reported_redemption_shares"] = next(
            row["redeemed"] for row in flows if row["share"] == inputs["share_class"]
        )
        execution_text = normalized_text(review["execution"]["quote"])
        execution_normal = (
            "未发现" in execution_text
            or (
                review.get("execution_profile") == "TRUSTEE_QUARTER_REPORT_REVIEW"
                and "保证复核内容不存在虚假记载" in execution_text
            )
            or (
                review.get("execution_profile") == "TRUSTEE_SUPERVISION_AND_RECONCILIATION"
                and all(term in execution_text for term in ("报告期", "监督", "对账"))
            )
        )
        scores["liquidity"] = (
            Decimal(75)
            if all(term in mechanism for term in ("赎回", "巨额"))
            and review["actual_effect_review_complete"]
            and execution_normal
            and not facts["reported_redemption_alert"]
            else None
        )

    with evidence_stage(errors, "cost"):
        reviewed = inputs.get("reviewed_cost")
        if reviewed:
            if reviewed.get("source_hash") != refs["summary"]:
                raise ValueError("EVIDENCE_PREVIEW_REVIEWED_COST_SOURCE")
            if reviewed.get("scenario"):
                from app.services.fund_rating_locked_holding_cost import replay as replay_locked_cost

                actual_cost = replay_locked_cost(inputs, source)
                scores["cost"] = number(actual_cost["score"])
                facts["annual_cost_per_10000"] = actual_cost["annual_cost_per_10000"]
                facts["cost_scenario"] = actual_cost
            else:
                scores["cost"] = number(reviewed["score"])
                facts["annual_cost_per_10000"] = reviewed["annual_cost_per_10000"]
        else:
            fees = fee_profile(
                "\n".join(source.pages(refs["summary"])),
                code=inputs["fund_code"],
                share_class=inputs["share_class"],
                legal_name=inputs["legal_name"],
                supporting_text="\n".join(source.pages(refs["prospectus"])),
                published_on=source.sources[refs["summary"]]["published_on"],
            )
            cost = annual_fee_score(fees["fees"])
            scores["cost"], facts["annual_cost_per_10000"] = number(cost["score"]), cost["cost_cny"]

    facts["source_sha256"] = sorted(
        set(source.raw)
        | set(inputs.get("performance_source_hashes", []))
        | inputs.get("management_inputs", {}).get("sources", {}).keys()
    )
    if errors:
        facts["evidence_errors"] = errors
    return scores, facts


@material_read_cache
def preview(bundle: dict, root: Path, *, review_required: bool) -> dict:
    family = bundle["family"]
    basis = {"INDEX": PASSIVE_BASIS, "ACTIVE_EQUITY": ACTIVE_BASIS}.get(family)
    if basis is None:
        return {"status": "UNSUPPORTED_FAMILY", "score": None}
    weights = policy(basis)["weights"]
    raw = bundle.get("r5_calculation", {})
    frozen = bundle.get("evidence_v1_calculation")
    events = (
        frozen["events"] if frozen else raw.get("anchors", {}).get("material_events", {}).get(bundle["fund_code"], [])
    )
    blocked = review_required or any(
        event["applicability"] == "FUND_CONFIRMED"
        and event["status"] == "UNRESOLVED"
        and event["severity"] in {"MATERIAL", "SEVERE"}
        for event in events
    )
    with localcontext() as context:
        context.prec = 60
        scores, facts = {}, {}
        if not blocked and family == "INDEX":
            packet = raw.get("packet", {})
            inputs = (
                frozen["inputs"]
                if frozen
                else packet.get("direct_index_compilation") or packet.get("passive_index_compilation")
            )
            if inputs and inputs["version"] in {VERSION, FEEDER_VERSION}:
                if (
                    any(inputs[k] != bundle[k] for k in ("fund_code", "as_of_date", "holding_date", "share_class"))
                    or inputs["legal_name"] != bundle["identity"]["legal_name"]
                    or inputs["manager_name"] != bundle["identity"]["manager_name"]
                    or inputs["target_refs"]["half"] != bundle["original_report_archive"]["sha256"]
                ):
                    raise ValueError("EVIDENCE_PREVIEW_FUND_SCOPE")
                scores, facts = passive(inputs, root)
        elif not blocked:
            if frozen:
                inputs = frozen["inputs"]
                if inputs.get("version") != "ACTIVE_EQUITY_ONE_YEAR_V1":
                    raise ValueError("EVIDENCE_PREVIEW_ACTIVE_VERSION")
                if (
                    any(inputs[k] != bundle[k] for k in ("fund_code", "as_of_date", "holding_date", "share_class"))
                    or inputs["legal_name"] != bundle["identity"]["legal_name"]
                    or inputs["manager_name"] != bundle["identity"]["manager_name"]
                    or inputs["target_refs"]["half"] != bundle["original_report_archive"]["sha256"]
                ):
                    raise ValueError("EVIDENCE_PREVIEW_FUND_SCOPE")
                scores, facts = active(inputs, root)
            else:
                scores, facts = active_partial(bundle, root)
        blocked = blocked or facts.get("reported_redemption_alert", False)
        missing = [key for key in weights if scores.get(key) is None]
        total = None if blocked or missing else sum(scores[key] * weight / 100 for key, weight in weights.items())
        return {
            "revision": REVISION,
            "family": family,
            "status": "EVENT_REVIEW" if blocked else "PARTIAL" if missing else "CANDIDATE",
            "score": str(total.quantize(Decimal(".01"))) if total is not None else None,
            "dimensions": {key: str(scores[key]) if scores.get(key) is not None else None for key in weights},
            "missing": [] if blocked else missing,
            "facts": facts,
        }
