"""把可核验的流动性原件事实接入V2资料链，保留其与评分所需概念的差异。"""

from datetime import date


def observable_liquidity_facts(pages: list[str], *, observed_on: str, source_ref: dict,
                               restricted_metric: dict | None) -> dict:
    """返回同一报告的现金合计义务边界及运作原句，不生成计分字段。

    调用方先核报告主体、原字节和公开时点。三项各自重核报告期及正文，
    一项失败不丢失另一项；版式未识别与区间未识别点值分别保留。受限金额
    仅取同原件、全资产范围已核实的V2原始比例证据，不读旧分项分数。
    """
    from app.services.fund_quality_v2_cash_clauses import cash_report_clause
    from app.services.fund_quality_v2_cash_constraints import rebuild_cash_constraint
    from app.services.fund_quality_v2_disclosures import report_period
    from app.services.fund_quality_v2_operating_disclosure import operating_disclosure

    date.fromisoformat(observed_on)
    sha = source_ref["sha256"]
    result = {}
    def clause_observation() -> dict:
        """条款事实与数值边界分开保存，附注失败不丢失已识别的报告义务。"""
        text = "\n".join(pages)
        if report_period(text)[1].isoformat() != observed_on:
            raise ValueError("QUALITY_CASH_REPORTED_CLAUSE_ECONOMIC_PERIOD_MISMATCH")
        return cash_report_clause(pages) | {"observed_on": observed_on}

    builders = (
        ("cash_report_clause_observation", clause_observation),
        ("cash_contract_basket_observation", lambda: rebuild_cash_constraint(
            pages, observed_on=observed_on, source_hash=sha, restricted_metric=restricted_metric)),
        ("operating_report_observation", lambda: operating_disclosure(
            pages, observed_on=observed_on, source_hash=sha)),
    )
    for name, build in builders:
        try:
            value = build()
            result[name] = {**value, "source_hashes": [sha], "source_url": source_ref["url"],
                            "published_on": source_ref.get("published_on"),
                            "raw4_metric_replaced": False, "score_or_grade_created": False}
        except (ValueError, KeyError, TypeError) as exc:
            result[name + "_gap"] = {
                "reason": str(exc), "source_hashes": [sha], "source_url": source_ref["url"],
                "observed_on": observed_on, "published_on": source_ref.get("published_on"),
                "scope": "ORIGINAL_PRESENT_PARSER_OR_IDENTIFICATION_GAP_NOT_PROOF_OF_NONDISCLOSURE",
                "raw4_metric_replaced": False,
            }
    return result
