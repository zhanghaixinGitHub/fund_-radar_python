"""从当前评分所用原件重做清单验收，保留数据日期、原件与实际读回。"""

from datetime import date


def published_inputs(scope):
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import Session

    from app.db.session import get_nav_preview_engine
    from app.services.fund_rating_pipeline import current_inputs

    try:
        with Session(get_nav_preview_engine()) as session:
            rows = current_inputs(session, list(scope))
    except SQLAlchemyError:
        return {}
    return {
        code: {
            "batch_id": str(row["batch_id"]), "as_of": str(row["as_of_date"]),
            "bundle": next(member["bundle"] for member in row["input_snapshot"]["members"] if member["fund_code"] == code),
        }
        for code, row in rows.items()
    }


def dated_scoring_input(record, code, day):
    from app.services.fund_rating_pipeline import _period_covered

    if not record:
        return None
    bundle = record["bundle"]
    observed, requested = date.fromisoformat(record["as_of"]), date.fromisoformat(day)
    if bundle.get("fund_code") != code or bundle.get("as_of_date") != record["as_of"]:
        return None
    if not _period_covered(bundle, observed, requested):
        return None
    return bundle


def archived_documents(inputs, root, *names):
    from app.services.fund_rating_direct_index import Originals

    refs = inputs["target_refs"]
    sources = dict(inputs["sources"])
    for name in names:
        sha = refs[name]
        if sha not in sources and name == "summary":
            sources[sha] = pipeline_product_summary(inputs, root, sha)
    source = Originals({**inputs, "sources": sources}, root, lazy=True)
    documents, provenance = {}, {}
    for name in names:
        sha = refs[name]
        documents[name] = "\n".join(source.pages(sha))
        ref = source.sources[sha]
        provenance[name] = {"sha256": sha, "path": str(root / ref["file"]), "published_on": ref.get("published_on")}
    return source, documents, provenance


def pipeline_product_summary(inputs, root, sha):
    import json
    import re

    from app.services.fund_rating_market_review import normalized_text
    from app.services.material_content import content_digest, source_bytes

    if not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("DAILY_ORIGINAL_SUMMARY_DIGEST")
    path = root / "sources" / "pipeline" / f"{sha}.json"
    raw = source_bytes(path)
    if content_digest(raw) != sha:
        raise ValueError("DAILY_ORIGINAL_SUMMARY_HASH")
    record = json.loads(raw)["data"]
    published = record["notice_date"][:10]
    if (date.fromisoformat(published) > date.fromisoformat(inputs["as_of_date"])
            or "基金产品资料概要" not in record["notice_title"]
            or normalized_text(inputs["legal_name"]) not in normalized_text(record["notice_title"])):
        raise ValueError("DAILY_ORIGINAL_SUMMARY_IDENTITY_DATE")
    return {"file": path.relative_to(root).as_posix(), "sha256": sha, "published_on": published,
            "archive_kind": "ORIGINAL"}


def v1_inputs(bundle, root):
    from app.services.fund_rating_reviewed_snapshot import _read

    calc = bundle.get("evidence_v1_calculation")
    if not calc or _read(bundle["evidence_v1_archive"], root) != calc:
        return None
    inputs = calc["inputs"]
    if (any(inputs.get(k) != bundle.get(k) for k in ("fund_code", "as_of_date", "holding_date", "share_class"))
            or inputs.get("legal_name") != bundle["identity"]["legal_name"]
            or inputs.get("manager_name") != bundle["identity"]["manager_name"]
            or inputs.get("target_refs", {}).get("half") != bundle["original_report_archive"]["sha256"]):
        return None
    return inputs


def fees_from_originals(record, code, day):
    from app.services.fund_rating_market_review import annual_fee_score, fee_profile
    from app.services.fund_rating_reviewed_snapshot import archive_root

    bundle = dated_scoring_input(record, code, day)
    if not bundle:
        return None
    root = archive_root()
    inputs = v1_inputs(bundle, root)
    if not inputs:
        return typed_fees(record, bundle, root)
    source, documents, refs = archived_documents(inputs, root, "summary", "prospectus")
    reviewed = inputs.get("reviewed_cost") or {}
    if reviewed.get("scenario"):
        from app.services.fund_rating_locked_holding_cost import replay

        result = replay(inputs, source)
    else:
        terms = fee_profile(
            documents["summary"], code=code, share_class=inputs["share_class"], legal_name=inputs["legal_name"],
            supporting_text=documents["prospectus"], published_on=refs["summary"]["published_on"],
        )
        result = {"terms": terms, "one_year_calculation": annual_fee_score(terms["fees"])}
    return {"fund_code": code, "as_of": record["as_of"], "batch_id": record["batch_id"],
            "sources": refs, "public_fee_inputs": result}


def typed_fees(record, bundle, root):
    from app.services.fund_rating_market_review import annual_fee_score, fee_profile
    from app.services.fund_rating_reviewed_snapshot import _read
    from app.services.fund_rating_typed_disclosure import check_binding, validate_input_archive

    raw = bundle.get("typed_disclosure_calculation")
    if not raw:
        return None
    validate_input_archive(bundle, root=root, archive_reader=_read)
    _, inputs = check_binding(bundle)
    if raw["profile"] == "QDII_ACTIVE":
        from app.services.fund_rating_qdii_disclosure import Originals, fees_from_original

        original = Originals(inputs, root)
        fees = fees_from_original(original, inputs)
        refs = {label: inputs[key] for label, key in (("summary", "summary_ref"), ("prospectus", "legal_ref"))}
        proof = {key: {"sha256": sha, "path": str(root / inputs["sources"][sha]["file"])} for key, sha in refs.items()}
    else:
        refs = inputs.get("refs") or {"summary": inputs.get("summary_ref"), "prospectus": inputs.get("prospectus_ref")}
        if not refs.get("summary") or not refs.get("prospectus"):
            return None
        _, docs, proof = archived_documents({**inputs, "target_refs": refs}, root, "summary", "prospectus")
        terms = fee_profile(
            docs["summary"], code=inputs["fund_code"], share_class=inputs["share_class"], legal_name=inputs["legal_name"],
            supporting_text=docs["prospectus"], published_on=proof["summary"]["published_on"],
        )
        fees = terms["fees"]
    return {"fund_code": inputs["fund_code"], "as_of": record["as_of"], "batch_id": record["batch_id"],
            "sources": proof, "public_fee_inputs": {"terms": fees, "one_year_calculation": annual_fee_score(fees)}}


def fund_manager_originals(record, code, day):
    from app.services.fund_rating_evidence_preview import management
    from app.services.fund_rating_reviewed_snapshot import archive_root

    bundle = dated_scoring_input(record, code, day)
    if not bundle:
        return None
    root = archive_root()
    inputs = v1_inputs(bundle, root)
    if not inputs or not inputs.get("management_inputs"):
        return None
    report = inputs["target_refs"]["half"]
    result = management(inputs, root, report)
    if result is None:
        return None
    docs = inputs["management_inputs"]["sources"]
    return {"fund_code": code, "as_of": record["as_of"], "batch_id": record["batch_id"],
            "primary_report": report, "management_originals": {
                sha: str(root / ref["file"]) for sha, ref in docs.items()
            }}


def performance_originals(record, code, day):
    """核对授权复权净值、合同基准和实际月末，提供可直接使用的原始时间序列。"""
    from app.services.fund_rating_direct_index import Originals, _history
    from app.services.fund_rating_evidence_preview import performance
    from app.services.fund_rating_public_market import provider_history
    from app.services.fund_rating_reviewed_snapshot import archive_root

    bundle = dated_scoring_input(record, code, day)
    if not bundle:
        return None
    root = archive_root()
    inputs = v1_inputs(bundle, root)
    if not inputs:
        return None
    refs = inputs["target_refs"]
    originals = Originals(inputs, root, lazy=True)
    official_comparison = bundle.get("family") == "INDEX"
    if official_comparison:
        series = _history(originals, inputs)
        _, facts = performance(series)
        rows = [{
            "date": month, "adjusted_nav": series["adjusted_nav"][series["dates"].index(month)],
            "benchmark_level": series["benchmark_index"][series["dates"].index(month)],
        } for month in series["month_ends"][-13:]]
    else:
        provider = originals.json(refs["provider"])
        if (provider.get("fund_code") != code or provider.get("source", {}).get("enabled") is not True
                or not provider.get("source", {}).get("authorization_verified_at")):
            return None
        metrics, series = provider_history(provider, date.fromisoformat(record["as_of"]))
        facts = {"from": series["dates"][0], "through": series["dates"][-1],
                 "valuation_points": len(series["dates"]), "drawdown": str(metrics["drawdown"])}
        rows = [{"date": month, "adjusted_nav": series["adjusted_nav"][series["dates"].index(month)]}
                for month in series["month_ends"][-13:]]
    if facts["through"] != record["as_of"] or len(rows) != 13:
        return None
    sources = {sha: str(root / originals.sources[sha]["file"]) for sha in originals.raw}
    return {"fund_code": code, "as_of": record["as_of"], "batch_id": record["batch_id"],
            "monthly_levels": rows, "facts": facts, "sources": sources,
            "official_comparison": official_comparison, "peer_count": bundle.get("comparison_product_count")}


def target_contract_index(record, owner, target):
    from app.services.fund_rating_market_review import normalized_text
    from app.services.fund_rating_passive_holdings import _fund_holding
    from app.services.fund_rating_passive_portfolio import portfolio
    from app.services.fund_rating_reviewed_snapshot import archive_root

    if not record or record["bundle"].get("fund_code") != owner:
        return None
    root = archive_root()
    inputs = v1_inputs(record["bundle"], root)
    relation = (inputs or {}).get("etf_refs", {}).get(owner, {})
    if relation.get("fund_code") != target:
        return None
    index = inputs.get("index") or {}
    if not index.get("code") or not index.get("name"):
        return None
    source, docs, proof = archived_documents(inputs, root, "summary", "prospectus", "half")
    # 招募书可能合并全部份额。用产品概要确认本份额，再从实际持有的底层基金核目标身份。
    text = normalized_text(docs["prospectus"])
    summary = normalized_text(docs["summary"])
    legal = normalized_text(inputs["legal_name"])
    label = normalized_text(index.get("contract_name", index["name"]))
    if owner not in summary or legal not in summary or legal not in text:
        return None
    if "目标ETF" not in text or "标的指数" not in text or label not in text:
        return None
    if index.get("contract_name", index["name"]) != index["name"]:
        claim = index["identity_claim"]
        if claim["source_hash"] in source.sources:
            source.claim(claim)
            original = normalized_text(claim["quote"])
            ref = source.sources[claim["source_hash"]]
            publisher = {"sha256": claim["source_hash"], "path": str(root / ref["file"]),
                         "published_on": ref.get("published_on")}
        elif index["code"] in {"930653.CSI", "930910.CSI"}:
            original, publisher = csi_index_original(index["code"])
        else:
            return None
        if not all(value in original for value in (label, index["code"].split(".")[0], normalized_text(index["name"]))):
            return None
        proof["index_publisher"] = publisher
    own = portfolio(source.pages(inputs["target_refs"]["half"]), holding_date=inputs["holding_date"])
    target_holding = _fund_holding(source, inputs["target_refs"]["half"], relation, own["balance"]["fund_amount"])
    return {"code": index["code"], "name": index.get("contract_name", index["name"]),
            "sources": proof, "target_holding": target_holding}


def csi_index_original(code):
    import httpx

    from app.services.fund_exposure_common import ROOT, blob, now
    from app.services.fund_rating_market_review import normalized_text
    from app.services.material_content import document_reader

    filename, terms = {
        "930653.CSI": ("930653hbook.pdf", ("指数名称:中证食品饮料指数", "指数简称:CS食品饮", "指数代码:930653")),
        "930910.CSI": ("930910factsheet.pdf", ("中证全指农牧渔指数930910农牧渔", "中证指数有限公司")),
    }[code]
    url = ("https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/"
           f"indices/detail/files/zh_CN/{filename}")
    with httpx.Client(timeout=httpx.Timeout(12, connect=5), trust_env=False) as client:
        response = client.get(url)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF-") or len(response.content) > 2_000_000:
        raise ValueError("DAILY_CSI_INDEX_ORIGINAL_NOT_PDF")
    text = normalized_text("\n".join(page.extract_text() or "" for page in document_reader(response.content).pages))
    if not all(term in text for term in terms):
        raise ValueError("DAILY_CSI_INDEX_ORIGINAL_IDENTITY")
    sha, path = blob(response.content, "pdf")
    return text, {"url": url, "sha256": sha, "path": str(ROOT / path), "retrieved_at": now().isoformat()}


def portfolio_originals(record, code, day):
    """从评分所绑定的原报告提取实际资产和份额流动，分别保留核对结果。"""
    from app.services.fund_quality_v2_balance_proofs import strict_balance_values
    from app.services.fund_quality_v2_disclosures import report_period
    from app.services.fund_quality_v2_stock_book import periodic_financial_view
    from app.services.fund_rating_direct_index import Originals, _flows
    from app.services.fund_rating_passive_holdings import _fund_holding, _identity
    from app.services.fund_rating_passive_portfolio import lookthrough, portfolio
    from app.services.fund_rating_r5 import number
    from app.services.fund_rating_reviewed_snapshot import _read, archive_root
    from app.services.fund_rating_typed_disclosure import check_binding, validate_input_archive

    bundle = dated_scoring_input(record, code, day)
    if not bundle:
        return None
    root = archive_root()
    inputs = v1_inputs(bundle, root)
    if inputs:
        primary = inputs["target_refs"]["half"]
    elif bundle.get("typed_disclosure_calculation"):
        validate_input_archive(bundle, root=root, archive_reader=_read)
        _, inputs = check_binding(bundle)
        primary = inputs["primary_ref"]
    else:
        return None
    if primary != bundle["original_report_archive"]["sha256"]:
        return None
    if inputs.get("profile") == "QDII_ACTIVE":
        return qdii_portfolio_originals(record, code, root, inputs)
    source = Originals(inputs, root, lazy=True)
    pages = _identity(source, primary, code, inputs["legal_name"])
    start, end = report_period("\n".join(pages))[:2]
    if str(end) != inputs["holding_date"]:
        raise ValueError("RATING_ORIGINAL_PORTFOLIO_PERIOD_MISMATCH")
    proof = {
        "fund_code": code, "as_of": record["as_of"], "report_from": str(start), "report_to": str(end),
        "currency": "CNY", "validation_errors": {},
    }
    for component in ("asset_balances", "share_flows", "stock_holdings"):
        try:
            if component == "asset_balances":
                view, _ = periodic_financial_view(pages)
                proof[component] = strict_balance_values(view, observed_on=str(end))["point_book"]
            elif component == "share_flows":
                flows, _ = _flows(pages, allow_reversed=True, allow_interleaved=True,
                                  additional_shares=inputs.get("additional_shares", ()), allow_period_variant=True)
                proof[component] = next(row for row in flows if row["share"] == inputs["share_class"])
            elif bundle.get("family") in {"INDEX", "ACTIVE_EQUITY"} and inputs.get("target_refs"):
                own = portfolio(pages, holding_date=str(end))
                positions = own["positions"]
                target = inputs.get("etf_refs", {}).get(code)
                if target:
                    relation = _fund_holding(source, primary, target, own["balance"]["fund_amount"])
                    target_pages = _identity(source, target["source_hash"], target["fund_code"], target["legal_name"])
                    if report_period("\n".join(target_pages))[1] != end:
                        raise ValueError("RATING_ORIGINAL_TARGET_PORTFOLIO_PERIOD_MISMATCH")
                    underlying = portfolio(target_pages, holding_date=str(end))
                    proof["underlying"] = {"relation": relation, "portfolio": underlying["balance"]}
                    proof["stock_risk_budget"] = lookthrough(own, underlying, fund_holding_cny=relation["amount_cny"])
                proof[component] = positions
                net_assets = number(own["balance"]["net_assets"])
                if positions and net_assets > 0:
                    values = sorted((number(row["market_value_cny"]) for row in positions), reverse=True)
                    proof["top_ten_stock_fraction_of_net_assets"] = str(sum(values[:10]) / net_assets)
        except (ArithmeticError, KeyError, TypeError, ValueError, StopIteration) as error:
            proof["validation_errors"][component] = str(error)[:160]
    proof["sources"] = {sha: str(root / source.sources[sha]["file"]) for sha in source.raw}
    return proof


def qdii_portfolio_originals(record, code, root, inputs):
    from app.services.fund_rating_qdii_disclosure import (
        Originals, execution, plain, portfolio, primary_binding, share_flows,
    )

    source = Originals(inputs, root)
    primary = inputs["primary_ref"]
    report = source.body(primary)
    header = plain(report[:18000])
    if code not in header or plain(inputs["legal_name"]) not in header:
        raise ValueError("RATING_QDII_ORIGINAL_REPORT_IDENTITY")
    binding = primary_binding(report, inputs)
    balance, stocks = portfolio(source, inputs, report)
    sources = {sha: str(root / source.sources[sha]["file"]) for sha in
               {primary, inputs["legal_ref"], inputs.get("stock_original_ref", primary)}}
    proof = {"fund_code": code, "as_of": record["as_of"], "currency": "CNY",
             "report_to": binding["cover_period"], "asset_balances": balance, "stock_holdings": stocks,
             "sources": sources, "validation_errors": {}}
    try:
        proof["share_flows"] = next(row for row in share_flows(report, inputs) if row["share"] == inputs["share_class"])
    except (ValueError, StopIteration) as error:
        proof["validation_errors"]["share_flows"] = str(error)[:160]
    try:
        _, proof["execution_snapshot"] = execution(source, inputs, report, balance)
    except (ValueError, KeyError, ArithmeticError) as error:
        proof["validation_errors"]["execution_snapshot"] = str(error)[:160]
    return proof
