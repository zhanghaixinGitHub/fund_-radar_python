"""本期月评编译队列；由每日入口消费真实新期研究包，不再只检查旧评分。"""

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.services.fund_exposure_common import now
from app.services.fund_materials_store import versioned_save
from app.services.fund_rating_period import last_completed_month
from app.services.fund_rating_pipeline import _period_covered, archive_json, current_inputs, register_update
from app.services.fund_rating_reviewed_snapshot import BASIS, KEYS, archive_root


def _partial_bundle(bundle):
    dimensions = bundle.get("reviewed_document", {}).get("dimensions", {})
    typed = bundle.get("typed_disclosure_calculation")
    typed_partial = False
    if typed:
        from app.services.fund_rating_typed_disclosure import PROFILES

        typed_partial = not PROFILES.get(typed.get("profile"), {}).get("enabled", False)
    return (
        bundle.get("evidence_v1_calculation", {}).get("partial_release_only") is True
        or bool(bundle.get("bosera_performance_calculation"))
        or bool(bundle.get("active_r5_partial_calculation"))
        or typed_partial
        or any(
            value is None or isinstance(value, dict) and value.get("candidate") is None for value in dimensions.values()
        )
    )


def _dossier_bundle(dossier, root):
    from app.services.fund_rating_reviewed_snapshot import _case, _read

    if not dossier.get("entries"):
        return {}
    entry = dossier["entries"][0]
    return {
        **entry,
        "reviewed_document": _case(
            _read(entry["research_archive"], root), entry["fund_code"], entry["research_format"]
        ),
    }


def _new_complete_prevents_partial(root, code, candidate_bundle):
    """登记前再读正式/已登记版本，未实际重审的旧候选不能丢失已验收维度。"""
    if not candidate_bundle or not _partial_bundle(candidate_bundle):
        return False
    cutoff = date.fromisoformat(candidate_bundle["as_of_date"])
    with Session(get_nav_preview_engine()) as session:
        current = current_inputs(session, [code]).get(code)
    if current and current["as_of_date"] >= cutoff:
        present = next(m["bundle"] for m in current["input_snapshot"]["members"] if m["fund_code"] == code)
        if _loses_accepted_dimensions(candidate_bundle, present) and not _actual_invalidation_review(
            candidate_bundle, present, root
        ):
            return True
    pointer = Path(root) / "pipeline" / "updates" / (code + ".json")
    if pointer.exists():
        from app.services.fund_rating_reviewed_snapshot import _read

        staged = json.loads(pointer.read_text(encoding="utf-8"))
        if staged.get("fund_code") != code:
            raise ValueError("MONTHLY_STAGED_FUND_IDENTITY")
        if date.fromisoformat(staged["as_of_date"]) >= cutoff:
            present = _dossier_bundle(_read(staged["dossier"], root), root)
            if (
                present
                and _loses_accepted_dimensions(candidate_bundle, present)
                and not _actual_invalidation_review(candidate_bundle, present, root)
            ):
                return True
    return False


def _known_dimensions(bundle):
    return {
        key
        for key, value in bundle.get("reviewed_document", {}).get("dimensions", {}).items()
        if value is not None and (not isinstance(value, dict) or value.get("candidate") is not None)
    }


def _loses_accepted_dimensions(candidate, previous):
    return bool(_known_dimensions(previous) - _known_dimensions(candidate))


def _actual_invalidation_review(candidate, previous, root):
    """真实新增原件使旧维失效时允许显式降级；不接受单独的allow标记或更换时间。"""
    from datetime import datetime

    from app.services.fund_rating_management_inputs import _plain, _source_text

    review = candidate.get("monthly_reassessment_review", {})
    lost = _known_dimensions(previous) - _known_dimensions(candidate)
    if (
        review.get("fund_code") != candidate.get("fund_code")
        or review.get("as_of_date") != candidate.get("as_of_date")
        or set(review.get("invalidated_dimensions", [])) != lost
        or not previous.get("research_archive", {}).get("sha256")
        or review.get("supersedes_research_sha256") != previous["research_archive"]["sha256"]
        or len(review.get("reason", "").strip()) < 80
        or not review.get("evidence")
    ):
        return False

    def refs(value):
        if isinstance(value, dict):
            if value.get("sha256"):
                yield value["sha256"]
            for item in value.values():
                yield from refs(item)
        elif isinstance(value, list):
            for item in value:
                yield from refs(item)

    previous_hashes = set(refs(previous))
    candidate_hashes = set(
        refs({key: value for key, value in candidate.items() if key != "monthly_reassessment_review"})
    )
    try:
        reviewed_at = datetime.fromisoformat(review["reviewed_at"])
        if reviewed_at.tzinfo is None or reviewed_at > now():
            return False
        for item in review["evidence"]:
            reference = item["source"]
            digest = reference["sha256"]
            quote = _plain(item["quote"])
            if (
                digest in previous_hashes
                or digest not in candidate_hashes
                or len(quote) < 20
                or quote not in _source_text(root, reference)
            ):
                return False
        return True
    except (KeyError, ValueError, TypeError, OSError):
        return False


def _input_identity(value):
    """观察/读取时刻不产生新的业务输入版本；原件摘要、真实日期和判断正文仍参与。"""
    if isinstance(value, dict):
        volatile = {
            "observed_at",
            "normalized_records_observed_at",
            "verified_at",
            "checked_at",
            "reviewed_at",
            "acquired_at",
            "received_at",
            "imported_at",
        }
        return {key: _input_identity(item) for key, item in value.items() if key not in volatile}
    if isinstance(value, list):
        return [_input_identity(item) for item in value]
    return value


def _registration_identity(dossier):
    """派生封套的取得时刻不造新评分；真实原件摘要、观察和研究文档仍逐项参与。"""
    stable = _input_identity(dossier)
    for entry in stable.get("entries", []):
        entry.pop("evidence_v1_archive", None)
        for key, value in entry.items():
            if key.endswith("_calculation") and isinstance(value, dict) and "inputs" in value:
                value.pop("input_archive", None)
    return stable


def _current_missing_reason(candidate, key, internal, result):
    supplied = candidate.get("monthly_missing_reasons", {})
    if supplied.get(key) or supplied.get(internal):
        return supplied.get(key) or supplied[internal]
    old = candidate.get("public_explanations", {}).get(key, "")
    if any(word in old for word in ("尚缺", "尚未全部", "尚未核", "暂不评分", "未全部核齐")):
        return old
    errors = result.get("facts", {}).get("evidence_errors", {})
    stage = "performance" if internal in {"return", "risk", "efficiency", "stability"} else internal
    detail = str(errors.get(stage, ""))
    missing_day = (
        candidate.get("evidence_v1_calculation", {}).get("inputs", {}).get("history", {}).get("benchmark_missing")
    )
    if stage == "performance" and missing_day:
        detail = str(missing_day["reason"])
    descriptions = {
        "EVIDENCE_PREVIEW_MANAGEMENT_INPUTS_REQUIRED": (
            "缺少本期实际经理职责、责任接续和适用事项的原件审阅，旧期管理分未沿用"
        ),
        "EVIDENCE_PREVIEW_INDEX_CONSTITUENTS_REQUIRED": (
            "缺少适用日期的目标指数完整成分原件，不能据指数名称判定持仓符合性"
        ),
        "R5_CMF_BRIDGE_REQUEST_OR_RESPONSE": "新期招商官网曲线的实际请求或响应未通过核验，不能沿用旧窗绩效",
    }
    reason = descriptions.get(detail, detail or "该维所需本期原件或实际判断未闭合")
    return f"截至{candidate['as_of_date']}，本项尚缺资料：{reason}；暂不评分。"


def _record_missing_explanation(candidate, key, reason):
    previous = candidate.setdefault("public_explanations", {}).get(key)
    if previous and previous != reason:
        history = candidate.setdefault("monthly_historical_explanations", {}).setdefault(key, [])
        if previous not in history:
            history.append(previous)
    candidate["public_explanations"][key] = reason


def generate_monthly_inputs(code, bundle, target, *, root, output, review=None):
    """先补真实净值；只有本基金、本期原件审阅到位才调用候选生成器。"""
    from app.services.direction_1d_protocol import calendar, digest
    from app.services.fund_exposure_common import read
    from app.services.fund_rating_monthly_inputs import _derived, build_monthly_inputs, provider_snapshot

    period = str(target)[:7]
    missing = []
    if code == "017437" and (not bundle or bundle.get("fee_calculation")):
        from app.services.fund_rating_pipeline_prepare import prepare_fees
        from app.services.fund_rating_reviewed_snapshot import _case, _read

        if not review or not review.get("fee_inputs"):
            return {"status": "MATERIALS_PENDING", "missing": [{
                "field": "fee_inputs",
                "reason": "本基金可独立续办费用首评；须提交本C概要和自身报告实际原件及审阅，其他七维保持未知",
            }]}
        raw = review["fee_inputs"]
        asof = str(raw.get("as_of_date", ""))
        if (review.get("fund_code") != code or raw.get("fund_code") != code
                or review.get("as_of_date") != asof or asof[:7] != period or date.fromisoformat(asof) > target
                or review.get("legal_name") != raw.get("legal_name")
                or review.get("share_class") != raw.get("share_class")
                or review.get("primary_report_hash") != raw.get("primary_ref")):
            raise ValueError("MONTHLY_PERIOD_REVIEW_FUND_PERIOD_ORIGINAL_MISMATCH")
        with Session(get_nav_preview_engine()) as session:
            dossier = prepare_fees(session, {"fund_code": code, "as_of_date": asof,
                                    "initial_partial_fee": True, "inputs": raw,
                                    "explanation": review.get("fee_explanation", "")}, root=root)
        candidate = {**dossier["entries"][0], "basis": BASIS}
        candidate["reviewed_document"] = _case(
            _read(candidate["research_archive"], root), code, candidate["research_format"]
        )
        destination = Path(output) / "candidate-bundle.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix(".pending")
        pending.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(destination)
        return {"fund_code": code, "as_of_date": asof, "status": "GENERATED_FOR_COMPILATION",
                "candidate_path": str(destination.resolve()), "missing": []}
    if bundle.get("bosera_performance_calculation"):
        # 博时已有原件工厂独立核验其境外估值日期，不能误走境内三年净值骨架。
        from app.services.fund_rating_bosera_qdii_performance import check_binding, prepare
        from app.services.fund_rating_reviewed_snapshot import _case, _read

        if not review or not review.get("bosera_inputs"):
            return {"status": "MATERIALS_PENDING", "missing": [{
                "field": "bosera_inputs",
                "reason": "须提交本C份额本期真实原件输入及收益、持续性审阅；不能复制旧分或自动补齐其他四维",
            }]}
        raw = review["bosera_inputs"]
        asof = str(raw.get("as_of_date", ""))
        if (
            raw.get("fund_code") != code or asof[:7] != period or date.fromisoformat(asof) > target
            or review.get("fund_code") != code or review.get("as_of_date") != asof
            or review.get("legal_name") != bundle["identity"]["legal_name"]
            or review.get("share_class") != bundle["share_class"]
            or review.get("primary_report_hash") != bundle["original_report_archive"]["sha256"]
        ):
            raise ValueError("MONTHLY_PERIOD_REVIEW_FUND_PERIOD_ORIGINAL_MISMATCH")
        check_binding(bundle, raw)
        with Session(get_nav_preview_engine()) as session:
            dossier = prepare(session, {
                "fund_code": code, "as_of_date": asof, "inputs": raw,
                "explanations": review.get("explanations", {}),
            }, root=root)
        candidate = {**dossier["entries"][0], "basis": BASIS}
        candidate["reviewed_document"] = _case(
            _read(candidate["research_archive"], root), code, candidate["research_format"]
        )
        destination = Path(output) / "candidate-bundle.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix(".pending")
        pending.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(destination)
        return {"fund_code": code, "as_of_date": asof, "status": "GENERATED_FOR_COMPILATION",
                "candidate_path": str(destination.resolve()), "missing": []}
    pending_performance = bundle.get("monthly_performance_followup", {}).get("pending_performance_source_archive")
    if pending_performance:
        # 已补出的新同行、原件和共同日期必须续接，不能再落回旧类型骨架。
        from app.services.fund_rating_monthly_huaan import pending_performance_bundle

        restored = pending_performance_bundle(bundle, root=root)
        restored.setdefault("monthly_performance_followup", {})["pending_performance_source_archive"] = (
            pending_performance
        )
        if str(restored["as_of_date"])[:7] != period:
            return {
                "status": "MATERIALS_PENDING",
                "pending_performance_source_archive": pending_performance,
                "missing": [
                    {
                        "field": "huaan_current_period_originals_and_review",
                        "reason": "已保留上期待补原件图；新月须实际滚动本期原件并重新审阅，不能改日期沿用",
                    }
                ],
            }
        from app.services.fund_rating_monthly_huaan import resume_monthly_performance

        return resume_monthly_performance(bundle, root=root, output=output)
    if not any(
        bundle.get(key)
        for key in (
            "evidence_v1_calculation",
            "typed_disclosure_calculation",
            "r5_calculation",
            "active_r5_partial_calculation",
        )
    ):
        return {
            "status": "INPUT_ADAPTER_REQUIRED",
            "missing": [{"field": "type_specific_adapter", "reason": "须使用本基金类型真实输入构造器"}],
        }
    sessions, _ = calendar()
    valuation_dates = [day for day in sessions if str(day)[:7] == period and day <= target]
    if not valuation_dates:
        return {
            "status": "MATERIALS_PENDING",
            "missing": [{"field": "official_calendar", "reason": "本期官方净值日尚未核准"}],
        }
    cutoff = max(valuation_dates)
    typed = bundle.get("typed_disclosure_calculation")
    foreign_typed = typed and typed["profile"] in {"QDII_ACTIVE", "GOLD"}
    if foreign_typed and review and str(review.get("as_of_date", ""))[:7] == period:
        cutoff = date.fromisoformat(review["as_of_date"])
        if cutoff > target:
            raise ValueError("MONTHLY_TYPED_FUTURE_PERIOD_REVIEW")
    with Session(get_nav_preview_engine()) as session:
        if foreign_typed:
            from app.services.fund_rating_monthly_typed import observed_provider_rows

            provider = observed_provider_rows(code, cutoff.replace(year=cutoff.year - 3), cutoff, session=session)
        else:
            provider = provider_snapshot(code, cutoff, session=session)
    # 取得时刻留在来源回执，不嵌入派生净值内容，避免同一组规范记录产生每日新摘要。
    provider = _input_identity(provider)
    generation_fingerprint = digest(
        {
            "provider": provider,
            "review": _input_identity(review),
            "fund_code": code,
            "as_of_date": str(cutoff),
            "primary_report": bundle.get("original_report_archive"),
        }
    )
    cache_path = Path(output) / "generation-state.json"
    if cache_path.exists():
        cache = read(cache_path)
        prior_result = cache.get("result", {})
        candidate = Path(prior_result.get("candidate_path", "__missing__"))
        import hashlib

        retry_incomplete_source = (
            bool(prior_result.get("attempt_errors"))
            or prior_result.get("status") == "RESEARCH_PENDING"
            or prior_result.get("preview", {}).get("status") == "PARTIAL"
        )
        if (
            not retry_incomplete_source
            and cache.get("input_fingerprint") == generation_fingerprint
            and candidate.is_file()
            and hashlib.sha256(candidate.read_bytes()).hexdigest() == cache.get("candidate_sha256")
        ):
            return {**prior_result, "reused_identical_inputs": True}

    def done(result):
        import hashlib

        if result.get("candidate_path"):
            candidate = Path(result["candidate_path"])
            if (
                result.get("status") == "GENERATED_FOR_COMPILATION"
                and result.get("preview", {}).get("status") == "CANDIDATE"
            ):
                # 只有实际新期原件审阅进入生成器、八维全部重新闭合的新包才能升完整。
                # 直接提交旧partial展示包的compile路径仍不自动升级。
                fresh = json.loads(candidate.read_text(encoding="utf-8"))
                frozen = fresh.get("evidence_v1_calculation", {})
                if frozen.get("partial_release_only") is True:
                    frozen.pop("partial_release_only")
                    fresh["monthly_completion_review"] = {
                        "fund_code": code,
                        "as_of_date": str(cutoff),
                        "input_fingerprint": generation_fingerprint,
                        "scope": "本期实际原件审阅已核，八维重新复算齐备；不是旧部分包自动晋级",
                    }
                    pending = candidate.with_suffix(".pending")
                    pending.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
                    pending.replace(candidate)
            versioned_save(
                cache_path,
                {
                    "input_fingerprint": generation_fingerprint,
                    "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    "result": result,
                },
            )
        return result

    provider_ref = _derived(root, provider, label="monthly-provider-" + code)
    base = {
        "fund_code": code,
        "as_of_date": str(cutoff),
        "provider_ref": provider_ref,
        "nav_points": len(provider["history"].get("dates", provider["history"].get("rows", []))),
    }
    if review is None:
        missing.append({"field": "period_review", "reason": "缺本基金本期原件实际阅读和判断；净值已补，不能自动批准"})
    elif (
        review.get("fund_code") != code
        or review.get("as_of_date") != str(cutoff)
        or review.get("legal_name") != bundle["identity"]["legal_name"]
        or review.get("share_class") != bundle["share_class"]
        or review.get("primary_report_hash") != bundle["original_report_archive"]["sha256"]
    ):
        raise ValueError("MONTHLY_PERIOD_REVIEW_FUND_PERIOD_ORIGINAL_MISMATCH")
    if missing:
        return {**base, "status": "MATERIALS_PENDING", "missing": missing}
    if typed:
        from app.services.fund_rating_monthly_typed import prepare_typed_monthly

        route = review.get("typed_inputs")
        if not route:
            return {
                **base,
                "status": "MATERIALS_PENDING",
                "missing": [
                    {
                        "field": "typed_inputs",
                        "reason": "净值已实际读取；仍须本类型新期原件路径替换、逐项审阅及适用事项，不从旧期批准复制",
                    }
                ],
            }
        replacements = {tuple(item["path"]): item["value"] for item in route.get("replacements", [])}
        profile = typed["profile"]
        if profile in {"MIXED", "CONVERTIBLE", "BOND_INDEX"}:
            replacements[("refs", "provider")] = provider_ref["sha256"]
        result = prepare_typed_monthly(
            code,
            bundle,
            cutoff,
            root,
            output,
            replacements=replacements,
            new_sources=[provider_ref, *route.get("new_sources", [])],
            period_review=route["period_review"],
            management_inputs=route.get("management_inputs"),
            event_audit=route.get("event_audit"),
        )
        return done(
            {
                **base,
                **result,
                "status": "GENERATED_FOR_COMPILATION" if not result.get("failure") else "RESEARCH_PENDING",
                "missing": [{"field": "typed_calculation", "reason": result["failure"]}]
                if result.get("failure")
                else [],
            }
        )
    if bundle.get("r5_calculation") or bundle.get("active_r5_partial_calculation"):
        route = review.get("r5_inputs")
        if not route:
            return {
                **base,
                "status": "MATERIALS_PENDING",
                "missing": [
                    {
                        "field": "r5_inputs",
                        "reason": "净值已实际读取；仍须同类产品、真实成交额和本C合同原件及逐项观察，不能手填新期分数",
                    }
                ],
            }
        if route.get("fund_code") != code or route.get("as_of_date") != str(cutoff):
            raise ValueError("MONTHLY_R5_ORIGINAL_INPUT_FUND_PERIOD_MISMATCH")
        from app.services.fund_rating_monthly_r5 import prepare_monthly_r5

        with Session(get_nav_preview_engine()) as session:
            if route.get("version") == "R5_ACTIVE_MONTHLY_PARTIAL_ORIGINALS_V1":
                from app.services.fund_rating_active_r5_partial import prepare

                dossier = prepare(session, route, root=root)
            elif route.get("version") == "R5_THEMATIC_COMPLETE_ORIGINALS_20261029_1":
                from app.services.fund_rating_thematic_complete import prepare

                dossier = prepare(session, {"fund_code": code, "as_of_date": str(cutoff), "inputs": route}, root=root)
            else:
                dossier = prepare_monthly_r5(session, route, root=root)
        from app.services.fund_rating_reviewed_snapshot import _case, _read

        candidate = {**dossier["entries"][0], "basis": BASIS}
        candidate["reviewed_document"] = _case(
            _read(candidate["research_archive"], root), code, candidate["research_format"]
        )
        destination = Path(output) / "candidate-bundle.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix(".pending")
        pending.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(destination)
        return done(
            {**base, "status": "GENERATED_FOR_COMPILATION", "candidate_path": str(destination.resolve()), "missing": []}
        )
    if bundle["family"] == "INDEX":
        if not review.get("management_review"):
            missing.append({"field": "management_review", "reason": "缺本期实际职责原文审阅，不能沿用旧期批准"})
        result = build_monthly_inputs(
            code, bundle, cutoff, root, output, provider=provider, management_review=review.get("management_review")
        )
        return done(
            {
                **base,
                **result,
                "status": "RESEARCH_PENDING" if missing else "GENERATED_FOR_COMPILATION",
                "missing": missing,
            }
        )
    active = review.get("active_inputs")
    if not active or not review.get("management_review"):
        return {
            **base,
            "status": "MATERIALS_PENDING",
            "missing": [
                {
                    "field": "active_inputs.contract_components",
                    "reason": "须提交本基金合同权重/生效日及各指数真实原件，不借其他基金",
                },
                {"field": "management_review", "reason": "须提交本期实际职责原文和连续责任窗口审阅"},
            ],
        }
    from app.services.fund_rating_monthly_active import build_monthly_inputs as build_active

    result = build_active(
        code,
        bundle,
        cutoff,
        root,
        output,
        provider=provider,
        management_review=review["management_review"],
        contract_review=active,
    )
    return done({**base, **result, "status": "GENERATED_FOR_COMPILATION", "missing": []})


def compile_monthly_bundle(bundle, *, root):
    """所有分数从新期原始输入重算；拒绝直接复制旧研究表作为新期结果。"""
    candidate = deepcopy(bundle)
    if candidate.get("fund_code") == "017437" and candidate.get("fee_calculation"):
        from datetime import UTC, datetime

        from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate
        from app.services.fund_rating_reviewed_snapshot import validate_sources

        validate_sources(candidate, datetime.now(UTC), root=root)
        checked = snapshot_calculate(candidate, root=root)
        if checked["status"] != "PARTIAL" or checked["score"] is not None:
            raise ValueError("MONTHLY_INITIAL_FEE_MUST_REMAIN_PARTIAL")
        return {"basis": BASIS, "entries": [{key: value for key, value in candidate.items()
                                             if key not in {"basis", "reviewed_document"}}]}
    if candidate.get("huaan_qdii_relative_calculation") and candidate.get("huaan_qdii_benchmark_calculation"):
        from app.services.fund_rating_monthly_huaan import compile_monthly_performance

        return compile_monthly_performance(candidate, root=root)
    if candidate.get("evidence_v1_calculation"):
        from app.services.fund_rating_evidence_preview import preview
        from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, PASSIVE_BASIS, policy

        frozen = candidate["evidence_v1_calculation"]
        result = preview(candidate, root, review_required=False)
        if result["status"] not in {"CANDIDATE", "PARTIAL"} or not any(
            value is not None for value in result.get("dimensions", {}).values()
        ):
            raise ValueError(
                "MONTHLY_EVIDENCE_INCOMPLETE:"
                + str(result.get("missing"))
                + ":"
                + str(result.get("facts", {}).get("evidence_errors"))
            )
        weights = policy(ACTIVE_BASIS if candidate["family"] == "ACTIVE_EQUITY" else PASSIVE_BASIS)["weights"]
        if result["status"] == "PARTIAL":
            frozen["partial_release_only"] = True
        dimensions = {
            key: result["dimensions"][value]
            if result["dimensions"][value] is not None
            else {
                "lower": "0",
                "upper": "100",
                "candidate": None,
                "reason": _current_missing_reason(candidate, key, value, result),
            }
            for key, value in KEYS.items()
        }
        candidate["evidence_v1_archive"] = archive_json(root, frozen)
    elif candidate.get("bosera_performance_calculation"):
        from app.services.fund_rating_bosera_qdii_performance import calculate, check_binding
        from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

        raw = candidate["bosera_performance_calculation"]
        check_binding(candidate, raw["inputs"])
        result = calculate(raw["inputs"], root=root)
        if result["score"] is not None or result["grade"] is not None:
            raise ValueError("MONTHLY_BOSERA_MUST_REMAIN_PARTIAL")
        document = deepcopy(candidate["reviewed_document"])
        document.update(code=candidate["fund_code"], as_of=candidate["as_of_date"])
        for key in ("return", "stability"):
            point = result["component_scores"][key]
            reason = (
                f"截至{candidate['as_of_date']}，按本C人民币份额本期官方原件重算一年及三年收益，其他缺项与存续条件继续保留。"
                if key == "return"
                else (
                    f"截至{candidate['as_of_date']}，按本期官方月末原件重算36个月度及25个滚动年度窗口，"
                    "其他缺项与存续条件继续保留。"
                )
            )
            document["dimensions"][key] = {
                "lower": point,
                "upper": point,
                "candidate": point,
                "reason": reason or "由本期实际官方原件重新复算；其他缺项继续保留。",
            }
            _record_missing_explanation(candidate, key, reason)
        for key, dimension in document["dimensions"].items():
            if dimension is None or isinstance(dimension, dict) and dimension.get("candidate") is None:
                reason = _current_missing_reason(candidate, key, KEYS[key], result)
                _record_missing_explanation(candidate, key, reason)
                if isinstance(dimension, dict):
                    dimension["reason"] = reason
        raw["input_archive"] = archive_json(root, raw["inputs"])
        candidate.update(
            reviewed_document=document, research_format="single", research_archive=archive_json(root, document)
        )
        checked = snapshot_calculate(candidate, root=root)
        if checked["status"] != "PARTIAL" or checked["score"] is not None:
            raise ValueError("MONTHLY_BOSERA_CANNOT_CLAIM_COMPLETE")
        return {
            "basis": BASIS,
            "entries": [{k: v for k, v in candidate.items() if k not in {"basis", "reviewed_document"}}],
        }
    elif candidate.get("typed_disclosure_calculation"):
        from app.services.fund_rating_typed_disclosure import calculate

        raw = candidate["typed_disclosure_calculation"]
        raw["input_archive"] = archive_json(root, raw["inputs"])
        result = calculate(candidate, root=root)
        if result["blocked"]:
            raise ValueError("MONTHLY_MATERIAL_EVENT_BLOCKED")
        weights = result["weights"]
        dimensions = {key: result["dimensions"][value] for key, value in KEYS.items()}
    elif candidate.get("active_r5_partial_calculation"):
        from app.services.fund_rating_active_r5_partial import prepare

        raw = candidate["active_r5_partial_calculation"]
        inputs = raw["inputs"]
        if inputs.get("fund_code") != candidate["fund_code"] or inputs.get("as_of_date") != candidate["as_of_date"]:
            raise ValueError("MONTHLY_ACTIVE_R5_PARTIAL_SCOPE")
        with Session(get_nav_preview_engine()) as session:
            return prepare(session, inputs, root=root)
    else:
        if candidate.get("r5_calculation"):
            from app.services.fund_rating_pipeline_prepare import prepare_complete

            raw = candidate["r5_calculation"]
            with Session(get_nav_preview_engine()) as session:
                return prepare_complete(session, raw["packet"], raw["anchors"], root=root)
        raise ValueError("MONTHLY_NEW_PERIOD_COMPILER_REQUIRED")
    for key, internal in KEYS.items():
        if result["dimensions"][internal] is None:
            _record_missing_explanation(candidate, key, _current_missing_reason(candidate, key, internal, result))
        else:
            old = candidate.get("public_explanations", {}).get(key, "")
            supplied = candidate.get("monthly_missing_reasons", {})
            if (
                key in supplied
                or internal in supplied
                or any(word in old for word in ("尚缺", "尚未核", "暂不评分", "未全部核齐"))
            ):
                _record_missing_explanation(
                    candidate,
                    key,
                    f"截至{candidate['as_of_date']}，本项已依据本期实际原件重新复算；历史缺项不再计入当前阻塞，分项结果不代表其余维度已完整。",
                )
                supplied.pop(key, None)
                supplied.pop(internal, None)
    document = {
        "code": candidate["fund_code"],
        "as_of": candidate["as_of_date"],
        "outer_weights_percent": {key: weights[value] for key, value in KEYS.items()},
        "dimensions": dimensions,
    }
    candidate.update(research_format="single", research_archive=archive_json(root, document))
    return {
        "basis": BASIS,
        "entries": [{k: v for k, v in candidate.items() if k not in {"basis", "reviewed_document"}}],
    }


def prepare_monthly(scope, directory):
    """动态读取持仓正式输入，新期候选通过编译才登记，再交正式评分器发布。"""
    at, root = now(), archive_root()
    target = last_completed_month(at.date())
    folder = Path(directory) / "monthly" / str(target)[:7]
    with Session(get_nav_preview_engine()) as session:
        existing = current_inputs(session, list(scope))
    rows = []
    for code in scope:
        previous = existing.get(code)
        bundle = (
            next((m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code), {})
            if previous
            else {}
        )
        row = {
            "fund_code": code,
            "required_month": str(target)[:7],
            "as_of_date": str(previous["as_of_date"]) if previous else None,
        }
        request_path = folder / code / "request.json"
        candidate_path = folder / code / "reviewed-bundle.json"
        partial = _partial_bundle(bundle)
        if previous and _period_covered(bundle, previous["as_of_date"], target) and not partial:
            row.update(status="CURRENT_PERIOD_SAVED")
        else:
            row.update(status="COMPILE_AND_REVIEW_REQUIRED", reviewed_bundle_path=str(candidate_path))
            versioned_save(
                request_path,
                {
                    **row,
                    "checked_at": at.isoformat(),
                    "prior_batch": previous["batch_id"] if previous else None,
                    "prior_calculations": [k for k in bundle if k.endswith("_calculation")],
                    "period_review_path": str(folder / code / "period-review.json"),
                    "reusable_period_review_path": str(
                        Path(root) / "pipeline/monthly-input-reviews" / str(target)[:7] / (code + ".json")
                    ),
                    "required_review_binding": [
                        "fund_code",
                        "as_of_date",
                        "legal_name",
                        "share_class",
                        "primary_report_hash",
                    ],
                    "supported_original_input_routes": [
                        "management_review",
                        "active_inputs",
                        "typed_inputs",
                        "r5_inputs",
                    ],
                    "instructions": (
                        "入口先补本份额真实规范净值；提交period-review.json中的本基金本期实际原件审阅，"
                        "按适用路由填写合同分段/指数、特殊类型或R5原件输入，程序构造候选、重算并登记。"
                        "同月部分结果继续补缺，不以旧总分替代。已有reviewed-bundle.json仍支持严格复算；"
                        "保留实际报告日和取得时刻，不得复制旧分数或只替换日期。"
                    ),
                },
            )
            if not candidate_path.exists() or partial:
                try:
                    review_path = folder / code / "period-review.json"
                    shared_review_path = (
                        Path(root) / "pipeline/monthly-input-reviews" / str(target)[:7] / (code + ".json")
                    )
                    from app.services.fund_exposure_common import read

                    review = (
                        json.loads(review_path.read_text(encoding="utf-8"))
                        if review_path.exists()
                        else read(shared_review_path)
                        if shared_review_path.exists()
                        else None
                    )
                    generated = generate_monthly_inputs(
                        code, bundle, target, root=root, output=folder / code / "preparation", review=review
                    )
                    row["preparation"] = generated
                    versioned_save(folder / code / "materials-pending.json", generated)
                    if generated["status"] == "GENERATED_FOR_COMPILATION":
                        # 同期同原件的实际审阅可跨每日运行续办；新月/原件变更仍重新严格绑定。
                        if review is not None:
                            versioned_save(shared_review_path, review)
                        contents = Path(generated["candidate_path"]).read_text(encoding="utf-8")
                        pending = candidate_path.with_suffix(".pending")
                        pending.write_text(contents, encoding="utf-8")
                        pending.replace(candidate_path)
                    else:
                        row["status"] = generated["status"]
                except Exception as error:
                    row.update(
                        status="MATERIAL_PREPARATION_FAILED", failure=type(error).__name__ + ":" + str(error)[:500]
                    )
            if candidate_path.exists():
                try:
                    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
                    if candidate.get("fund_code") != code or not _period_covered(
                        candidate, date.fromisoformat(candidate["as_of_date"]), target
                    ):
                        raise ValueError("MONTHLY_CANDIDATE_CODE_OR_PERIOD_MISMATCH")
                    dossier = compile_monthly_bundle(candidate, root=root)
                    if _new_complete_prevents_partial(root, code, _dossier_bundle(dossier, root)):
                        row.update(
                            status="NEWER_RESEARCH_PRESERVED",
                            reason="旧候选不取消同月已验收维度；如确有新失效原件须显式重审并保留历史",
                        )
                        rows.append(row)
                        continue
                    from app.services.direction_1d_protocol import digest
                    from app.services.fund_exposure_common import read

                    fingerprint = digest(_registration_identity(dossier))
                    registration_path = folder / code / "registration-state.json"
                    shared_registration_path = (
                        Path(root) / "pipeline/monthly-registration-state" / str(target)[:7] / (code + ".json")
                    )
                    previous_registration = (
                        read(shared_registration_path)
                        if shared_registration_path.exists()
                        else read(registration_path)
                        if registration_path.exists()
                        else {}
                    )
                    if previous_registration.get("dossier_fingerprint") == fingerprint:
                        row.update(
                            status="UNCHANGED_PARTIAL_INPUT" if partial else "UNCHANGED_RESEARCH_INPUT",
                            receipt=previous_registration["receipt"],
                        )
                    else:
                        receipt = register_update(root, dossier)
                        registration = {"dossier_fingerprint": fingerprint, "receipt": receipt}
                        versioned_save(registration_path, registration)
                        versioned_save(shared_registration_path, registration)
                        row.update(status="REGISTERED_FOR_PUBLICATION", receipt=receipt)
                except Exception as error:
                    row.update(status="COMPILATION_FAILED", failure=type(error).__name__ + ":" + str(error)[:500])
        rows.append(row)
    result = {
        "checked_at": at.isoformat(),
        "required_month": str(target)[:7],
        "items": rows,
        "pending": sum(r["status"] != "CURRENT_PERIOD_SAVED" for r in rows),
    }
    versioned_save(folder / "status.json", result)
    return result
