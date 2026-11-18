"""华安月评待办原件续接；保留新同行和真实窗口，不回退旧期骨架。"""

import hashlib
import json
import os
import tempfile
from calendar import monthrange
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import httpx
from bs4 import BeautifulSoup


def _atomic_bytes(path, raw):
    """临时文件明确同目录落D盘；存储错误直接上抛停止本次写入。"""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        stream.write(raw)
        temporary = stream.name
    os.replace(temporary, path)


def deduplicate_fx_sources(source, inputs):
    """仅移除被同值新原件完整覆盖的旧引用，原文注册表与历史仍保留。"""
    refs = list(inputs["fx_refs"])
    tables = {}
    for ref in refs:
        soup = BeautifulSoup(source.raw[ref].decode("utf-8-sig"), "html.parser")
        table = soup.find("table", id="InfoTable")
        if table is None:
            raise ValueError("HUAAN_MONTHLY_FX_ORIGINAL_TABLE")
        rows = [[c.get_text(" ", strip=True) for c in r.find_all(["th", "td"])] for r in table.find_all("tr")]
        if not rows or rows[0][:2] != ["日期", "美元"]:
            raise ValueError("HUAAN_MONTHLY_FX_ORIGINAL_COLUMNS")
        values = {r[0]: r[1] for r in rows[1:]}
        if not values or len(values) != len(rows) - 1:
            raise ValueError("HUAAN_MONTHLY_FX_ORIGINAL_DUPLICATE")
        tables[ref] = values
    for n, ref in enumerate(refs):
        for other in refs[n + 1 :]:
            common = tables[ref].keys() & tables[other].keys()
            if any(tables[ref][day] != tables[other][day] for day in common):
                raise ValueError("HUAAN_MONTHLY_FX_OVERLAPPING_ORIGINAL_CONFLICT")
            if common and not (
                tables[ref].keys() <= tables[other].keys() or tables[other].keys() <= tables[ref].keys()
            ):
                raise ValueError("HUAAN_MONTHLY_FX_PARTIAL_OVERLAP_REQUIRES_NEW_COMPLETE_SOURCE")
    retained = []
    for ref in refs:
        supersets = [other for other in refs if ref != other and tables[ref].keys() <= tables[other].keys()]
        if any(len(tables[other]) > len(tables[ref]) or refs.index(other) > refs.index(ref) for other in supersets):
            continue
        retained.append(ref)
    return retained


def pending_performance_bundle(bundle, *, root):
    """只恢复同一份额同一期的哈希归档；不得将旧期待办充作新月已审阅。"""
    from app.services.fund_rating_huaan_qdii_relative import Originals
    from app.services.fund_rating_pipeline import _load, archive_json

    reference = bundle.get("monthly_performance_followup", {}).get("pending_performance_source_archive")
    if not reference:
        return None
    pending = _load(root, reference)
    if pending.get("fund_code") != bundle.get("fund_code") or pending.get("as_of_date") != bundle.get("as_of_date"):
        raise ValueError("HUAAN_MONTHLY_PENDING_FUND_OR_PERIOD_MISMATCH")
    result = deepcopy(pending)
    for key in ("huaan_qdii_relative_calculation", "huaan_qdii_benchmark_calculation"):
        inputs = result[key]["inputs"]
        if inputs["fund_code"] != bundle["fund_code"] or inputs["as_of_date"] != bundle["as_of_date"]:
            raise ValueError("HUAAN_MONTHLY_PENDING_COMPONENT_IDENTITY")
        source = Originals(inputs, root)
        inputs["fx_refs"] = deduplicate_fx_sources(source, inputs)
        result[key]["input_archive"] = archive_json(root, inputs)
    shared = _shared_inputs(result, root=root)
    for key in ("huaan_qdii_relative_calculation", "huaan_qdii_benchmark_calculation"):
        result[key]["inputs"] = deepcopy(shared)
        result[key]["input_archive"] = archive_json(root, shared)
    return result


def _shared_inputs(bundle, *, root):
    """双组件必须用同一份经济输入；仅合并真实取得时间/同义协议冻结回执。"""
    from app.services.fund_rating_huaan_qdii_relative import Originals

    first = deepcopy(bundle["huaan_qdii_relative_calculation"]["inputs"])
    second = bundle["huaan_qdii_benchmark_calculation"]["inputs"]
    harmless = {"source_registry", "benchmark_protocol_ref", "research_on"}
    if {k: v for k, v in first.items() if k not in harmless} != {k: v for k, v in second.items() if k not in harmless}:
        raise ValueError("HUAAN_MONTHLY_ECONOMIC_INPUTS_DIFFER")
    sources = [Originals(value, root) for value in (first, second)]
    protocols = [
        source.json(value["benchmark_protocol_ref"]) for source, value in zip(sources, (first, second), strict=True)
    ]
    if {k: v for k, v in protocols[0].items() if k != "frozen_at"} != {
        k: v for k, v in protocols[1].items() if k != "frozen_at"
    }:
        raise ValueError("HUAAN_MONTHLY_BENCHMARK_RULES_DIFFER")
    for digest, ref in second["source_registry"].items():
        if digest in first["source_registry"] and first["source_registry"][digest] != ref:
            raise ValueError("HUAAN_MONTHLY_SOURCE_IDENTITY_CONFLICT")
        first["source_registry"][digest] = deepcopy(ref)
    first["research_on"] = max((first["research_on"], second["research_on"]), key=datetime.fromisoformat)
    return first


def _publisher_values(raw, days):
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != 1 or value[0].get("titleText") != "收益率走势":
        raise ValueError("HUAAN_MONTHLY_PUBLISHER_RESPONSE_IDENTITY")
    rows = value[0].get("chartsDatas")
    if not isinstance(rows, list) or not rows:
        raise ValueError("HUAAN_MONTHLY_PUBLISHER_RESPONSE_EMPTY")
    values = {}
    for row in rows:
        day = f"{int(row['year']):04}-{int(row['month']) + 1:02}-{int(row['day']):02}"
        datetime.fromisoformat(day)
        number = Decimal(str(row["value"]))
        if day in values or not number.is_finite():
            raise ValueError("HUAAN_MONTHLY_PUBLISHER_RESPONSE_DUPLICATE_OR_NONFINITE")
        values[day] = number
    missing = sorted(set(days) - values.keys())
    if missing:
        raise ValueError("HUAAN_MONTHLY_PUBLISHER_REQUIRED_DATES_MISSING:" + ",".join(missing))
    return values


def _fetch_publisher(url, params):
    with httpx.Client(
        timeout=httpx.Timeout(18, connect=8),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.huaan.com.cn/"},
    ) as client:
        with client.stream("POST", url, data=params) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > 5_000_000:
                    raise ValueError("HUAAN_MONTHLY_PUBLISHER_RESPONSE_TOO_LARGE")
                chunks.append(chunk)
            return b"".join(chunks), response.status_code


def _attach_month_end_proof(candidate, raw, reference, *, root):
    """评分齐点不等于自然月底已覆盖；只用同一本C跨月原回包生成证明。"""
    from app.services.fund_rating_month_end import official_month_end

    observed = date.fromisoformat(candidate["as_of_date"])
    month_end = observed.replace(day=monthrange(observed.year, observed.month)[1])
    values = _publisher_values(raw, [])
    before = sorted(day for day in values if day <= str(month_end))
    after = sorted(day for day in values if day > str(month_end))
    candidate.pop("monthly_valuation_proof", None)
    if not before or not after or before[-1] != str(observed):
        candidate["monthly_valuation_proof_pending"] = "同一本C官网原回包尚不能同时证明真实月内末点和下次估值日期。"
        return
    proof = {
        "format": "HUAAN_ISSUER_RETURN_CURVE_V1",
        "fund_code": candidate["fund_code"],
        "as_of_date": str(observed),
        "month_end": str(month_end),
        "next_valuation_date": after[0],
        "issuer_nav_source_ref": deepcopy(reference),
        "sequence": "chartsDatas:year,month(zero-based),day; request.params.fundcodes exact C share",
    }
    candidate["monthly_valuation_proof"] = proof
    try:
        official_month_end(candidate, observed, month_end, root=root)
    except (ValueError, KeyError, TypeError) as error:
        candidate.pop("monthly_valuation_proof", None)
        candidate["monthly_valuation_proof_pending"] = str(error)
    else:
        candidate.pop("monthly_valuation_proof_pending", None)


def resume_monthly_performance(bundle, *, root, output, fetch=None):
    """生产续办：正常官网匿名刷新；任一必需点或原件未齐仍保留四维。"""
    from app.services.fund_exposure_common import now
    from app.services.fund_materials_store import versioned_save
    from app.services.fund_rating_huaan_qdii_relative import Originals
    from app.services.fund_rating_pipeline import archive_json
    from app.services.fund_rating_pipeline_prepare import canonical_source

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pending = pending_performance_bundle(bundle, root=root)
    if pending is None:
        raise ValueError("HUAAN_MONTHLY_NO_FROZEN_PENDING_PERFORMANCE")
    inputs = pending["huaan_qdii_relative_calculation"]["inputs"]
    source = Originals(inputs, root)
    days = source.json(inputs["calendar_ref"])["history"]["dates"]
    url = "https://www.huaan.com.cn/fund/FundPerformance/fundPerformanceChart.do"
    params = {
        "selecttype": "ZS",
        "fundcodes": bundle["fund_code"],
        "beginDate": inputs["window_start"],
        "endDate": now().strftime("%Y/%m/%d"),
    }
    receipt = {"url": url, "method": "POST", "params": params, "attempted_at": now().isoformat()}
    try:
        raw, status = (fetch or _fetch_publisher)(url, params)
        receipt.update(
            state="RECEIVED",
            status_code=status,
            received_at=now().isoformat(),
            sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),
        )
        path = output / (receipt["sha256"] + ".json")
        _atomic_bytes(path, raw)
        ref = canonical_source(
            root,
            {
                **receipt,
                "file": str(path.resolve()),
                "scope": "MARKET",
                "format": "JSON",
                "kind": "ORIGINAL",
                "archive_kind": "ORIGINAL",
            },
        )
        receipt["source"] = ref
        if status != 200:
            raise ValueError("HUAAN_MONTHLY_PUBLISHER_HTTP_STATUS")
        _publisher_values(raw, days)
        inputs["source_registry"][ref["sha256"]] = ref
        inputs.update(own_publisher_ref=ref["sha256"], research_on=now().isoformat())
        for key in ("huaan_qdii_relative_calculation", "huaan_qdii_benchmark_calculation"):
            pending[key]["inputs"] = deepcopy(inputs)
            pending[key]["input_archive"] = archive_json(root, inputs)
        candidate = deepcopy(bundle)
        for key in ("huaan_qdii_relative_calculation", "huaan_qdii_benchmark_calculation"):
            candidate[key] = pending[key]
        _attach_month_end_proof(candidate, raw, ref, root=root)
        # 此时仍未批准升完整，严格编译必须独立通过全部组件。
        dossier = compile_monthly_performance(candidate, root=root)
        versioned_save(output / "completed-dossier.json", dossier)
        result = {
            "status": "GENERATED_FOR_COMPILATION",
            "candidate": dossier["entries"][0],
            "request_receipt": receipt,
            "dossier_path": str(output / "completed-dossier.json"),
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        receipt.setdefault("state", "FAILED")
        receipt["failure"] = type(error).__name__ + ":" + str(error)
        result = {
            "status": "MATERIALS_PENDING",
            "candidate": deepcopy(bundle),
            "restored_pending_originals": pending,
            "request_receipt": receipt,
            "missing": [{"field": "huaan_monthly_performance", "reason": receipt["failure"]}],
        }
    candidate_path = output / "resumed-candidate.json"
    _atomic_bytes(candidate_path, json.dumps(result.pop("candidate"), ensure_ascii=False, indent=2).encode("utf-8"))
    result["candidate_path"] = str(candidate_path)
    restored = result.pop("restored_pending_originals", None)
    if restored is not None:
        restored_path = output / "restored-pending-originals.json"
        _atomic_bytes(restored_path, json.dumps(restored, ensure_ascii=False, indent=2).encode("utf-8"))
        result["restored_pending_candidate_path"] = str(restored_path)
    versioned_save(output / "publisher-refresh-receipt.json", receipt)
    versioned_save(output / "resume-result.json", result)
    return result


def compile_monthly_performance(bundle, *, root):
    """两性能、持仓流动性、真实管理费用全部重算后，才生成八维正式候选。"""
    from app.services import fund_rating_huaan_qdii_benchmark as benchmark
    from app.services import fund_rating_huaan_qdii_disclosure as disclosure
    from app.services import fund_rating_huaan_qdii_relative as relative
    from app.services.fund_rating_pipeline import archive_json
    from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy
    from app.services.fund_rating_reviewed_snapshot import BASIS, KEYS
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot

    candidate = deepcopy(bundle)
    shared = _shared_inputs(candidate, root=root)
    for key in ("huaan_qdii_relative_calculation", "huaan_qdii_benchmark_calculation"):
        candidate[key] = {"inputs": deepcopy(shared), "input_archive": archive_json(root, shared)}
    performance = relative.calculate(shared, root=root)
    benchmark_result = benchmark.calculate(shared, root=root)
    inputs = candidate["huaan_qdii_disclosure_calculation"]["inputs"]
    inputs.update(version=disclosure.MONTHLY_COMPLETE_VERSION, monthly_disclosure_only=False)
    disclosure_result = disclosure.calculate(inputs, root=root)
    points = {**performance["dimensions"], **benchmark_result["dimensions"], **disclosure_result["dimensions"]}
    if any(points.get(internal) is None for internal in KEYS.values()):
        raise ValueError("HUAAN_MONTHLY_EIGHT_REAL_COMPONENTS_REQUIRED")
    weights = policy(PASSIVE_BASIS)["weights"]
    document = {
        "code": candidate["fund_code"],
        "as_of": candidate["as_of_date"],
        "outer_weights_percent": {key: weights[internal] for key, internal in KEYS.items()},
        "dimensions": {key: points[internal] for key, internal in KEYS.items()},
    }
    candidate.update(
        monthly_disclosure_only=False,
        research_format="single",
        research_archive=archive_json(root, document),
        reviewed_document=document,
    )
    candidate["huaan_qdii_disclosure_calculation"]["input_archive"] = archive_json(root, inputs)
    candidate["monthly_missing_input_facts"] = {}
    candidate["public_explanations"] = {
        key: f"截至{candidate['as_of_date']}，本项按本份额、本期真实原件及已启用规则独立重算，其他原件与历史记录保留。"
        for key in KEYS
    }
    candidate.setdefault("monthly_performance_followup", {})["status"] = "COMPLETED_EIGHT_COMPONENTS"
    result = snapshot(candidate, root=root)
    if result["status"] != "SCORED" or result["score"] is None:
        raise ValueError("HUAAN_MONTHLY_EIGHT_COMPONENT_SNAPSHOT_NOT_COMPLETE")
    return {"basis": BASIS, "entries": [{k: v for k, v in candidate.items() if k != "reviewed_document"}]}
