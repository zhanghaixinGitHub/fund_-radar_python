"""从已保存原件重建 V2 原始指标；无采集、写库、训练或正式发布行为。

原件存在、哈希正确、成功解析与指标完整是不同状态。旧包只作为原件目录和
可复算事实入口，绝不使用旧版维度分数；每次转换先运行既有原文重建校验。
"""

import hashlib
import json
import re
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, localcontext
from pathlib import Path
from urllib.parse import urlsplit

from app.services.fund_quality_v2_contract import contract_hash, metric_contract
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_metrics import standard_cost, total_return
from app.services.fund_rating_quality_v2 import VERSION, candidate_policy, metric_names
from app.services.fund_rating_rules import decimal

COMPILER = "QUALITY_ORIGINALS_20261017_57"
ROLES = {
    "return": ("history", "benchmark", "contract"),
    "risk": ("history", "benchmark", "contract"),
    "efficiency": ("report", "contract"),
    "stability": ("history", "benchmark", "contract"),
    "management": ("report", "contract", "responsibility_review"),
    "holdings": ("report", "target_or_peer_structure"),
    "liquidity": ("report", "contract", "capacity_review"),
    "cost": ("summary",),
}

# 来源要求按指标列明，不能让同维度的“容量评审”变成持有人表的必要原件。
# 角色只说明档案登记类型；即使全部登记，仍须由对应解析器重读正文证明数值。
METRIC_ROLES = {
    **{
        name: ("history", "benchmark", "contract")
        for name in (
            "relative_return_1y",
            "relative_return_3y",
            "extra_drawdown",
            "extra_downside",
            "rolling_target_success",
            "monthly_target_success",
            "tracking_error",
        )
    },
    "allocation_deviation": ("historical_asset_reports", "historical_effective_contracts"),
    "mandate_deviation": ("historical_asset_reports", "historical_effective_contracts", "mandate_asset_evidence"),
    "governance_state": ("official_governance_events", "governance_scope_review"),
    "key_person_coverage": ("appointment_history", "formal_responsibility"),
    "team_months": ("appointment_history", "current_team_original"),
    "issuer_replication_deviation": ("report", "target_structure", "issuer_mapping"),
    "industry_replication_deviation": ("report", "target_structure", "industry_mapping"),
    "issuer_excess_concentration": ("report", "independent_peer_reference", "issuer_mapping"),
    "industry_excess_concentration": ("report", "independent_peer_reference", "industry_mapping"),
    "restricted_ratio": ("report",),
    "cash_target_deviation": ("report", "cash_target_interval"),
    "holder_pressure_band": ("report",),
    "capacity_pressure": ("report", "capacity_review"),
    "cost_1y": ("summary",),
}


def benchmark_expression(text: str) -> str | None:
    """读取明确加权公式，排除收益比较表、同名叙述及公式后的风险提示。

    原文必须同时给出成分收益率/利率和百分比权重，不能仅因为附近有收益率
    和百分数就认定是基准。重复同一公式可接受，存在不同公式时留待核对版本。
    """
    weight = r"(?:100|[0-9]{1,2})(?:\.[0-9]+)?[%％]"
    name = r"[^+＋。；;:：%％①②③④]{1,90}?(?:收益率|利率)(?:[（(][^（）()。；;%％]{1,24}[）)])?"
    component = rf"(?:{name}[×*]{weight}|{weight}[×*]?{name})"
    formula = re.compile(rf"({component}(?:[+＋]{component})*)")
    candidates = []
    for match in re.finditer("业绩比较基准", text):
        section = text[match.end() : match.end() + 240].lstrip("：:为")
        if section.startswith(("之间", "的", "与", "收益阶段", "收益率①")):
            continue
        if expression := formula.match(section):
            value = expression[1]
            if not any(word in value for word in ("标准差", "净值增长", "收益阶段")):
                candidates.append(value)
    return candidates[0] if candidates and len(set(candidates)) == 1 else None


def inspect_sources(refs: list[dict], root: Path, frozen: datetime) -> list[dict]:
    """逐件核验真实文件、内容摘要、范围与取得时刻；不把登记布尔值当核验结果。"""
    from app.services.fund_quality_v2_publication_sources import publication_proof_refs

    # 日期证明也属于冻结输入；不得仅登记证明标志而不检查原始证明文件。
    refs = [*refs, *publication_proof_refs(refs)]
    result = []
    if len(refs) > 200:
        raise ValueError("QUALITY_V2_SOURCE_LIMIT")
    for ref in refs:
        item = {
            k: ref[k]
            for k in (
                "file",
                "sha256",
                "label",
                "url",
                "acquired_at",
                "published_on",
                "title",
                "report_end",
                "api_name",
                "ts_code",
                "source_code",
                "master_code",
                "source_kind",
                "legal_entity",
                "decision_date",
                "publication_date_status",
                "cninfo_receipt_sha256",
                "contract_notice_receipt_sha256",
                "component_receipt_sha256",
                "component_code",
                "calendar_year",
            )
            if k in ref
        }
        item["state"] = "INVALID_SOURCE"
        try:
            path = (root / ref["file"]).resolve()
            if not path.is_relative_to(root.resolve() / "sources"):
                raise ValueError("SOURCE_PATH")
            if not path.is_file():
                item.update(state="COLLECTION_REQUIRED", reason="ARCHIVE_FILE_MISSING")
                result.append(item)
                continue
            if path.stat().st_size > 20_000_000:
                raise ValueError("SOURCE_SIZE")
            if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                raise ValueError("SOURCE_HASH")
            received = datetime.fromisoformat(ref["acquired_at"])
            if received.tzinfo is None or received > frozen:
                raise ValueError("SOURCE_TIME")
            item.update(state="ORIGINAL_AVAILABLE_PARSE_REQUIRED", size_bytes=path.stat().st_size)
        except PermissionError:
            item.update(state="PERMISSION_DENIED", reason="ARCHIVE_READ_PERMISSION_DENIED")
        except (ValueError, KeyError, OSError) as exc:
            item["reason"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        result.append(item)
    return result


def _legacy_partial_cost(code: str, name: str, refs: list[dict], root: Path, as_of: date, frozen: datetime) -> dict:
    """单独复用费用原件；其他维度缺资料不应抹掉本份额已经能核实的费用。

    此适配不涵盖ETF双层收费及减免，遇到这类原件仍交由专用适配核验。
    不取旧cost_1y或旧分项，费用和份额身份从原文重新解析。
    """
    from app.services.fund_rating_public_market import (
        ATTACHMENT_HOST,
        HOSTS,
        attachment_identity,
        compact,
        key_information,
        public_document,
    )

    by_label = {r.get("label"): r for r in refs}
    summary = by_label.get("summary")
    if summary is None:
        raise ValueError("QUALITY_V2_SUMMARY_COLLECTION_REQUIRED")
    selected = [by_label[k] for k in ("summary", "prospectus", "provider") if k in by_label]
    checked = inspect_sources(selected, root, frozen)
    if any(r["state"] != "ORIGINAL_AVAILABLE_PARSE_REQUIRED" for r in checked):
        raise ValueError("QUALITY_V2_COST_SOURCE_INVALID")
    for ref in selected:
        url = urlsplit(ref["url"])
        if url.scheme != "https" or url.hostname not in HOSTS or url.username or url.password:
            raise ValueError("QUALITY_V2_COST_SOURCE_HOST")
        if url.hostname == ATTACHMENT_HOST:
            attachment_identity(by_label, root, ref["label"], code)

    def document(ref, title):
        return public_document((root / ref["file"]).read_bytes(), ref, code, title)[0]

    pages = document(summary, "产品资料概要")
    summary_text = compact("\n".join(pages))
    title = summary_text.split("编制日期")[0].split("送出日期")[0][:180]
    if "FOF" in title or "基金中基金" in title:
        raise ValueError("QUALITY_V2_FOF_COST_LOOKTHROUGH_REQUIRED")
    prospectus = document(by_label["prospectus"], "招募说明书") if "prospectus" in by_label else None
    names = []
    if "provider" in by_label:
        snapshot = json.loads((root / by_label["provider"]["file"]).read_bytes())
        if snapshot["fund_code"] != code:
            raise ValueError("QUALITY_V2_COST_PROVIDER_IDENTITY")
        names = [
            r["manager_name"]
            for r in snapshot["manager_records"]
            if r["begin_date"] and r["begin_date"] <= str(as_of) and (not r["end_date"] or r["end_date"] > str(as_of))
        ]
    stem = re.sub(r"[-_]?[ACE]$", "", name)
    if "联接" in title:
        # 复用已有联接费用适配，不把正文提及ETF的主动股票基金误判为联接基金。
        from app.services.fund_rating_public_etf import _reprint_key_information

        expression = benchmark_expression(summary_text)
        index = re.search(r"([\u4e00-\u9fffA-Za-z0-9（）()]+指数)收益率", expression or "")
        if index is None or not prospectus:
            raise ValueError("QUALITY_V2_ETF_FEE_WAIVER_AND_TARGET_DOCUMENT_REQUIRED")
        info = _reprint_key_information(pages, prospectus, code, stem, index_name=index[1])
    else:
        info = key_information(pages, code, stem, prospectus, names, profile_only=True)
    if date.fromisoformat(info["published_on"]) > as_of:
        raise ValueError("QUALITY_V2_FUTURE_COST_DOCUMENT")
    return {
        "state": "VERIFIED_METRIC",
        "reason": "FEE_REBUILT_FROM_ORIGINALS_ONLY",
        "value": info["cost_1y"],
        "unit": "ratio",
        "observed_on": info["published_on"],
        "formula_version": contract_hash(),
        "source_hashes": sorted({r["sha256"] for r in selected}),
        "product_document_id": info["product_id"],
    }


def partial_cost(code: str, name: str, refs: list[dict], root: Path, as_of: date, frozen: datetime) -> dict:
    """先沿用已有适配；费用单独核验，避免经理字段不匹配时连费用也无法复用。"""
    if any(r["state"] != "ORIGINAL_AVAILABLE_PARSE_REQUIRED" for r in inspect_sources(refs, root, frozen)):
        raise ValueError("QUALITY_V2_COST_SOURCE_INVALID")
    try:
        return _legacy_partial_cost(code, name, refs, root, as_of, frozen)
    except (ValueError, KeyError):
        from app.services.fund_quality_v2_disclosures import cost_from_sources

        evidence = cost_from_sources(code, refs, root, as_of)
        return {
            **evidence,
            "state": "VERIFIED_METRIC",
            "reason": "FEE_REBUILT_WITHOUT_UNRELATED_MANAGER_FIELDS",
            "unit": "ratio",
            "formula_version": contract_hash(),
        }


def role_sources(refs: list[dict], bundle: dict | None = None) -> dict[str, list[str]]:
    """来源角色来自已保存目录；只有转换成功才宣称对应语义已被重建。"""
    roles = {role: [] for values in [*ROLES.values(), *METRIC_ROLES.values()] for role in values}
    for ref in refs:
        label = ref.get("label", "")
        key = ref.get("sha256")
        targets = []
        if "summary" in label or key == (bundle or {}).get("summary_hash"):
            targets += ["summary", "contract"]
        if "half" in label or label in {"annual", "historical-annual"} or key == (bundle or {}).get("report_hash"):
            targets += ["report", "current_team_original"]
        if label in {"annual", "historical-half", "historical-quarter", "historical-annual"}:
            targets += ["historical_asset_reports"]
        if "prospectus" in label:
            targets += ["contract"]
        if label == "provider" or "nav" in label:
            targets += ["history"]
        if label in {"provider", "manager-api", "manager-master-api"}:
            targets += ["appointment_history"]
        if label == "appointment":
            targets += ["appointment_history", "current_team_original"]
        if label == "governance-regulator":
            targets += ["official_governance_events"]
        if label == "contract-amendment-notice":
            # 角色只表示有这类来源，转换器仍逐条核主体、生效日及版本匹配。
            targets += ["historical_effective_contracts"]
        if "summary" in label:
            targets += ["current_team_original"]
        if "benchmark" in label or label == "link-curve":
            targets += ["benchmark"]
        for target in targets:
            roles[target].append(key)
    return {key: sorted(set(value)) for key, value in roles.items()}


def metric_gaps(kind: str | None, sources: list[dict], bundle: dict | None = None) -> dict:
    """逐指标记录欠缺的来源类别；不支持方法与无资料分别报告。"""
    roles = role_sources(sources, bundle)
    definitions = metric_contract()["metrics"]
    names = metric_names(candidate_policy(kind)) if kind else set(definitions)
    result = {}
    valid = {r["sha256"] for r in sources if r["state"] == "ORIGINAL_AVAILABLE_PARSE_REQUIRED"}
    for name in sorted(names):
        required = list(METRIC_ROLES[name])
        missing = [r for r in required if not set(roles[r]) & valid]
        result[name] = {
            "state": "UNSUPPORTED"
            if kind is None
            else "COLLECTION_REQUIRED"
            if missing
            else "ORIGINAL_AVAILABLE_PARSE_REQUIRED",
            "unit": definitions[name]["unit"],
            "required_roles": required,
            "missing_roles": missing,
            "source_hashes": sorted({h for role in required for h in roles[role]}),
            "reason": "METHOD_NOT_SUPPORTED" if kind is None else "REQUIRES_V2_SEMANTIC_VERIFICATION",
        }
    return result


def relative_history(history: dict, annual_days: int = 252) -> dict[str, Decimal]:
    """完整37个月末/逐估值日两曲线生成原始指标，缺日、错序、零净值均拒绝。

    相对回撤使用 F/B 的路径；下行风险与跟踪误差使用同日简单收益差。
    25个年度窗口相互重叠，输出不把它们当成25个独立年度样本。
    """
    days = list(map(date.fromisoformat, history["dates"]))
    ends = list(map(date.fromisoformat, history["month_ends"]))
    if days != sorted(set(days)) or len(ends) != 37 or ends != sorted(set(ends)):
        raise ValueError("QUALITY_V2_HISTORY_ALIGNMENT")
    months = [v.year * 12 + v.month for v in ends]
    if months != list(range(months[0], months[0] + 37)) or days[0] != ends[0] or days[-1] != ends[-1]:
        raise ValueError("QUALITY_V2_HISTORY_WINDOW")
    if any(d not in days for d in ends) or not 200 <= annual_days <= 366:
        raise ValueError("QUALITY_V2_HISTORY_CALENDAR")
    # adjusted_nav 必须经既有来源适配逐日重建；另一条路径从分红与折算原件重建。
    fund = (
        list(map(decimal, history["adjusted_nav"]))
        if "adjusted_nav" in history
        else total_return(history["unit_nav"], history["cash_dividend_per_share"], history["split_factor"])
    )
    benchmark = list(map(decimal, history["benchmark_index"]))
    if len(days) != len(fund) or len(days) != len(benchmark) or any(v <= 0 for v in fund + benchmark):
        raise ValueError("QUALITY_V2_HISTORY_CURVES")
    if any(max(d for d in days if (d.year, d.month) == (e.year, e.month)) != e for e in ends):
        raise ValueError("QUALITY_V2_HISTORY_MONTH_END")
    with localcontext() as context:
        context.prec = 40
        f = [fund[days.index(d)] for d in ends]
        b = [benchmark[days.index(d)] for d in ends]
        active = [fund[i] / fund[i - 1] - benchmark[i] / benchmark[i - 1] for i in range(1, len(days))]
        relative = [x / y for x, y in zip(fund, benchmark, strict=True)]
        peak, drawdown = relative[0], Decimal(0)
        for value in relative:
            peak = max(peak, value)
            drawdown = max(drawdown, 1 - value / peak)
        mean = sum(active) / len(active)
        exponent = Decimal("365.25") / (days[-1] - days[0]).days

        def success(step):
            returns = [f[i] / f[i - step] - b[i] / b[i - step] for i in range(step, 37)]
            return sum(Decimal(1) if v > 0 else Decimal("0.5") if v == 0 else Decimal(0) for v in returns) / len(
                returns
            )

        return {
            "relative_return_1y": f[-1] / f[-13] - b[-1] / b[-13],
            "relative_return_3y": (f[-1] / f[0]) ** exponent - (b[-1] / b[0]) ** exponent,
            "extra_drawdown": drawdown,
            "extra_downside": (annual_days * sum(min(v, 0) ** 2 for v in active) / len(active)).sqrt(),
            "tracking_error": (annual_days * sum((v - mean) ** 2 for v in active) / (len(active) - 1)).sqrt(),
            "rolling_target_success": success(12),
            "monthly_target_success": success(1),
        }


HISTORICAL_SUPPLEMENT_ROLES = {
    "contract-amendment-notice",
    "contract-notice-download-receipt",
    "cninfo-download-receipt",
    "historical-annual",
    "historical-half",
    "historical-quarter",
    "historical-contract",
    "historical-prospectus",
}
MANAGER_SUPPLEMENT_ROLES = {"manager-api", "manager-master-api"}


def historical_supplements(refs: list[dict] | None, as_of: date) -> list[dict]:
    """验证历史附件的角色和日期；此检查不是原字节或基金身份核验。

    不允许当前报告、概要、净值或API角色混进历史附件覆盖旧包语义。
    原字节随后仍须inspect_sources及逐报告重建；目录标志不能代替原件。
    """
    refs = deepcopy(refs or [])
    if len(refs) > 200:
        raise ValueError("QUALITY_V2_HISTORICAL_SUPPLEMENT_LIMIT")
    for ref in refs:
        if ref.get("label") not in HISTORICAL_SUPPLEMENT_ROLES:
            raise ValueError("QUALITY_V2_HISTORICAL_SUPPLEMENT_ROLE")
        if not ref.get("published_on") or date.fromisoformat(str(ref["published_on"])[:10]) > as_of:
            raise ValueError("QUALITY_V2_HISTORICAL_SUPPLEMENT_PUBLICATION")
        if ref.get("report_end") and date.fromisoformat(ref["report_end"]) > as_of:
            raise ValueError("QUALITY_V2_HISTORICAL_SUPPLEMENT_OBSERVATION")
    return refs


def manager_supplements(
    refs: list[dict] | None, code: str, root: Path, frozen: datetime
) -> tuple[list[dict], list[dict]]:
    """重读独立任职附件，返回原始引用及文件检查结果，旧完整包保持原样。

    仅接纳既有供应商fund_manager的份额/产品任职响应。角色、端点、请求
    身份、真实字节、取得时刻及响应行均重新核验；“核验成功”标志无效。
    产品级任职史还须在披露转换中与正式报告主代码及份额任职记录交叉核对。
    此入口不接纳净值、历史报告、任免简历或治理结论，也不证明职责履行质量。
    """
    from app.services.fund_quality_v2_manager_sources import manager_api_records

    refs = deepcopy(refs or [])
    if not isinstance(refs, list) or any(not isinstance(ref, dict) for ref in refs):
        raise ValueError("QUALITY_V2_MANAGER_SUPPLEMENT_SCHEMA")
    if len(refs) > 12:
        raise ValueError("QUALITY_V2_MANAGER_SUPPLEMENT_LIMIT")
    for ref in refs:
        origin = urlsplit(ref.get("url", ""))
        label, ts_code = ref.get("label"), ref.get("ts_code", "")
        expected = code if label == "manager-api" else ref.get("master_code")
        if (
            label not in MANAGER_SUPPLEMENT_ROLES
            or ref.get("api_name") != "fund_manager"
            or ref.get("source_code") != "TUSHARE_PRO_FUND"
            or origin.scheme != "https"
            or origin.hostname != "api.tushare.pro"
            or origin.port not in {None, 443}
            or origin.path not in {"", "/"}
            or origin.query or origin.fragment or origin.username or origin.password
            or not isinstance(ts_code, str)
            or not re.fullmatch(r"[0-9]{6}\.(?:OF|SZ|SH)", ts_code)
            or ts_code.split(".")[0] != expected
        ):
            raise ValueError("QUALITY_V2_MANAGER_SUPPLEMENT_ROLE_OR_IDENTITY")
    sources = inspect_sources(refs, root, frozen)
    if any(source["state"] != "ORIGINAL_AVAILABLE_PARSE_REQUIRED" for source in sources):
        raise ValueError("QUALITY_V2_MANAGER_SUPPLEMENT_SOURCE")
    for ref in refs:
        manager_api_records(
            json.loads((root / ref["file"]).read_bytes()),
            fund_code=ref["ts_code"].split(".")[0], ts_code=ref["ts_code"],
        )
    return refs, sources


def convert_bundle(
    bundle: dict,
    root: Path,
    frozen: datetime,
    *,
    additional_nav_refs: list[dict] | None = None,
    additional_disclosure_refs: list[dict] | None = None,
    additional_manager_refs: list[dict] | None = None,
    structure_reference: dict | None = None,
) -> dict:
    """重读既有可靠原件，返回部分指标包；资料不足不产生评分或推断缺项为0。

    source_bundle 随包冻结是为了试算时重新生成并比对，不是自证材料。
    不接受只有 benchmark_verified/classification_verified 的旧式手填包。
    """
    from app.services.fund_rating_public_inputs import validate_public_sources, verify_public_originals
    from app.services.fund_rating_public_market import compact, public_document

    as_of = date.fromisoformat(bundle["as_of_date"])
    if frozen.tzinfo is None or as_of > frozen.date():
        raise ValueError("QUALITY_V2_FREEZE_TIME")
    sources = inspect_sources(bundle["raw_archive"], root, frozen)
    if any(r["state"] != "ORIGINAL_AVAILABLE_PARSE_REQUIRED" for r in sources):
        raise ValueError("QUALITY_V2_ORIGINAL_UNAVAILABLE")
    validate_public_sources(bundle, frozen, root=root)
    # 现有原文适配绑定配置目录；禁止传另一个 root 后读取不同的同名原件。
    from app.core.config import get_settings

    if root.resolve() != (Path(get_settings().fund_insight_directory) / "ratings").resolve():
        raise ValueError("QUALITY_V2_CONFIGURED_ROOT_REQUIRED")
    verify_public_originals(bundle)
    original_roles = role_sources(sources, bundle)
    # 新增净值原件作为V2独立附件冻结，不改V1来源包的字段、摘要或旧批次。
    additional_nav_refs = deepcopy(additional_nav_refs or [])
    if any(ref.get("label") != "nav-database-snapshot" for ref in additional_nav_refs):
        raise ValueError("QUALITY_V2_SUPPLEMENTAL_NAV_ROLE")
    additional = inspect_sources(additional_nav_refs, root, frozen)
    if any(ref["state"] != "ORIGINAL_AVAILABLE_PARSE_REQUIRED" for ref in additional):
        raise ValueError("QUALITY_V2_SUPPLEMENTAL_NAV_SOURCE")
    sources += additional
    # 历史附件只供V2补读披露事实；原成对曲线、费用和分类来源继续取旧包。
    additional_disclosure_refs = historical_supplements(additional_disclosure_refs, as_of)
    disclosures = inspect_sources(additional_disclosure_refs, root, frozen)
    if any(ref["state"] != "ORIGINAL_AVAILABLE_PARSE_REQUIRED" for ref in disclosures):
        raise ValueError("QUALITY_V2_HISTORICAL_SUPPLEMENT_SOURCE")
    sources += disclosures
    # 任职原件单独登记，不混入净值或历史披露，也不改写V1来源包。
    additional_manager_refs, manager_sources = manager_supplements(
        additional_manager_refs, bundle["fund_code"], root, frozen
    )
    sources += manager_sources
    # 指数及分类是独立参考原件，不能冒充基金历史报告或改写旧包的来源。
    from app.services.fund_quality_v2_asset_structure import REFERENCE_ROLES

    structure_reference = deepcopy(structure_reference or {})
    if structure_reference and set(structure_reference) != REFERENCE_ROLES:
        raise ValueError("QUALITY_V2_STRUCTURE_REFERENCE_ROLES")
    for role, ref in structure_reference.items():
        if ref.get("label") != "structure-" + role.replace("_", "-"):
            raise ValueError("QUALITY_V2_STRUCTURE_REFERENCE_LABEL")
    structure_sources = inspect_sources(list(structure_reference.values()), root, frozen)
    if any(ref["state"] != "ORIGINAL_AVAILABLE_PARSE_REQUIRED" for ref in structure_sources):
        raise ValueError("QUALITY_V2_STRUCTURE_REFERENCE_SOURCE")
    sources += structure_sources
    refs = {r["sha256"]: r for r in bundle["raw_archive"]}
    summary = refs[bundle["summary_hash"]]
    pages, _ = public_document((root / summary["file"]).read_bytes(), summary, bundle["fund_code"], "产品资料概要")
    text = compact("\n".join(pages))
    # 细分类来自已核验原件标题，不能修改旧包的scope布尔字段后改变新方法分类。
    title = text.split("编制日期")[0].split("送出日期")[0][:180]
    if bundle["currency"] != bundle["key_information"]["currency"] or bundle["currency"] != "CNY":
        raise ValueError("QUALITY_V2_CURRENCY_UNSUPPORTED")
    from app.services.fund_quality_v2_classification import classify_original

    # 封面可能不印QDII，“指数研究增强”也不能落入普通被动指数。
    # 同产品正式报告已由上游重核身份；只读产品说明内的目标与风险范围，
    # 不借网站标签、境外持仓或经理履历推断基金类别。
    report_ref = refs[bundle["report_hash"]]
    report_pages, _ = public_document(
        (root / report_ref["file"]).read_bytes(), report_ref, bundle["fund_code"], "报告"
    )
    classification = classify_original(title, "\n".join(report_pages))
    kind = classification["kind"]
    if not kind:
        raise ValueError("QUALITY_V2_KIND_UNSUPPORTED")
    # 主代码由原文适配重新抽取；不能按名称删除A/C后缀或相信relationship_verified。
    original_product = bundle["key_information"]["product_id"]
    if original_product != bundle["classification"]["product_id"] or not original_product:
        raise ValueError("QUALITY_V2_PRODUCT_CONFLICT")
    master = original_product.rsplit(":", 1)[-1]
    if not re.fullmatch(r"[0-9]{6}", master):
        raise ValueError("QUALITY_V2_PRODUCT_MASTER_UNVERIFIED")
    # 同一主代码的不同既有适配器前缀不应增加独立产品数。
    product = "CN_PUBLIC_FUND:" + master
    contract = benchmark_expression(text)
    packet = {
        "version": VERSION,
        "compiler": COMPILER,
        "contract_hash": contract_hash(),
        "fund_code": bundle["fund_code"],
        "product_id": product,
        "kind": kind,
        "as_of_date": str(as_of),
        "frozen_at": frozen.isoformat(),
        "history_months": 36,
        "source_bundle": bundle,
        "additional_nav_refs": additional_nav_refs,
        "additional_disclosure_refs": additional_disclosure_refs,
        "additional_manager_refs": additional_manager_refs,
        "structure_reference": structure_reference,
        "classification_evidence": {"product_id": product, "source_hashes": bundle["classification"]["source_refs"]},
        "benchmark_evidence": {
            "expression": contract,
            "source_hashes": [bundle["summary_hash"]],
        },
        "sources": {r["sha256"]: r for r in sources},
        "metrics": {},
        "inventory": metric_gaps(kind, sources, bundle),
    }
    required = metric_names(candidate_policy(kind))

    def record(name, value, observed, hashes):
        if name not in required:
            return
        definition = metric_contract()["metrics"][name]
        evidence = {
            "value": str(value),
            "unit": definition["unit"],
            "observed_on": observed,
            "source_hashes": sorted(set(hashes)),
            "formula_version": contract_hash(),
        }
        packet["metrics"][name] = evidence
        packet["inventory"][name].update(state="VERIFIED_METRIC", reason="REBUILT_FROM_ORIGINALS", **evidence)

    history = bundle["history"]
    # 成对历史指标仍取旧包原件重建的同一条曲线，不把新增净值快照冒充该曲线来源。
    roles = original_roles
    if history.get("benchmark_index") and contract:
        # 仅国内既有适配确认252；QDII完整估值年化口径未核定时保持待解析。
        if kind != "GLOBAL_ACTIVE_EQUITY":
            for name, value in relative_history(history).items():
                record(name, value, history["dates"][-1], roles["history"] + roles["benchmark"] + roles["contract"])
    info = bundle["key_information"]
    fees = info["fees"]
    cost = standard_cost(
        *(fees.get(k, 0) for k in ("management", "custody", "sales")),
        fees["subscription"],
        fees["redemption"],
        mode=fees["subscription_mode"],
        operating_cost=fees["operating_cost"],
    )
    if cost != decimal(info["cost_1y"]):
        raise ValueError("QUALITY_V2_COST_CONFLICT")
    record("cost_1y", cost, info["published_on"], [bundle["summary_hash"], *roles["contract"]])
    manager = bundle.get("manager")
    if manager and manager.get("teams"):
        teams = manager["teams"]
        # 时间线已由原文重建；此处再约束当前端点，未来任免不能提前进入。
        if teams != sorted(teams, key=lambda r: r["effective_date"]) or any(
            t["effective_date"] > str(as_of) for t in teams
        ):
            raise ValueError("QUALITY_V2_TEAM_TIME")
        months = min(
            Decimal(36), Decimal((as_of - date.fromisoformat(teams[-1]["effective_date"])).days) / Decimal("30.4375")
        )
        record("team_months", months, str(as_of), roles["history"] + roles["contract"])
    # 补读完整原文：有全资产范围声明才提升受限口径；旧版风险档、分项分数均不复用。
    from app.services.fund_quality_v2_disclosures import extract_disclosures

    original_refs = (
        deepcopy(bundle["raw_archive"]) + additional_nav_refs + additional_disclosure_refs + additional_manager_refs
    )
    for ref in original_refs:
        if ref["sha256"] == bundle["report_hash"]:
            ref.setdefault("label", "half")
        if ref["sha256"] == bundle["summary_hash"]:
            ref.setdefault("label", "summary")
        if "招募" in ref.get("url", ""):
            ref.setdefault("label", "prospectus")
    extracted = extract_disclosures(bundle["fund_code"], original_refs, root, as_of, bundle=bundle, frozen=frozen)
    apply_extraction(packet, extracted)
    if structure_reference and kind == "PASSIVE_INDEX":
        from app.services.fund_quality_v2_asset_structure import rebuild_industry_structure
        from app.services.fund_quality_v2_disclosures import report_period

        try:
            report_ref = refs[bundle["report_hash"]]
            report_pages, _ = public_document(
                (root / report_ref["file"]).read_bytes(), report_ref, bundle["fund_code"], ""
            )
            observed = str(report_period("\n".join(report_pages))[1])
            if observed > str(as_of) or (as_of - date.fromisoformat(observed)).days > 210:
                raise ValueError("QUALITY_V2_STRUCTURE_REPORT_STALE_OR_FUTURE")
            structure = rebuild_industry_structure(
                report_pages,
                benchmark=contract,
                reference=structure_reference,
                raw={role: (root / ref["file"]).read_bytes() for role, ref in structure_reference.items()},
                observed_on=observed,
                as_of=str(as_of),
            )
            packet["parsed_facts"]["full_asset_industry_structure"] = structure
            record(
                "industry_replication_deviation",
                structure["value"],
                observed,
                [bundle["report_hash"], bundle["summary_hash"], *[r["sha256"] for r in structure_reference.values()]],
            )
            packet["inventory"]["issuer_replication_deviation"].update(
                state="REFERENCE_DATA_REQUIRED",
                reason="INDEX_WEIGHTS_PRESENT_FULL_ISSUER_IDENTITY_MAPPING_REQUIRED",
                missing_roles=["issuer_mapping"],
                verified_reference_roles=["target_structure", "industry_mapping"],
                required_data="同日指数成分和行业原件已核实；仍缺股票/债券跨证券发行人身份及非证券资产归属规则，证券代码不能代替发行人",
            )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            packet["inventory"]["industry_replication_deviation"].update(
                state="ORIGINAL_AVAILABLE_PARSE_REQUIRED",
                reason="FULL_ASSET_INDUSTRY_REBUILD_FAILED:" + str(exc),
                source_hashes=sorted(ref["sha256"] for ref in structure_reference.values()),
            )
    packet["raw_metrics_complete"] = required <= packet["metrics"].keys()
    clear_verified_role_gaps(packet)
    packet["hash"] = digest(packet)
    return packet


def clear_verified_role_gaps(packet: dict) -> None:
    """数值已经由原件重建时，移除最初角色模板留下的假缺口。

    对未核准字段继续保留登记缺口；清理不会使任何字段变成已核准，也不补值。
    """
    for name in packet["metrics"]:
        packet["inventory"][name].pop("missing_roles", None)
        packet["inventory"][name].pop("required_data", None)


def apply_extraction(packet: dict, extracted: dict) -> None:
    """原件新事实进入对应唯一指标；缺口必须带具体所缺内容，不能把未知补成0。"""
    from app.services.fund_quality_v2_contract import metric_contract

    packet["parsed_facts"] = extracted["facts"]
    packet["document_errors"] = extracted["document_errors"]
    nav = extracted["facts"].get("own_nav_coverage", {})
    late = nav.get("late_announcements", [])
    unknown_dates = nav.get("unknown_announcement_dates", [])
    unavailable = bool(late or unknown_dates)
    nav_metrics = {"relative_return_1y", "relative_return_3y", "extra_drawdown", "extra_downside",
                   "rolling_target_success", "monthly_target_success", "tracking_error"}
    if unavailable:
        # 旧完整包已经重建出的数值也须遵守V2信息截止日。保留原件与原批次，
        # 只撤销本次候选包内不具备时间资格的字段，不能凭旧路径绕过新核验。
        for name in nav_metrics:
            packet["metrics"].pop(name, None)
    for name, evidence in extracted["metrics"].items():
        if unavailable and name in nav_metrics:
            continue
        if name not in packet["inventory"] or name in packet["metrics"]:
            continue
        value = {**evidence, "unit": metric_contract()["metrics"][name]["unit"], "formula_version": contract_hash()}
        packet["metrics"][name] = value
        packet["inventory"][name].update(state="VERIFIED_METRIC", reason="REBUILT_FROM_ORIGINALS", **value)
    for name, finding in extracted["findings"].items():
        if name in packet["inventory"] and name not in packet["metrics"]:
            packet["inventory"][name].update(finding)
    for name in packet["inventory"]:
        if unavailable and name in nav_metrics:
            for stale in ("value", "observed_on", "formula_version", "publisher_cross_check",
                          "return_source_basis", "missing_roles"):
                packet["inventory"][name].pop(stale, None)
            packet["inventory"][name].update(
                state="SOURCE_NOT_AVAILABLE_AS_OF" if late else "SOURCE_PUBLICATION_UNVERIFIED",
                reason="NAV_ANNOUNCEMENT_AFTER_AS_OF" if late else "NAV_ANNOUNCEMENT_DATE_REQUIRED",
                required_data={"availability_cutoff": packet["as_of_date"],
                               "late_announcements": late, "unknown_announcement_dates": unknown_dates,
                               "required": "计算所用逐估值日净值的登记公布日不得晚于考察日；缺少公布日不推定已知。"
                               "不回写旧日期，不删末期净值缩短窗口；需按真实可得日另建包或补充原始公布证据。"},
                source_hashes=nav.get("source_hashes", []),
            )
            continue
        if name in packet["metrics"] or name in extracted["findings"]:
            continue
        if name.startswith(("issuer_", "industry_")):
            passive = packet.get("kind") == "PASSIVE_INDEX"
            packet["inventory"][name].update(
                state="REFERENCE_DATA_REQUIRED",
                reason="INDEX_CONSTITUENT_WEIGHTS_AND_ISSUER_TAXONOMY_REQUIRED"
                if passive
                else "INDEPENDENT_STYLE_PEER_STRUCTURE_REFERENCE_NOT_FROZEN",
                required_data="同日目标指数完整成分权重、发行人及行业映射；目标ETF实际持仓不能冒充指数目标权重"
                if passive
                else "预先冻结的同目标独立产品持仓参照、同报告日和同分类版本；不从本次对象反推参照",
            )
        elif name in {
            "relative_return_1y",
            "relative_return_3y",
            "extra_drawdown",
            "extra_downside",
            "rolling_target_success",
            "monthly_target_success",
            "tracking_error",
        }:
            nav = extracted["facts"].get("own_nav_coverage", {})
            nav_complete = nav.get("status") == "OWN_ADJUSTED_NAV_WINDOW_REBUILT"
            paired_error = extracted["facts"].get("paired_contract_history", {}).get("error")
            packet["inventory"][name].update(
                state="SOURCE_VALUE_CONFLICT" if paired_error and "NAV_CONFLICT" in paired_error
                else "SOURCE_WINDOW_INCOMPLETE" if paired_error and "DATES_MISSING" in paired_error
                else "SOURCE_VALUE_UNRESOLVED" if paired_error else "COLLECTION_REQUIRED",
                reason=paired_error or ("CONTRACT_BENCHMARK_HISTORY_REQUIRED_OWN_NAV_REBUILT"
                if nav_complete
                else "ALIGNED_CONTRACT_BENCHMARK_AND_FUND_HISTORY_REQUIRED"),
                required_data={
                    "own_nav_status": nav.get("status", nav.get("error", "OWN_NAV_ORIGINAL_NOT_REGISTERED")),
                    "own_nav_missing_dates": nav.get("missing_adjusted_nav_dates"),
                    "paired_source_error": paired_error,
                    "required": ("两条基金收益曲线的原始分红/复权口径、冲突日期与供应商修订核验；"
                                 "不能放宽容差或宣称未采集")
                    if paired_error and "NAV_CONFLICT" in paired_error
                    else "完整37个月末、逐估值日合同基准；本份额净值已重建，不能重复报告为未采集"
                    if nav_complete
                    else "本份额再投资净值与合同基准完整37个月末及逐估值日；跨境须另核真实估值日历",
                },
            )
        elif name == "key_person_coverage":
            packet["inventory"][name].update(
                state="COLLECTION_REQUIRED",
                reason="FORMAL_RESPONSIBILITY_AND_COMPLETE_APPOINTMENT_HISTORY_REQUIRED",
                required_data="正式职责原文、正式任免/代理记录及三年连续覆盖",
            )


def convert_archived(code: str, name: str, refs: list[dict], root: Path, as_of: date, frozen: datetime) -> dict:
    """无需旧八维准入即可转换原件中独立可核实的字段，按新编译器隔离留档。"""
    from app.services.fund_quality_v2_disclosures import extract_disclosures

    sources = inspect_sources(refs, root, frozen)
    if any(r["state"] != "ORIGINAL_AVAILABLE_PARSE_REQUIRED" for r in sources):
        raise ValueError("QUALITY_V2_ARCHIVED_SOURCE_INVALID")
    extracted = extract_disclosures(code, refs, root, as_of, frozen=frozen)
    kind = extracted["facts"].get("classification", {}).get("kind")
    product = extracted["facts"].get("product_identity", {}).get("product_id")
    packet = {
        "version": VERSION,
        "compiler": COMPILER,
        "contract_hash": contract_hash(),
        "fund_code": code,
        "product_id": product,
        "kind": kind,
        "as_of_date": str(as_of),
        "frozen_at": frozen.isoformat(),
        "history_months": 36
        if extracted["facts"].get("paired_contract_history", {}).get("history")
        else 0,
        "source_bundle": {
            "origin": "ARCHIVED_ORIGINALS_V2",
            "fund_code": code,
            "fund_name": name,
            "as_of_date": str(as_of),
            "raw_archive": refs,
        },
        "sources": {r["sha256"]: r for r in sources},
        "classification_evidence": extracted["facts"].get("classification", {}) | {"product_id": product},
        "benchmark_evidence": extracted["facts"].get("contract_benchmark_expression", {}),
        "fund_nav_history_months": 36 if extracted["facts"].get("own_nav_coverage", {}).get("curve") else 0,
        "history_coverage_note": "history_months要求基金/合同基准成对完整；fund_nav_history_months仅说明本份额净值",
        "metrics": {},
        "inventory": metric_gaps(kind, sources),
    }
    try:
        evidence = partial_cost(code, name, refs, root, as_of, frozen)
        packet["metrics"]["cost_1y"] = {k: v for k, v in evidence.items() if k not in {"state", "reason"}}
        packet["inventory"]["cost_1y"].update(evidence)
    except (ValueError, KeyError, OSError) as exc:
        packet["inventory"]["cost_1y"].update(
            state="SOURCE_VALUE_UNRESOLVED",
            reason=str(exc),
            required_data="本份额原文标准费率、365天交易费区间和适用减免；原件存在但版式/身份必须真实核对",
        )
    apply_extraction(packet, extracted)
    packet["raw_metrics_complete"] = bool(
        kind and product and metric_names(candidate_policy(kind)) <= packet["metrics"].keys()
    )
    clear_verified_role_gaps(packet)
    packet["hash"] = digest(packet)
    return packet


def verify_packet(packet: dict, root: Path) -> None:
    """按当前原件重新转换再逐字段比对；伪造布尔核验、指标值或来源摘要均不能通过。"""
    value = {k: v for k, v in packet.items() if k != "hash"}
    if packet.get("hash") != digest(value) or packet.get("compiler") != COMPILER:
        raise ValueError("QUALITY_V2_PACKET_HASH_OR_COMPILER")
    if packet["source_bundle"].get("origin") == "ARCHIVED_ORIGINALS_V2":
        b = packet["source_bundle"]
        actual = convert_archived(
            b["fund_code"],
            b["fund_name"],
            b["raw_archive"],
            root,
            date.fromisoformat(b["as_of_date"]),
            datetime.fromisoformat(packet["frozen_at"]),
        )
    else:
        actual = convert_bundle(
            packet["source_bundle"],
            root,
            datetime.fromisoformat(packet["frozen_at"]),
            additional_nav_refs=packet.get("additional_nav_refs", []),
            additional_disclosure_refs=packet.get("additional_disclosure_refs", []),
            additional_manager_refs=packet.get("additional_manager_refs", []),
            structure_reference=packet.get("structure_reference", {}),
        )
    if actual != packet:
        raise ValueError("QUALITY_V2_PACKET_ORIGINAL_CONFLICT")


def build_cohort(packets: list[dict], assignments: dict[str, str], root: Path) -> dict:
    """按原件产品主键冻结互斥清单；只建立资料集，不生成锚点或独立验收结论。

    assignments 的键必须是已重建产品ID，值为 calibration 或 validation。
    同产品所有份额、所有月份属于同一侧；每产品每月仅取代码最小的完整份额。
    """
    if set(assignments.values()) - {"calibration", "validation"}:
        raise ValueError("QUALITY_V2_COHORT_PARTITION")
    members, gaps, seen, products = [], [], set(), set()
    for packet in sorted(packets, key=lambda p: (p["as_of_date"], p["fund_code"])):
        verify_packet(packet, root)
        product = packet["product_id"]
        if not product:
            gaps.append({"fund_code": packet["fund_code"], "reason": "PRODUCT_IDENTITY_OR_METHOD_UNVERIFIED"})
            continue
        products.add(product)
        if product not in assignments:
            gaps.append({"product_id": product, "reason": "PRODUCT_PARTITION_REQUIRED"})
            continue
        if not packet["raw_metrics_complete"]:
            gaps.append({"product_id": product, "fund_code": packet["fund_code"], "reason": "RAW_METRICS_INCOMPLETE"})
            continue
        key = product, packet["as_of_date"]
        if key in seen:
            continue
        seen.add(key)
        members.append(
            {k: packet[k] for k in ("fund_code", "product_id", "kind", "as_of_date", "hash")}
            | {"partition": assignments[product]}
        )
    if set(assignments) - products:
        raise ValueError("QUALITY_V2_COHORT_UNKNOWN_PRODUCT")
    return {
        "version": VERSION,
        "status": "DATASET_ONLY_NOT_CALIBRATION",
        "contract_hash": contract_hash(),
        "members": members,
        "gaps": gaps,
        "note": "30产品仅候选门槛；资料集不代表锚点、统计充分或独立验收。",
    }
