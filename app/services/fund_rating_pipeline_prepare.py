"""将已有真实原始观察包转为标准R5更新清单；不接收总分作为原始指标。"""

# 原始响应统一归档外接盘；历史路径只作为数据库别名，不生成C盘源文件。
import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from app.services import material_content as _sources
from app.services.fund_rating_pipeline import archive_json, current_inputs
from app.services.fund_rating_r5 import calculate
from app.services.fund_rating_reviewed_snapshot import BASIS, KEYS
from app.services.material_content import (
    content_digest,
    location_exists,
    same_content,
    source_bytes,
    source_exists,
    source_size,
    source_text,
)


def canonical_source(root: Path, reference: dict) -> dict:
    """核对已有字节后归档；只允许项目既有来源，路径和摘要不匹配即拒绝。"""
    root = root.resolve()
    source = Path(reference.get("path") or reference["file"])
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    project = Path(__file__).resolve().parents[2]
    if not source.is_relative_to(project) or not source_exists(source) or source_size(source) > 50_000_000:
        raise ValueError("RATING_R5_IMPORT_SOURCE_SCOPE")
    content = source_bytes(source)
    digest = content_digest(content)
    if digest != reference["sha256"]:
        raise ValueError("RATING_R5_IMPORT_SOURCE_HASH")
    relative = "sources/pipeline/" + digest + source.suffix.lower()
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if location_exists(destination):
        if not same_content(source_bytes(destination), content):
            raise ValueError("RATING_R5_CANONICAL_SOURCE_CHANGED")
    else:
        from app.services.material_content import register_content

        if not register_content(destination, content):
            _sources.write_source(destination, content)
    # 旧响应未带上游取得时刻时，明确记录本次本地原件复用的真实时刻；
    # 不把转存说成当年的远程下载。复用首次回执，重复导入保持幂等。
    receipt_path = root / "pipeline" / "imports" / (digest + ".json")
    if location_exists(receipt_path):
        receipt = json.loads(source_text(receipt_path, encoding="utf-8"))
    else:
        receipt = {"sha256": digest, "imported_at": datetime.now(UTC).isoformat()}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        from app.services.material_content import write_source

        write_source(receipt_path, json.dumps(receipt).encode("utf-8"))
    return {**reference, "file": relative, "path": relative, "imported_at": receipt["imported_at"]}


def prepare_management(session, request: dict, *, root: Path) -> dict:
    """把本份额同一考察期的真实管理观察编译为更新清单，不延长资料有效期。

    request包含fund_code、as_of_date、inputs及业务explanation；inputs是已按
    摘要归档的原文、日期和责任区间。分值由纯计算器产生，拒绝直接传入分数。
    新月份须先建立该月基础资料包，不能复制旧报告并替换日期。
    """
    from app.services.fund_rating_management_inputs import calculate as calculate_management
    from app.services.fund_rating_reviewed_snapshot import calculate as calculate_snapshot
    from app.services.fund_rating_reviewed_snapshot import validate_sources

    code = request["fund_code"]
    if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("RATING_MANAGEMENT_CODE")
    if len(request.get("explanation", "").strip()) < 40:
        raise ValueError("RATING_MANAGEMENT_REVIEW_EXPLANATION_REQUIRED")
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("RATING_MANAGEMENT_CURRENT_PERIOD_INPUT_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    if bundle.get("basis") != BASIS or bundle.get("r5_calculation"):
        # 完整原始包应重编全包，不能只替换分项而留下相互矛盾的八维原始观察。
        raise ValueError("RATING_MANAGEMENT_PARTIAL_SNAPSHOT_REQUIRED")
    raw = deepcopy(request["inputs"])
    value = calculate_management(
        raw,
        root=root,
        manager_name=bundle["identity"]["manager_name"],
        as_of=request["as_of_date"],
        fund_code=code,
        legal_name=bundle["identity"]["legal_name"],
        primary_report_hash=bundle["original_report_archive"]["sha256"],
    )
    document = bundle["reviewed_document"]
    document["dimensions"]["management"] = {
        "lower": value["lower"],
        "upper": value["upper"],
        "candidate": value["score"],
        "reason": request["explanation"].strip(),
    }
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        management_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    bundle.setdefault("public_explanations", {})["management"] = request["explanation"].strip()
    validate_sources(bundle, datetime.now(UTC), root=root)
    calculate_snapshot(bundle, root=root)
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}


def prepare_complete(session, packet: dict, anchors: dict, *, root: Path) -> dict:
    """重读本基金原件、核准法律身份、复算八维并生成可幂等登记的单基金清单。

    参数packet必须包含逐子项原始观测、来源、显式单位和封存核验时间；函数
    不延长有效期。anchors只取本基金目标及已核适用事项，不借用目标ETF合同。
    """
    packet = deepcopy(packet)
    code = packet["fund_code"]
    if not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("RATING_R5_FUND_CODE")
    master = (
        session.execute(
            text("""
        SELECT s.fund_name AS public_name,s.share_class,s.fund_master_id,
               m.fund_name AS catalog_legal_name,m.manager_name
        FROM fund_share_class s JOIN fund_master m USING(fund_master_id)
        WHERE s.fund_code=:code AND s.status='ACTIVE'
    """),
            {"code": code},
        )
        .mappings()
        .one()
    )
    originals = {}
    for ref in packet["raw_archive"]:
        canonical = canonical_source(root, ref)
        canonical.pop("path", None)
        if not canonical.get("acquired_at"):
            canonical.update(
                acquired_at=canonical["imported_at"], acquisition_scope="LOCAL_ARCHIVE_REUSE", upstream_acquired_at=None
            )
        originals[canonical["sha256"]] = canonical
    for digest, ref in packet["r4_evidence_registry"].items():
        canonical = canonical_source(root, ref)
        packet["r4_evidence_registry"][digest] = canonical
        if digest not in originals:
            originals[digest] = {
                **canonical,
                "kind": "ORIGINAL",
                "acquired_at": canonical["imported_at"],
                "acquisition_scope": "LOCAL_ARCHIVE_REUSE",
                "upstream_acquired_at": None,
                "title": "已核公开基金公告原始响应",
            }
    packet["raw_archive"] = list(originals.values())
    own_anchors = {"material_events": {code: anchors.get("material_events", {}).get(code, [])}}
    if code in anchors:
        own_anchors[code] = canonical_source(root, anchors[code])
    # 所有观察仍绑定原始取得日；转存时间不是重新取得或重新核验的日期。
    from app.services.fund_rating_reviewed_quality import validate_sources

    validate_sources(packet, datetime.now(UTC), root=root)
    computed = calculate(packet, own_anchors, root=root)
    # 财报来源由本包真实支付观察绑定，不按基金代码、当前年份或目录顺序猜测。
    primary_hashes = set(packet["quality_metrics"]["observations"]["known_payment"]["source_refs"])
    reports = [
        r
        for r in packet["raw_archive"]
        if r["sha256"] in primary_hashes
        and r["file"].endswith((".pdf", ".json"))
        and re.search("中期报告|年度报告|半年度报告", r.get("title", ""))
    ]
    from app.services.fund_rating_pdf import text_pages

    identity = None
    for candidate in reports:
        if candidate["file"].endswith(".pdf"):
            pages = text_pages(root / candidate["file"], limit=12)
        else:
            from bs4 import BeautifulSoup

            original = json.loads(source_text(root / candidate["file"], encoding="utf-8"))
            content = original.get("data", {}).get("notice_content")
            if not isinstance(content, str) or len(content) < 15000:
                # 元数据、摘要或只有附件URL不能冒充完整原报告。
                continue
            pages = [BeautifulSoup(content, "html.parser").get_text("\n", strip=True)[:20000]]
        plain = re.sub(r"\s+", "", "".join(pages))
        if code in plain and re.sub(r"\s+", "", master["manager_name"]) in plain:
            report = candidate
            # 发行人标题可能使用“二0二六年”，不得把年份混入法律产品名。
            legal_name = re.sub(r"(?:20\d{2}|[〇零一二三四五六七八九0-9]{4})年.*$", "", report["title"])
            if re.sub(r"\s+", "", legal_name) not in plain:
                continue
            identity = {
                "code": code,
                "identity_verified": True,
                "published_on": report["published_on"],
                "title": report["title"],
                "pages": pages,
                "source": {"sha256": report["sha256"]},
            }
            break
    if identity is None:
        raise ValueError("RATING_R5_REPORT_LEGAL_IDENTITY_NOT_VERIFIED")
    doc = {
        "code": code,
        "as_of": packet["as_of_date"],
        "outer_weights_percent": {k: computed["weights"][v] for k, v in KEYS.items()},
        "dimensions": {k: computed["dimensions"][v] for k, v in KEYS.items()},
        "calculation_revision": computed["calculation_revision"],
        "full_total": computed["score"],
        "independent_product_validation": False,
        "grade": None,
    }
    source = {"title": report["title"], "url": report["url"], "publishedOn": report["published_on"]}
    dates = packet["quality_public"]["dataDates"]
    explanations = {
        "return": "按本份额实际费后回报与适用参照比较收益差距；历史表现不代表未来收益。",
        "risk": "结合相对下行表现和实际风险暴露评价；较高收益不能抵消已发生的回撤。",
        "execution": "核对实际投资执行和适用合同；被动基金结合本份额真实跟踪误差评价。",
        "stability": "观察完整月度与滚动年度中有意义领先的持续性；滚动窗口相互重叠。",
        "management": "使用已核的投研职责、责任接续和治理事项；不由年限或无检索命中推定优秀。",
        "holdings": "按产品目标评价公开持仓结构；被动产品评价复制匹配，主动产品保留实际集中暴露。",
        "liquidity": "核对成交承接、受限资产、用途匹配支付及实际申赎履职；报告日覆盖不保证未来赎回安全。",
        "fee": "以一万元持有365天的本份额公开费用评价；未公开交易成本不冒充零费用。",
    }
    entry = {
        "fund_code": code,
        "family": packet["family"],
        "public_name": master["public_name"],
        "share_class": master["share_class"],
        "as_of_date": packet["as_of_date"],
        "holding_date": dates["holdings"],
        "verified_at": packet["verified_at"],
        "identity": {"legal_name": legal_name, "manager_name": master["manager_name"]},
        "existing_catalog_binding": {
            "fund_master_id": str(master["fund_master_id"]),
            "catalog_legal_name": master["catalog_legal_name"],
        },
        "research_format": "single",
        "research_archive": archive_json(root, doc),
        "identity_archive": archive_json(root, identity),
        "original_report_archive": {
            "file": report["file"],
            "sha256": report["sha256"],
            "kind": "pdf" if report["file"].endswith(".pdf") else "json",
        },
        "report_title": report["title"],
        "report_url": report["url"],
        "report_published_on": report["published_on"],
        "extra_archives": [],
        "public_explanations": explanations,
        "dimension_sources": {k: [source] for k in KEYS},
        "data_dates": dates,
        "comparison_product_count": packet["quality_public"]["comparison"]["productCount"],
        "public_limitations": [
            "完整分评价已披露历史信息；等级尚未确定，未声称完成全类型独立验证。",
            "持仓为定期披露快照；历史收益和报告日账面覆盖均不保证未来表现或赎回安全。",
        ],
        "r5_calculation": {
            "packet": packet,
            "anchors": own_anchors,
            "input_archive": archive_json(root, packet),
            "anchors_archive": archive_json(root, own_anchors),
        },
    }
    previous = current_inputs(session, [code]).get(code)
    replacements = {}
    if previous and previous["input_snapshot"]["method"].get("basis") != BASIS:
        replacements[code] = [previous["batch_id"]]
    return {"basis": BASIS, "entries": [entry], "replace_current_batches": replacements}


def prepare_gold_components(session, request: dict, *, root: Path) -> dict:
    """从已核黄金联接原件独立合并收益、稳定性，保留其余缺项及全部旧历史。"""
    from app.services.fund_rating_gold_components import prepare

    return prepare(session, request, root=root)


def prepare_direct_index(session, packet: dict, anchors: dict, *, root: Path) -> dict:
    """从直接指数/ETF联接原件编译观察并进入同一完整发布校验；不延长日期。

    packet的direct_index_compilation或passive_index_compilation包含真实原件
    及计算前固定的参照清单，两者只允许出现一个；
    即使调用者传了旧观察，也全部由原件重新产生，不读取预填分值。
    """
    from app.services.fund_rating_direct_index import compile_observations

    packet = deepcopy(packet)
    if bool(packet.get("direct_index_compilation")) == bool(packet.get("passive_index_compilation")):
        raise ValueError("R5_PASSIVE_INDEX_SINGLE_COMPILATION_REQUIRED")
    compiled = compile_observations(
        packet.get("direct_index_compilation") or packet["passive_index_compilation"], root=root
    )
    packet["quality_metrics"]["observations"] = compiled["observations"]
    code = packet["fund_code"]
    if code not in anchors.get("material_events", {}):
        raise ValueError("R5_DIRECT_INDEX_EXPLICIT_MATERIAL_EVENT_REVIEW_REQUIRED")
    # 只重编合同条款，不清空调用者已核的重大事项；适用限制仍进入完整计算。
    anchors = {"material_events": deepcopy(anchors["material_events"]), code: compiled["anchor"]}
    return prepare_complete(session, packet, anchors, root=root)


def prepare_passive_holdings(session, request: dict, *, root: Path) -> dict:
    """仅补已由完整原件复算的被动持仓维度，保留其他维度、日期和历史。

    不接收分数；更新清单进入原标准登记/逐只事务发布。即使八维已有其他
    历史候选，也不凭本组件直接晋级完整状态，完整总分仍须全包R5复算。
    """
    from app.services.fund_rating_passive_holdings import calculate as calculate_holdings
    from app.services.fund_rating_reviewed_snapshot import calculate as calculate_snapshot
    from app.services.fund_rating_reviewed_snapshot import validate_sources

    code = request["fund_code"]
    if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("RATING_PASSIVE_HOLDINGS_CODE")
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("RATING_PASSIVE_HOLDINGS_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    if bundle.get("basis") != BASIS or bundle.get("r5_calculation") or bundle.get("family") != "INDEX":
        raise ValueError("RATING_PASSIVE_HOLDINGS_PARTIAL_SNAPSHOT_REQUIRED")
    raw = deepcopy(request["inputs"])
    if (
        raw["fund_code"] != code
        or raw["as_of_date"] != request["as_of_date"]
        or raw["holding_date"] != bundle["holding_date"]
        or raw["legal_name"] != bundle["identity"]["legal_name"]
        or raw["primary_ref"] != bundle["original_report_archive"]["sha256"]
    ):
        raise ValueError("RATING_PASSIVE_HOLDINGS_INPUT_SCOPE")
    explanation = request.get("explanation", "").strip()
    if len(explanation) < 40:
        raise ValueError("RATING_PASSIVE_HOLDINGS_EXPLANATION_REQUIRED")
    calculated = calculate_holdings(raw, root=root)
    point = calculated["score"]
    document = bundle["reviewed_document"]
    document["dimensions"]["holdings"] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        passive_holdings_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    bundle.setdefault("public_explanations", {})["holdings"] = explanation
    validate_sources(bundle, datetime.now(UTC), root=root)
    calculate_snapshot(bundle, root=root)
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}


def prepare_passive_execution(session, request: dict, *, root: Path) -> dict:
    """从完整三年同份额原件独立补目标执行；沿用登记和事务发布入口。

    同行不参与此绝对指标，不能成为阻塞条件；其他维度和历史批次保留。
    有全包/七维包时应更新其原包，不能用独立组件造成两套原始观察冲突。
    """
    from app.services.fund_rating_passive_execution import calculate as calculate_execution
    from app.services.fund_rating_reviewed_snapshot import calculate as calculate_snapshot
    from app.services.fund_rating_reviewed_snapshot import validate_sources

    code = request["fund_code"]
    if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("R5_EXECUTION_CODE")
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("R5_EXECUTION_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    if (
        bundle.get("basis") != BASIS
        or bundle.get("r5_calculation")
        or bundle.get("passive_components_calculation")
        or bundle.get("family") != "INDEX"
    ):
        raise ValueError("R5_EXECUTION_PARTIAL_INDEX_REQUIRED")
    raw = deepcopy(request["inputs"])
    if (
        raw["fund_code"] != code
        or raw["as_of_date"] != request["as_of_date"]
        or raw["holding_date"] != bundle["holding_date"]
        or raw["legal_name"] != bundle["identity"]["legal_name"]
        or raw["primary_ref"] != bundle["original_report_archive"]["sha256"]
    ):
        raise ValueError("R5_EXECUTION_INPUT_SCOPE")
    explanation = request.get("explanation", "").strip()
    if len(explanation) < 40:
        raise ValueError("R5_EXECUTION_EXPLANATION_REQUIRED")
    point = calculate_execution(raw, root=root)["score"]
    document = bundle["reviewed_document"]
    document["dimensions"]["execution"] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        passive_execution_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    bundle.setdefault("public_explanations", {})["execution"] = explanation
    bundle.get("public_metrics", {}).pop("execution", None)
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = calculate_snapshot(bundle, root=root)
    if checked["status"] != "PARTIAL" or checked["score"] is not None:
        raise ValueError("R5_EXECUTION_CANNOT_CLAIM_COMPLETE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}


def prepare_active_execution(session, request: dict, *, root: Path) -> dict:
    """沿已核主动合同结构，由公告目录和真实各期资产独立编译目标执行。"""
    from app.services.fund_rating_active_execution import prepare

    return prepare(session, request, root=root)


def prepare_stability(session, request: dict, *, root: Path) -> dict:
    """从已核发行人原件补稳定性；不让同行筛选阻塞独立合同基准比较。"""
    from app.services.fund_rating_active_stability import calculate as calculate_stability
    from app.services.fund_rating_reviewed_snapshot import calculate as calculate_snapshot
    from app.services.fund_rating_reviewed_snapshot import validate_sources

    code, raw = request["fund_code"], deepcopy(request["inputs"])
    if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("R5_STABILITY_CODE")
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("R5_STABILITY_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    primary = raw.get("refs", {}).get("half") or raw.get("reports", {}).get("2026-06-30")
    if (
        bundle.get("basis") != BASIS
        or bundle.get("family") != "ACTIVE_EQUITY"
        or raw.get("fund_code") != code
        or raw.get("as_of_date") != request["as_of_date"]
        or raw.get("share_class") != bundle["share_class"]
        or raw.get("holding_date") != bundle["holding_date"]
        or raw.get("legal_name") != bundle["identity"]["legal_name"]
        or raw.get("manager_name") != bundle["identity"]["manager_name"]
        or primary != bundle["original_report_archive"]["sha256"]
        or any(
            bundle.get(k)
            for k in ("r5_calculation", "passive_components_calculation", "passive_performance_calculation")
        )
    ):
        raise ValueError("R5_STABILITY_INPUT_SCOPE")
    explanation = request.get("explanation", "").strip()
    if len(explanation) < 40:
        raise ValueError("R5_STABILITY_EXPLANATION_REQUIRED")
    point = calculate_stability(raw, root=root)["score"]
    document = bundle["reviewed_document"]
    document["dimensions"]["stability"] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        stability_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    bundle.setdefault("public_explanations", {})["stability"] = explanation
    bundle.get("public_metrics", {}).pop("stability", None)
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = calculate_snapshot(bundle, root=root)
    if checked["status"] != "PARTIAL" or checked["score"] is not None:
        raise ValueError("R5_STABILITY_CANNOT_CLAIM_COMPLETE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}


def prepare_passive_performance(session, request: dict, *, root: Path) -> dict:
    """独立编译收益、风险、稳定性，绑定已发布的本份额三年执行原件。

    不接受外部分数；每次从本份额、固定同行合同、逐日净值及真实分红
    原件重算三维。保留其他五维和历史，流动性未闭合时不生成完整总分。
    """
    from app.services.fund_rating_passive_performance import calculate as calculate_performance
    from app.services.fund_rating_reviewed_snapshot import calculate as calculate_snapshot
    from app.services.fund_rating_reviewed_snapshot import validate_sources

    code, raw = request["fund_code"], deepcopy(request["inputs"])
    if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("R5_PASSIVE_PERFORMANCE_CODE")
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("R5_PASSIVE_PERFORMANCE_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    own = bundle.get("passive_execution_calculation", {}).get("inputs")
    if (
        bundle.get("basis") != BASIS
        or bundle.get("family") != "INDEX"
        or not own
        or bundle.get("r5_calculation")
        or bundle.get("passive_components_calculation")
        or raw.get("fund_code") != code
        or raw.get("as_of_date") != request["as_of_date"]
        or raw.get("share_class") != bundle["share_class"]
        or raw.get("target_execution_inputs") != {k: v for k, v in own.items() if k != "sources"}
    ):
        raise ValueError("R5_PASSIVE_PERFORMANCE_INPUT_SCOPE")
    explanation = request.get("explanation", "").strip()
    if len(explanation) < 40:
        raise ValueError("R5_PASSIVE_PERFORMANCE_EXPLANATION_REQUIRED")
    value = calculate_performance(raw, root=root)
    document = bundle["reviewed_document"]
    for key in ("return", "risk", "stability"):
        point = value["dimensions"][key]
        document["dimensions"][key] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
        bundle.setdefault("public_explanations", {})[key] = explanation
        bundle.get("public_metrics", {}).pop(key, None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        passive_performance_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = calculate_snapshot(bundle, root=root)
    if checked["status"] != "PARTIAL" or checked["score"] is not None:
        raise ValueError("R5_PASSIVE_PERFORMANCE_CANNOT_CLAIM_COMPLETE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}


def prepare_enhanced_performance(session, request: dict, *, root: Path) -> dict:
    """增强指数三维逐件复算；份额、期间和重大事项均绑定原件，保留其他维度。"""
    from app.services.fund_rating_enhanced_performance import prepare

    return prepare(session, request, root=root)


def prepare_mixed_performance(session, request: dict, *, root: Path) -> dict:
    """按本A份额、79独立产品及原件事项核两项表现，保留现金等实际未知。"""
    from app.services.fund_rating_mixed_performance import prepare

    return prepare(session, request, root=root)


def prepare_ca005187_execution(session, request: dict, *, root: Path) -> dict:
    """只补本C公开执行维；全公司证券合并重算，稳定性诊断不进入点分。"""
    from app.services.fund_rating_ca005187_execution import prepare

    return prepare(session, request, root=root)


def prepare_bd500_performance(session, request: dict, *, root: Path) -> dict:
    """核本C份额真实净值、七独立产品及逐事件审阅后补三项表现。"""
    from app.services.fund_rating_bd500_performance import prepare

    canonical = deepcopy(request)
    canonical["inputs"]["sources"] = {
        digest: canonical_source(root, reference) for digest, reference in request["inputs"]["sources"].items()
    }
    return prepare(session, canonical, root=root)


def prepare_bosera_performance(session, request: dict, *, root: Path) -> dict:
    """同C人民币QDII的两项历史表现逐件复算；其他维度及存续未知仍保留。"""
    from app.services.fund_rating_bosera_qdii_performance import prepare

    return prepare(session, request, root=root)


def prepare_active_performance(session, request: dict, *, root: Path) -> dict:
    """从最新底稿核原件补主动表现，收益与稳定两包按独立作用范围保留。"""
    from app.services.fund_rating_active_performance_adapter import prepare_active_performance as prepare

    return prepare(session, request, root=root)


def prepare_active_broad_components(session, request: dict, *, root: Path) -> dict:
    """从全505发现池原件重编007950收益/稳定性，不接收执行点分。"""
    from app.services.fund_rating_active_broad_components import prepare

    canonical = deepcopy(request)
    canonical["inputs"]["sources"] = {
        digest: canonical_source(root, reference) for digest, reference in request["inputs"]["sources"].items()
    }
    return prepare(session, canonical, root=root)


def prepare_fees(session, request: dict, *, root: Path) -> dict:
    """独立补本期本份额的绝对费用项，其余维度、日期和历史批次保持。"""
    from app.services.fund_rating_fee_inputs import calculate as calculate_fees
    from app.services.fund_rating_reviewed_snapshot import calculate as calculate_snapshot
    from app.services.fund_rating_reviewed_snapshot import validate_sources

    code = request["fund_code"]
    if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("R5_FEE_CODE")
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("R5_FEE_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    if bundle.get("basis") != BASIS or bundle.get("r5_calculation") or bundle.get("passive_components_calculation"):
        raise ValueError("R5_FEE_PARTIAL_SNAPSHOT_REQUIRED")
    raw = deepcopy(request["inputs"])
    if (
        raw["fund_code"] != code
        or raw["as_of_date"] != request["as_of_date"]
        or raw["share_class"] != bundle["share_class"]
        or raw["legal_name"] != bundle["identity"]["legal_name"]
        or raw["primary_ref"] != bundle["original_report_archive"]["sha256"]
    ):
        raise ValueError("R5_FEE_INPUT_SCOPE")
    explanation = request.get("explanation", "").strip()
    if len(explanation) < 40:
        raise ValueError("R5_FEE_EXPLANATION_REQUIRED")
    calculated = calculate_fees(raw, root=root)
    point = calculated["score"]
    document = bundle["reviewed_document"]
    document["dimensions"]["fee"] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        fee_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    bundle.setdefault("public_explanations", {})["fee"] = explanation
    bundle.setdefault("data_dates", {})["fees"] = calculated["published_on"]
    bundle.get("public_metrics", {}).pop("fee", None)
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = calculate_snapshot(bundle, root=root)
    if checked["status"] != "PARTIAL" or checked["score"] is not None:
        raise ValueError("R5_FEE_CANNOT_CLAIM_COMPLETE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}


def prepare_passive_components(session, request: dict, *, root: Path) -> dict:
    """完整流动性未披露时，独立发布从原件复算的其余七维，保持完整总分为空。"""
    from app.services.fund_rating_r5 import calculate_passive_components
    from app.services.fund_rating_reviewed_quality import validate_sources as validate_quality_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as calculate_snapshot
    from app.services.fund_rating_reviewed_snapshot import validate_sources

    packet, anchors = deepcopy(request["packet"]), deepcopy(request["anchors"])
    code = packet["fund_code"]
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != packet["as_of_date"]:
        raise ValueError("R5_PASSIVE_COMPONENT_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    if bundle.get("basis") != BASIS or bundle.get("r5_calculation") or bundle.get("family") != "INDEX":
        raise ValueError("R5_PASSIVE_COMPONENT_PARTIAL_INDEX_REQUIRED")
    inputs = packet["passive_index_compilation"]
    if (
        inputs["target_refs"]["half"] != bundle["original_report_archive"]["sha256"]
        or inputs["legal_name"] != bundle["identity"]["legal_name"]
        or inputs["holding_date"] != bundle["holding_date"]
    ):
        raise ValueError("R5_PASSIVE_COMPONENT_CURRENT_ORIGINAL_REQUIRED")
    blocker = request.get("liquidity_blocker", "").strip()
    explanations = request.get("explanations", {})
    if len(blocker) < 40 or any(len(explanations.get(k, "").strip()) < 20 for k in KEYS if k != "liquidity"):
        raise ValueError("R5_PASSIVE_COMPONENT_SPECIFIC_EXPLANATIONS_REQUIRED")
    if code not in anchors.get("material_events", {}):
        raise ValueError("R5_PASSIVE_COMPONENT_EXPLICIT_EVENT_REVIEW_REQUIRED")
    validate_quality_sources(packet, datetime.now(UTC), root=root)
    computed = calculate_passive_components(packet, anchors, root=root)
    document = bundle["reviewed_document"]
    for key, calculation_key in KEYS.items():
        if key == "liquidity":
            continue
        value = computed["dimensions"][calculation_key]
        explanation = explanations[key].strip()
        document["dimensions"][key] = {"lower": value, "upper": value, "candidate": value, "reason": explanation}
        bundle.setdefault("public_explanations", {})[key] = explanation
        bundle.get("public_metrics", {}).pop(key, None)
    bundle.setdefault("public_explanations", {})["liquidity"] = blocker
    # 被本批七维原件计算替换的旧单维计算保留在旧批次；当前包只绑定新的
    # 同一组原始观察，避免旧的范围计算与新点值在同一当前包内相互冲突。
    bundle.pop("management_calculation", None)
    bundle.pop("passive_holdings_calculation", None)
    bundle.pop("passive_execution_calculation", None)
    bundle.pop("fee_calculation", None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        comparison_product_count=len(inputs["peers"]),
        passive_components_calculation={
            "packet": packet,
            "anchors": anchors,
            "input_archive": archive_json(root, packet),
            "anchors_archive": archive_json(root, anchors),
        },
    )
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = calculate_snapshot(bundle, root=root)
    if checked["status"] != "PARTIAL" or checked["score"] is not None:
        raise ValueError("R5_PASSIVE_COMPONENT_MUST_REMAIN_PARTIAL")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
