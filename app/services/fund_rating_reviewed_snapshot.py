"""将用户批准的已有研究结果追加到正式批次，保留区间、缺项及覆盖率。

这是本机受控导入入口，不接收浏览器提交的分数。原研究原件按哈希归档，
完整八维仍为空时绝不生成总分或等级；货币M3可核的三维由正式纯计算器重算。
未完成的独立方法验收如实封存，不伪造旧校准通过记录。
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.fund import FundMaster, FundShareClass
from app.models.fund_rating import (
    RatingAudit,
    RatingBatch,
    RatingClassification,
    RatingCurrent,
    RatingEvidence,
    RatingMember,
    RatingMethodology,
    RatingResult,
)
from app.services.fund_rating_inputs import digest
from app.services.material_content import content_digest, source_bytes, source_exists, source_size, verification_memo

BASIS = "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3"
KEYS = {
    "return": "return",
    "risk": "risk",
    "execution": "efficiency",
    "stability": "stability",
    "management": "management",
    "holdings": "holdings",
    "liquidity": "liquidity",
    "fee": "cost",
}
NAMES = dict(
    zip(
        KEYS,
        ["收益能力", "风险控制", "目标执行", "业绩稳定性", "管理与策略", "持仓结构", "规模与流动性", "费用"],
        strict=True,
    )
)
PUBLIC_EXPLANATIONS = {
    "return": "使用本份额实际费后收益评价；缺少完整可比收益或适配参照时保留范围。",
    "risk": "按基金类型评价已核风险观测；未明确的信用、期限等暴露仍保留缺项。",
    "execution": "核对已公开的合同约束和实际配置；尚未核准的资产资格、历史偏离与调整记录继续列为缺项。",
    "stability": "观察跨月份和滚动年度的持续表现；比较资料不完整时保留范围。",
    "management": "根据已核的经理职责、交接连续性和治理事项评价。",
    "holdings": "核对持仓集中、资产质量及适用的底层穿透；未识别机构或信用依据不足的部分保留范围。",
    "liquidity": "根据已公开的资产可变现性、现金负债及持有人结构评价；逐笔支取和到账条件未明确的部分仍为缺项。",
    "fee": "按该份额公开费用和统一持有场景评价；底层费用减免等未明确部分保留缺项。",
}


def archive_root() -> Path:
    return (Path(get_settings().fund_insight_directory) / "ratings").resolve()


def _number(value) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError("RATING_SNAPSHOT_NUMBER_MISSING")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("RATING_SNAPSHOT_NUMBER_INVALID")
    return result


def _read(ref: dict, root: Path, *, max_bytes: int = 25_000_000) -> dict | None:
    """限定历史sources范围，读取库内提取信息并核来源版本；不要求原件存在。"""
    path = (root / ref["file"]).resolve()
    if not 0 < max_bytes <= 50_000_000:
        raise ValueError("RATING_SNAPSHOT_ARCHIVE_SIZE_BOUND")
    if not path.is_relative_to(root / "sources") or not source_exists(path) or source_size(path) > max_bytes:
        raise ValueError("RATING_SNAPSHOT_ARCHIVE_PATH")
    content = source_bytes(path)
    if content_digest(content) != ref["sha256"]:
        raise ValueError("RATING_SNAPSHOT_ARCHIVE_HASH")
    return json.loads(content.decode("utf-8-sig")) if ref.get("kind") == "json" else None


def _case(document: dict, code: str, format_name: str) -> dict:
    if format_name == "nine_cases":
        matches = [r for r in document["all_cases"] if r["code"] == code]
        if len(matches) != 1:
            raise ValueError("RATING_SNAPSHOT_CASE_IDENTITY")
        return matches[0]
    if document.get("code") != code:
        raise ValueError("RATING_SNAPSHOT_CASE_IDENTITY")
    return document


def validate_sources(bundle: dict, frozen: datetime, *, root: Path | None = None) -> None:
    """研究取得日可晚于截止日；报告本身原公开日必须早于评价截止日。"""
    if get_settings().app_environment != "local" or bundle.get("basis") != BASIS:
        raise ValueError("RATING_SNAPSHOT_LOCAL_SCOPE")
    root = (root or archive_root()).resolve()
    document = _read(bundle["research_archive"], root)
    if _case(document, bundle["fund_code"], bundle["research_format"]) != bundle["reviewed_document"]:
        raise ValueError("RATING_SNAPSHOT_DOCUMENT_MISMATCH")
    identity = _read(bundle["identity_archive"], root)
    if not identity.get("identity_verified") or identity["code"] != bundle["fund_code"]:
        raise ValueError("RATING_SNAPSHOT_ORIGINAL_IDENTITY")
    if identity["published_on"] > bundle["as_of_date"]:
        raise ValueError("RATING_SNAPSHOT_ORIGINAL_FUTURE")
    _read(bundle["original_report_archive"], root)
    if identity["source"]["sha256"] != bundle["original_report_archive"]["sha256"]:
        raise ValueError("RATING_SNAPSHOT_REPORT_IDENTITY_HASH")
    plain = re.sub(r"\s+", "", "".join(identity["pages"]))
    if (
        bundle["fund_code"] not in plain
        or bundle["identity"]["manager_name"] not in plain
        or re.sub(r"\s+", "", bundle["identity"]["legal_name"]) not in re.sub(r"\s+", "", identity["title"])
    ):
        raise ValueError("RATING_SNAPSHOT_LEGAL_IDENTITY_MISMATCH")
    if bundle.get("evidence_v1_calculation"):
        from app.services.fund_rating_evidence_preview import REVISION

        v1 = bundle["evidence_v1_calculation"]
        if (
            v1["revision"] != REVISION
            or _read(bundle["evidence_v1_archive"], root) != v1
            or any(key.endswith("_calculation") and key != "evidence_v1_calculation" for key in bundle)
        ):
            raise ValueError("RATING_EVIDENCE_V1_ARCHIVE_MISMATCH")
    if bundle.get("money_input_archive") and _read(bundle["money_input_archive"], root) != bundle.get("money_inputs"):
        raise ValueError("RATING_SNAPSHOT_MONEY_INPUT_HASH")
    if bundle.get("active_portfolio_components_calculation"):
        from app.services.fund_rating_active_portfolio_components import validate_input_archive

        validate_input_archive(bundle, root=root, archive_reader=_read)
    if bundle.get("active_004605_holdings_calculation"):
        from app.services.fund_rating_004605_holdings import validate_input_archive

        validate_input_archive(bundle, root=root or archive_root(), archive_reader=_read)
    if bundle.get("active_004605_execution_calculation"):
        from app.services.fund_rating_004605_execution import validate_input_archive

        validate_input_archive(bundle, root=root or archive_root(), archive_reader=_read)
    if "active_performance_calculations" in bundle:
        from app.services.fund_rating_active_performance_adapter import validate_input_archives

        validate_input_archives(bundle, root=root, archive_reader=_read)
    if bundle.get("management_calculation"):
        raw = bundle["management_calculation"]
        if _read(raw["input_archive"], root) != raw["inputs"]:
            raise ValueError("RATING_MANAGEMENT_INPUT_ARCHIVE")
    if bundle.get("passive_holdings_calculation"):
        raw = bundle["passive_holdings_calculation"]
        if _read(raw["input_archive"], root) != raw["inputs"]:
            raise ValueError("RATING_PASSIVE_HOLDINGS_INPUT_ARCHIVE")
    if bundle.get("passive_execution_calculation"):
        raw = bundle["passive_execution_calculation"]
        if _read(raw["input_archive"], root) != raw["inputs"]:
            raise ValueError("R5_EXECUTION_INPUT_ARCHIVE")
    for field in (
        "active_execution_calculation",
        "stability_calculation",
        "passive_performance_calculation",
        "gold_components_calculation",
        "enhanced_performance_calculation",
        "bosera_performance_calculation",
        "bd500_performance_calculation",
        "bd500_liquidity_calculation",
        "health_stock_liquidity_interval_calculation",
        "mixed_performance_calculation",
        "ca005187_execution_calculation",
        "gf_battery_liquidity_interval_calculation",
        "active_broad_components_calculation",
        "japan_active_execution_calculation",
        "japan_return_calculation",
        "huaan_qdii_relative_calculation",
        "huaan_qdii_benchmark_calculation",
        "fund_013853_execution_calculation",
        "money_aggregate_components_calculation",
        "efunds_complete_holdings_calculation",
        "yy_manufacturing_performance_calculation",
        "yy_manufacturing_published_liquidity_calculation",
        "yy_manufacturing_reviewed_holdings_calculation",
        "efunds_enhanced_components_calculation",
    ):
        if bundle.get(field):
            raw = bundle[field]
            if _read(raw["input_archive"], root) != raw["inputs"]:
                raise ValueError("R5_COMPONENT_INPUT_ARCHIVE_MISMATCH:" + field)
    if bundle.get("japan_active_execution_calculation"):
        from app.services.fund_rating_japan_active_execution import check_binding

        check_binding(bundle, bundle["japan_active_execution_calculation"]["inputs"])
    if bundle.get("japan_return_calculation"):
        from app.services.fund_rating_japan_return import check_binding as check_japan_return

        check_japan_return(bundle, bundle["japan_return_calculation"]["inputs"])
    if bundle.get("huaan_qdii_relative_calculation"):
        from app.services.fund_rating_huaan_qdii_relative import replay_snapshot as replay_huaan_relative

        replay_huaan_relative(bundle, bundle["reviewed_document"]["dimensions"], root=root)
    if bundle.get("huaan_qdii_benchmark_calculation"):
        from app.services.fund_rating_huaan_qdii_benchmark import replay_snapshot as replay_huaan_benchmark

        replay_huaan_benchmark(bundle, bundle["reviewed_document"]["dimensions"], root=root)
    if bundle.get("fund_013853_execution_calculation"):
        from app.services.fund_rating_013853_execution import check_binding

        check_binding(bundle, bundle["fund_013853_execution_calculation"]["inputs"])
    if bundle.get("money_aggregate_components_calculation"):
        from app.services.fund_rating_money_aggregate_components import validate_input_archive

        validate_input_archive(bundle, root=root)
    if bundle.get("gf_battery_liquidity_interval_calculation"):
        from app.services.fund_rating_gf_battery_liquidity_interval import check_binding, replay_snapshot_interval

        raw = bundle["gf_battery_liquidity_interval_calculation"]["inputs"]
        check_binding(bundle, raw)
        interval = bundle["reviewed_document"]["dimensions"]["liquidity"]
        replay_snapshot_interval(bundle, {"liquidity": interval}, root=root)
    if bundle.get("health_stock_liquidity_interval_calculation"):
        from app.services.fund_rating_health_stock_liquidity_interval import check_binding, replay_snapshot_interval

        raw = bundle["health_stock_liquidity_interval_calculation"]["inputs"]
        check_binding(bundle, raw)
        interval = bundle["reviewed_document"]["dimensions"]["liquidity"]
        if not isinstance(interval, dict) or interval.get("candidate") is not None:
            raise ValueError("HEALTH_INTERVAL_SNAPSHOT_RECALCULATION")
        # 登记和发布前也从两件原件核区间，不能只验研究JSON自身的摘要。
        replay_snapshot_interval(bundle, {"liquidity": interval}, root=root)
    if bundle.get("fee_calculation"):
        raw = bundle["fee_calculation"]
        if _read(raw["input_archive"], root) != raw["inputs"]:
            raise ValueError("R5_FEE_INPUT_ARCHIVE")
    if bundle.get("r5_calculation"):
        # 完整评分从冻结的原始观察重新计算；不能把研究分数表直接改成完整状态。
        raw = bundle["r5_calculation"]
        packet = _read(raw["input_archive"], root)
        anchors = _read(raw["anchors_archive"], root)
        if packet != raw["packet"] or anchors != raw["anchors"]:
            raise ValueError("RATING_R5_INPUT_ARCHIVE_MISMATCH")
        if packet["fund_code"] != bundle["fund_code"] or packet["as_of_date"] != bundle["as_of_date"]:
            raise ValueError("RATING_R5_INPUT_SCOPE_MISMATCH")
        from app.services.fund_rating_reviewed_quality import validate_sources as validate_quality_sources

        validate_quality_sources(packet, frozen, root=root)
    if bundle.get("passive_components_calculation"):
        raw = bundle["passive_components_calculation"]
        if _read(raw["input_archive"], root) != raw["packet"] or _read(raw["anchors_archive"], root) != raw["anchors"]:
            raise ValueError("R5_PASSIVE_COMPONENT_INPUT_ARCHIVE")
        packet = raw["packet"]
        inputs = packet["passive_index_compilation"]
        if (
            packet["fund_code"] != bundle["fund_code"]
            or packet["as_of_date"] != bundle["as_of_date"]
            or inputs["holding_date"] != bundle["holding_date"]
            or inputs["legal_name"] != bundle["identity"]["legal_name"]
            or inputs["target_refs"]["half"] != bundle["original_report_archive"]["sha256"]
        ):
            raise ValueError("R5_PASSIVE_COMPONENT_BUNDLE_SCOPE")
        from app.services.fund_rating_reviewed_quality import validate_sources as validate_quality_sources

        validate_quality_sources(packet, frozen, root=root)
    extra_archives = bundle.get("extra_archives", [])
    if extra_archives:
        # 主动参照的跨期适用性底稿可能超过25MB；它是派生审阅清单，
        # 不冒充发行人原件。仅对已绑定到正式组件输入的同一清单沿导入器
        # 50MB硬上限验字节，普通归档仍维持原25MB界，绝不做无界读取。
        registered_large = {
            sha: ref
            for packet in (
                *bundle.get("active_performance_calculations", []),
                *(
                    [bundle["active_broad_components_calculation"]]
                    if bundle.get("active_broad_components_calculation")
                    else []
                ),
            )
            for sha, ref in packet["inputs"]["sources"].items()
            if ref.get("kind") == "DERIVED_MANDATE_REVIEW"
        }
        # 仅同一混合基金组件已绑定的原PDF可沿其50MB契约校验；普通归档仍25MB。
        mixed_registered_originals = {
            sha: ref
            for sha, ref in bundle.get("mixed_performance_calculation", {}).get("inputs", {}).get("sources", {}).items()
            if ref.get("kind") == "pdf" and ref.get("archive_kind", "ORIGINAL") == "ORIGINAL"
        }
        # 仅本持仓组件已绑定的原PDF沿原有50MB契约核验；普通归档仍维持25MB。
        yyhr_registered_originals = {
            sha: ref
            for sha, ref in bundle.get("yy_manufacturing_reviewed_holdings_calculation", {})
            .get("inputs", {})
            .get("original_inputs", {})
            .get("sources", {})
            .items()
            if ref.get("kind") == "pdf" and ref.get("archive_kind") == "ORIGINAL"
        }
        # 本基金持仓只接纳已固定原件集；大PDF由组件逐个核精确SHA及字节数。
        efunds_holdings_originals = {
            sha: ref
            for sha, ref in bundle.get("efunds_complete_holdings_calculation", {})
            .get("inputs", {})
            .get("sources", {})
            .items()
            if ref.get("kind") == "pdf" and ref.get("archive_kind") == "ORIGINAL"
        }

        # 持续补证后可能有上千份资料。每次请求核对受限来源、原长度和库内内容摘要，
        # 及 JSON 格式，仅把相互独立的本地读取并行，避免页面超过服务间读取时限。
        # 不跨请求缓存业务结论；原件清理不影响读取，新来源版本通过摘要独立隔离。
        def verify_archive(ref: dict) -> None:
            quality_sources = (
                bundle.get("active_004605_holdings_calculation", {})
                .get("inputs", {})
                .get("asset_quality_inputs", {})
                .get("sources", {})
            )
            quality_ref = quality_sources.get(ref["sha256"])
            # 仅本持仓包已绑定的同一原PDF允许调用其精确大小和流式SHA校验，
            # 不放宽其他基金、普通JSON或未登记原件的既有读取上限。
            if (
                quality_ref
                and quality_ref.get("kind") == "pdf"
                and {key: quality_ref[key] for key in ("file", "sha256", "kind")} == ref
            ):
                from app.services.fund_rating_004605_holdings import verify_registered_original

                verify_registered_original(bundle, ref, root=root)
            elif efunds_holdings_originals.get(ref["sha256"]) == ref:
                from app.services.fund_rating_efunds_complete_holdings import verify_registered_original

                verify_registered_original(bundle, ref, root=root)
            elif registered_large.get(ref["sha256"]) == ref:
                _read(ref, root, max_bytes=50_000_000)
            elif mixed_registered_originals.get(ref["sha256"]) == ref:
                from app.services.fund_rating_mixed_performance import verify_registered_original

                verify_registered_original(bundle, ref, root=root)
            elif yyhr_registered_originals.get(ref["sha256"]) == ref:
                from app.services.fund_rating_yy_manufacturing_holdings_reviewed import verify_registered_original

                verify_registered_original(bundle, ref, root=root)
            else:
                _read(ref, root)

        with ThreadPoolExecutor(max_workers=min(8, len(extra_archives))) as executor:
            # 按输入顺序取完结果，确保任一工作线程的校验异常都向调用方抛出。
            # 工作线程不返回解析正文，避免把全部大份 JSON 同时留在结果列表中。
            for _ in executor.map(verify_archive, extra_archives):
                pass
    verified = datetime.fromisoformat(bundle["verified_at"])
    if verified.tzinfo is None or verified > frozen or verified + timedelta(days=31) <= frozen:
        raise ValueError("RATING_SNAPSHOT_VERIFICATION_EXPIRED")
    if bundle["holding_date"] > bundle["as_of_date"] or date.fromisoformat(bundle["as_of_date"]) > frozen.date():
        raise ValueError("RATING_SNAPSHOT_DATE_FUTURE")


@verification_memo
def calculate(bundle: dict, *, root: Path | None = None) -> dict:
    """从封存的逐维研究观测复算原权重合成；不把0..100域当已评分区间。"""
    with localcontext() as context:
        context.prec = 80
        doc = bundle["reviewed_document"]
        code = bundle["fund_code"]
        if doc.get("code") != code:
            raise ValueError("RATING_SNAPSHOT_CASE_IDENTITY")
        if doc.get("as_of", doc.get("cutoff")) != bundle["as_of_date"]:
            raise ValueError("RATING_SNAPSHOT_AS_OF_MISMATCH")
        weights = doc["outer_weights_percent"]
        if set(weights) != set(KEYS) or sum(_number(v) for v in weights.values()) != 100:
            raise ValueError("RATING_SNAPSHOT_EIGHT_WEIGHTS")
        if set(doc["dimensions"]) != set(KEYS):
            raise ValueError("RATING_SNAPSHOT_EIGHT_DIMENSIONS")
        dimensions = {}
        # JSONB不保留对象键顺序；始终按唯一八维顺序复算和生成展示数组。
        for key in KEYS:
            value = doc["dimensions"][key]
            if isinstance(value, dict):
                lo, hi = _number(value["lower"]), _number(value["upper"])
                point = value.get("candidate")
                reason = value.get("reason", "")
            elif value is None:
                lo, hi, point, reason = Decimal(0), Decimal(100), None, doc.get("specific_blocker", "必要资料尚未核准")
            else:
                lo = hi = _number(value)
                point = str(value)
                reason = "沿用已核原件及原方法的确定候选"
            if not 0 <= lo <= hi <= 100 or (point is not None and not lo == hi == _number(point)):
                raise ValueError("RATING_SNAPSHOT_INTERVAL_INVALID")
            dimensions[key] = {
                "lower": str(lo),
                "upper": str(hi),
                "candidate": str(point) if point is not None else None,
                "reason": reason or "按公开披露数据计算；未公开的底层信息仍保留缺项",
            }
        if bundle.get("gf_battery_liquidity_interval_calculation"):
            from app.services.fund_rating_gf_battery_liquidity_interval import replay_snapshot_interval

            replay_snapshot_interval(bundle, dimensions, root=root or archive_root())
        if bundle.get("ca005187_execution_calculation"):
            from app.services.fund_rating_ca005187_execution import validate_snapshot_component

            validate_snapshot_component(bundle, root=root or archive_root())
        if bundle.get("health_stock_liquidity_interval_calculation"):
            from app.services.fund_rating_health_stock_liquidity_interval import replay_snapshot_interval

            # 查询、历史重放及发布复核均走同一原件计算；未知容量不变成点分。
            replay_snapshot_interval(bundle, dimensions, root=root or archive_root())
        if bundle.get("money_inputs"):
            from app.services.fund_rating_money_public import execution, holdings, liquidity

            raw = bundle["money_inputs"]
            values = {
                "execution": execution(raw["numeric_execution_observations"], raw["category_observations"]),
                "holdings": holdings(raw["holdings"]),
                "liquidity": liquidity(raw["liquidity"]),
            }
            for key, value in values.items():
                for bound in ["lower", "upper"]:
                    if abs(_number(value[bound]) - _number(dimensions[key][bound])) > Decimal("1e-60"):
                        raise ValueError("RATING_SNAPSHOT_MONEY_RECALCULATION")
        if bundle.get("money_aggregate_components_calculation"):
            from app.services.fund_rating_money_aggregate_components import replay_snapshot as replay_money_aggregate

            replay_money_aggregate(bundle, dimensions, root=root or archive_root())
        if bundle.get("active_004605_holdings_calculation"):
            from app.services.fund_rating_004605_holdings import replay_snapshot as replay_004605_holdings

            replay_004605_holdings(bundle, dimensions, root=root or archive_root())
        if bundle.get("active_004605_execution_calculation"):
            from app.services.fund_rating_004605_execution import replay_snapshot as replay_004605_execution

            replay_004605_execution(bundle, dimensions, root=root or archive_root())
        if bundle.get("fund_013853_execution_calculation"):
            from app.services.fund_rating_013853_execution import replay_snapshot as replay_013853_execution

            replay_013853_execution(bundle, dimensions, root=root or archive_root())
        if bundle.get("huaan_qdii_relative_calculation"):
            from app.services.fund_rating_huaan_qdii_relative import replay_snapshot as replay_huaan_relative

            replay_huaan_relative(bundle, dimensions, root=root or archive_root())
        if bundle.get("huaan_qdii_benchmark_calculation"):
            from app.services.fund_rating_huaan_qdii_benchmark import replay_snapshot as replay_huaan_benchmark

            replay_huaan_benchmark(bundle, dimensions, root=root or archive_root())
        if bundle.get("japan_active_execution_calculation"):
            from app.services.fund_rating_japan_active_execution import replay_snapshot as replay_japan_execution

            replay_japan_execution(bundle, dimensions, root=root or archive_root())
        if bundle.get("japan_return_calculation"):
            from app.services.fund_rating_japan_return import replay_snapshot as replay_japan_return

            replay_japan_return(bundle, dimensions, root=root or archive_root())
        if bundle.get("active_portfolio_components_calculation"):
            from app.services.fund_rating_active_portfolio_components import replay_snapshot

            replay_snapshot(bundle, dimensions, root=root or archive_root())
        if "active_performance_calculations" in bundle:
            from app.services.fund_rating_active_performance_adapter import replay_components

            replay_components(bundle, dimensions, root=root or archive_root())
        if bundle.get("management_calculation"):
            from app.services.fund_rating_management_inputs import calculate as calculate_management

            value = calculate_management(
                bundle["management_calculation"]["inputs"],
                root=root or archive_root(),
                manager_name=bundle["identity"]["manager_name"],
                as_of=bundle["as_of_date"],
                fund_code=bundle["fund_code"],
                legal_name=bundle["identity"]["legal_name"],
                primary_report_hash=bundle["original_report_archive"]["sha256"],
            )
            candidate = dimensions["management"]["candidate"]
            if (
                (candidate is None) != (value["score"] is None)
                or (candidate is not None and _number(candidate) != _number(value["score"]))
                or any(
                    _number(dimensions["management"][bound]) != _number(value[bound]) for bound in ("lower", "upper")
                )
            ):
                raise ValueError("RATING_MANAGEMENT_DIMENSION_RECALCULATION")
        if bundle.get("passive_holdings_calculation"):
            from app.services.fund_rating_passive_holdings import calculate as calculate_holdings

            raw = bundle["passive_holdings_calculation"]["inputs"]
            if (
                raw["fund_code"] != bundle["fund_code"]
                or raw["as_of_date"] != bundle["as_of_date"]
                or raw["holding_date"] != bundle["holding_date"]
                or raw["legal_name"] != bundle["identity"]["legal_name"]
                or raw["primary_ref"] != bundle["original_report_archive"]["sha256"]
            ):
                raise ValueError("RATING_PASSIVE_HOLDINGS_INPUT_SCOPE")
            value = calculate_holdings(raw, root=root or archive_root())
            if any(
                dimensions["holdings"][key] is None
                or abs(_number(dimensions["holdings"][key]) - _number(value["score"])) > Decimal("1e-55")
                for key in ("lower", "upper", "candidate")
            ):
                raise ValueError("RATING_PASSIVE_HOLDINGS_DIMENSION_RECALCULATION")
        if bundle.get("gold_components_calculation"):
            from app.services.fund_rating_gold_components import calculate as calculate_gold

            raw = bundle["gold_components_calculation"]["inputs"]
            if (
                bundle.get("family") != "INDEX"
                or raw.get("fund_code") != bundle["fund_code"]
                or raw.get("as_of_date") != bundle["as_of_date"]
                or raw.get("holding_date") != bundle["holding_date"]
                or raw.get("share_class") != bundle["share_class"]
                or raw.get("legal_name") != bundle["identity"]["legal_name"]
                or raw.get("manager_name") != bundle["identity"]["manager_name"]
                or raw.get("primary_ref") != bundle["original_report_archive"]["sha256"]
                or any(
                    bundle.get(k)
                    for k in (
                        "r5_calculation",
                        "passive_components_calculation",
                        "passive_performance_calculation",
                        "stability_calculation",
                    )
                )
            ):
                raise ValueError("R5_GOLD_COMPONENT_INPUT_SCOPE")
            points = calculate_gold(raw, root=root or archive_root())["dimensions"]
            for key in ("return", "stability"):
                if any(
                    dimensions[key][bound] is None
                    or abs(_number(dimensions[key][bound]) - _number(points[key])) > Decimal("1e-55")
                    for bound in ("lower", "upper", "candidate")
                ):
                    raise ValueError("R5_GOLD_COMPONENT_DIMENSION_RECALCULATION:" + key)
        if bundle.get("bosera_performance_calculation"):
            from app.services.fund_rating_bosera_qdii_performance import calculate as calculate_bosera
            from app.services.fund_rating_bosera_qdii_performance import check_binding

            raw = bundle["bosera_performance_calculation"]["inputs"]
            check_binding(bundle, raw)
            points = calculate_bosera(raw, root=root or archive_root())["component_scores"]
            for key in ("return", "stability"):
                if any(
                    dimensions[key][bound] is None
                    or abs(_number(dimensions[key][bound]) - _number(points[key])) > Decimal("1e-55")
                    for bound in ("lower", "upper", "candidate")
                ):
                    raise ValueError("BOSERA_QDII_DIMENSION_RECALCULATION:" + key)
        if bundle.get("bd500_liquidity_calculation"):
            # 流动性独立包必须从完整持仓和真实成交原件重放，不能继承候选点值。
            from app.services.fund_rating_bd500_liquidity import check_snapshot as check_bd500_liquidity

            check_bd500_liquidity(bundle, root=root or archive_root())
        if bundle.get("bd500_performance_calculation"):
            from app.services.fund_rating_bd500_performance import check_snapshot

            check_snapshot(bundle, dimensions, root=root or archive_root())
        if bundle.get("mixed_performance_calculation"):
            from app.services.fund_rating_mixed_performance import calculate as calculate_mixed
            from app.services.fund_rating_mixed_performance import check_binding

            raw = bundle["mixed_performance_calculation"]["inputs"]
            check_binding(bundle, raw)
            if any(
                bundle.get(k)
                for k in (
                    "r5_calculation",
                    "passive_components_calculation",
                    "passive_performance_calculation",
                    "enhanced_performance_calculation",
                    "bosera_performance_calculation",
                    "gold_components_calculation",
                )
            ):
                raise ValueError("R5_MIXED_PERFORMANCE_CONFLICTING_INPUTS")
            result = calculate_mixed(raw, root=root or archive_root())
            if result.get("admission_ready") is not True:
                raise ValueError("R5_MIXED_PERFORMANCE_CONFIRMED_MATERIAL_EVENT")
            for key in ("return", "risk"):
                if any(
                    dimensions[key][bound] is None
                    or abs(_number(dimensions[key][bound]) - _number(result["dimensions"][key])) > Decimal("1e-55")
                    for bound in ("lower", "upper", "candidate")
                ):
                    raise ValueError("R5_MIXED_PERFORMANCE_DIMENSION_RECALCULATION:" + key)
        if bundle.get("active_broad_components_calculation"):
            from app.services.fund_rating_active_broad_components import check_snapshot

            check_snapshot(bundle, dimensions, root=root or archive_root())
        if bundle.get("efunds_enhanced_components_calculation"):
            from app.services.fund_rating_efunds_components import calculate as calculate_efunds
            from app.services.fund_rating_efunds_components import check_binding as check_efunds_binding

            raw = bundle["efunds_enhanced_components_calculation"]["inputs"]
            check_efunds_binding(bundle, raw)
            # 同一维度只允许一个原件重放组件；已有稳定性组件可继续独立核验。
            if any(
                bundle.get(k)
                for k in (
                    "r5_calculation",
                    "passive_components_calculation",
                    "passive_execution_calculation",
                    "passive_performance_calculation",
                    "enhanced_performance_calculation",
                    "active_execution_calculation",
                    "active_broad_components_calculation",
                    "gold_components_calculation",
                    "efunds_mechanism_components_calculation",
                )
            ):
                raise ValueError("EFUNDS_CONFLICTING_COMPONENTS")
            points = calculate_efunds(raw, root=root or archive_root())["dimensions"]
            if set(points) != set(raw["selected_dimensions"]):
                raise ValueError("EFUNDS_SELECTED_DIMENSION_SCOPE")
            for key in raw["selected_dimensions"]:
                if any(
                    dimensions[key][bound] is None
                    or abs(_number(dimensions[key][bound]) - _number(points[key])) > Decimal("1e-55")
                    for bound in ("lower", "upper", "candidate")
                ):
                    raise ValueError("EFUNDS_DIMENSION_RECALCULATION:" + key)
        if bundle.get("efunds_complete_holdings_calculation"):
            from app.services.fund_rating_efunds_complete_holdings import calculate as calculate_efunds_holdings
            from app.services.fund_rating_efunds_complete_holdings import check_binding as check_efunds_holdings_binding

            raw = bundle["efunds_complete_holdings_calculation"]["inputs"]
            check_efunds_holdings_binding(bundle, raw)
            if raw["selected_dimensions"] != ["holdings"]:
                raise ValueError("EFUNDS_COMPLETE_HOLDINGS_SCOPE")
            points = calculate_efunds_holdings(raw, root=root or archive_root())["dimensions"]
            if set(points) != {"holdings"}:
                raise ValueError("EFUNDS_COMPLETE_HOLDINGS_RESULT_SCOPE")
            for bound in ("lower", "upper", "candidate"):
                if dimensions["holdings"][bound] is None or abs(
                    _number(dimensions["holdings"][bound]) - _number(points["holdings"])
                ) > Decimal("1e-55"):
                    raise ValueError("EFUNDS_COMPLETE_HOLDINGS_RECALCULATION")
        if bundle.get("enhanced_performance_calculation"):
            from app.services.fund_rating_enhanced_performance import calculate as calculate_enhanced
            from app.services.fund_rating_enhanced_performance import check_binding

            raw = bundle["enhanced_performance_calculation"]["inputs"]
            check_binding(bundle, raw)
            if any(
                bundle.get(k)
                for k in (
                    "r5_calculation",
                    "passive_components_calculation",
                    "passive_performance_calculation",
                    "stability_calculation",
                    "gold_components_calculation",
                )
            ):
                raise ValueError("R5_ENHANCED_PERFORMANCE_CONFLICTING_INPUTS")
            points = calculate_enhanced(raw, root=root or archive_root())["dimensions"]
            for key in ("return", "risk", "stability"):
                if any(
                    dimensions[key][bound] is None
                    or abs(_number(dimensions[key][bound]) - _number(points[key])) > Decimal("1e-55")
                    for bound in ("lower", "upper", "candidate")
                ):
                    raise ValueError("R5_ENHANCED_PERFORMANCE_DIMENSION_RECALCULATION:" + key)
        if bundle.get("active_execution_calculation"):
            from app.services.fund_rating_active_execution import calculate as calculate_active_execution

            raw = bundle["active_execution_calculation"]["inputs"]
            if (
                bundle.get("family") != "ACTIVE_EQUITY"
                or raw.get("fund_code") != bundle["fund_code"]
                or raw.get("as_of_date") != bundle["as_of_date"]
                or raw.get("holding_date") != bundle["holding_date"]
                or raw.get("share_class") != bundle["share_class"]
                or raw.get("legal_name") != bundle["identity"]["legal_name"]
                or raw.get("manager_name") != bundle["identity"]["manager_name"]
                or raw.get("primary_ref") != bundle["original_report_archive"]["sha256"]
                or any(
                    bundle.get(k)
                    for k in ("r5_calculation", "passive_components_calculation", "passive_execution_calculation")
                )
            ):
                raise ValueError("R5_ACTIVE_EXECUTION_INPUT_SCOPE")
            point = calculate_active_execution(raw, root=root or archive_root())["score"]
            if any(
                dimensions["execution"][bound] is None
                or abs(_number(dimensions["execution"][bound]) - _number(point)) > Decimal("1e-55")
                for bound in ("lower", "upper", "candidate")
            ):
                raise ValueError("R5_ACTIVE_EXECUTION_DIMENSION_RECALCULATION")
        if bundle.get("passive_execution_calculation"):
            from app.services.fund_rating_passive_execution import calculate as calculate_execution

            raw = bundle["passive_execution_calculation"]["inputs"]
            if (
                bundle.get("family") != "INDEX"
                or raw["fund_code"] != bundle["fund_code"]
                or raw["as_of_date"] != bundle["as_of_date"]
                or raw["holding_date"] != bundle["holding_date"]
                or raw["legal_name"] != bundle["identity"]["legal_name"]
                or raw["primary_ref"] != bundle["original_report_archive"]["sha256"]
                or bundle.get("r5_calculation")
                or bundle.get("passive_components_calculation")
            ):
                raise ValueError("R5_EXECUTION_INPUT_SCOPE")
            value = calculate_execution(raw, root=root or archive_root())
            if any(
                dimensions["execution"][key] is None
                or abs(_number(dimensions["execution"][key]) - _number(value["score"])) > Decimal("1e-55")
                for key in ("lower", "upper", "candidate")
            ):
                raise ValueError("R5_EXECUTION_DIMENSION_RECALCULATION")
        if bundle.get("stability_calculation"):
            from app.services.fund_rating_active_stability import calculate as calculate_stability

            raw = bundle["stability_calculation"]["inputs"]
            primary = raw.get("refs", {}).get("half") or raw.get("reports", {}).get("2026-06-30")
            if (
                raw.get("fund_code") != bundle["fund_code"]
                or raw.get("as_of_date") != bundle["as_of_date"]
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
            point = calculate_stability(raw, root=root or archive_root())["score"]
            if any(
                dimensions["stability"][bound] is None
                or abs(_number(dimensions["stability"][bound]) - _number(point)) > Decimal("1e-55")
                for bound in ("lower", "upper", "candidate")
            ):
                raise ValueError("R5_STABILITY_DIMENSION_RECALCULATION")
        if bundle.get("passive_performance_calculation"):
            from app.services.fund_rating_passive_performance import calculate as calculate_performance

            raw = bundle["passive_performance_calculation"]["inputs"]
            own = bundle.get("passive_execution_calculation", {}).get("inputs")
            if (
                not own
                or bundle.get("family") != "INDEX"
                or raw.get("fund_code") != bundle["fund_code"]
                or raw.get("as_of_date") != bundle["as_of_date"]
                or raw.get("share_class") != bundle["share_class"]
                or raw.get("target_execution_inputs") != {k: v for k, v in own.items() if k != "sources"}
                or bundle.get("r5_calculation")
                or bundle.get("passive_components_calculation")
            ):
                raise ValueError("R5_PASSIVE_PERFORMANCE_INPUT_SCOPE")
            values = calculate_performance(raw, root=root or archive_root())["dimensions"]
            for key in ("return", "risk", "stability"):
                if any(
                    dimensions[key][bound] is None
                    or abs(_number(dimensions[key][bound]) - _number(values[key])) > Decimal("1e-55")
                    for bound in ("lower", "upper", "candidate")
                ):
                    raise ValueError("R5_PASSIVE_PERFORMANCE_DIMENSION_RECALCULATION")
        if bundle.get("fee_calculation"):
            from app.services.fund_rating_fee_inputs import calculate as calculate_fees

            raw = bundle["fee_calculation"]["inputs"]
            if (
                raw["fund_code"] != bundle["fund_code"]
                or raw["as_of_date"] != bundle["as_of_date"]
                or raw["share_class"] != bundle["share_class"]
                or raw["legal_name"] != bundle["identity"]["legal_name"]
                or raw["primary_ref"] != bundle["original_report_archive"]["sha256"]
                or bundle.get("r5_calculation")
                or bundle.get("passive_components_calculation")
            ):
                raise ValueError("R5_FEE_INPUT_SCOPE")
            value = calculate_fees(raw, root=root or archive_root())
            if any(
                dimensions["fee"][key] is None
                or abs(_number(dimensions["fee"][key]) - _number(value["score"])) > Decimal("1e-55")
                for key in ("lower", "upper", "candidate")
            ):
                raise ValueError("R5_FEE_DIMENSION_RECALCULATION")
        if bundle.get("passive_components_calculation"):
            from app.services.fund_rating_r5 import calculate_passive_components

            raw = bundle["passive_components_calculation"]
            computed = calculate_passive_components(raw["packet"], raw["anchors"], root=root or archive_root())
            for key, calculation_key in KEYS.items():
                if key == "liquidity":
                    continue
                point = dimensions[key]["candidate"]
                if point is None or abs(_number(point) - _number(computed["dimensions"][calculation_key])) > Decimal(
                    "1e-55"
                ):
                    raise ValueError("R5_PASSIVE_COMPONENT_DIMENSION_RECALCULATION")
            if bundle.get("r5_calculation"):
                raise ValueError("R5_PASSIVE_COMPONENT_CANNOT_CLAIM_COMPLETE")
        full = None
        if bundle.get("evidence_v1_calculation"):
            from app.services.fund_rating_evidence_preview import preview
            from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy

            full = preview(bundle, root or archive_root(), review_required=False)
            expected_weights = policy(PASSIVE_BASIS)["weights"]
            if full["status"] != "CANDIDATE" or any(
                _number(weights[k]) != expected_weights[v]
                or dimensions[k]["candidate"] != full["dimensions"][v] for k, v in KEYS.items()
            ):
                raise ValueError("RATING_EVIDENCE_V1_REPLAY_MISMATCH")
        if bundle.get("r5_calculation"):
            from app.services.fund_rating_r5 import calculate as calculate_r5

            raw = bundle["r5_calculation"]
            full = calculate_r5(raw["packet"], raw["anchors"], root=root or archive_root())
            for key, value in dimensions.items():
                if value["candidate"] is None or abs(
                    _number(value["candidate"]) - _number(full["dimensions"][KEYS[key]])
                ) > Decimal("1e-55"):
                    raise ValueError("RATING_R5_DIMENSION_RECALCULATION")
        if bundle.get("yy_manufacturing_performance_calculation"):
            # 三项业绩重新核公告审阅及参照原件；目标执行的未知不在此伪装为正常。
            from app.services.fund_rating_yy_manufacturing_performance import (
                replay_snapshot_points as replay_yy_performance,
            )

            replay_yy_performance(bundle, dimensions, root=root or archive_root())
        if bundle.get("yy_manufacturing_published_liquidity_calculation"):
            # 采用已声明的原始公布成交额口径，保留跨源差异和公开数据的精度边界。
            from app.services.fund_rating_yy_manufacturing_published_liquidity import (
                replay_snapshot_points as replay_yy_liquidity,
            )

            replay_yy_liquidity(bundle, dimensions, root=root or archive_root())
        if bundle.get("yy_manufacturing_reviewed_holdings_calculation"):
            # 行业映射与信用质量为有原件锚点的明示研究判断，快照重放不得手填点分。
            from app.services.fund_rating_yy_manufacturing_holdings_reviewed import replay_snapshot_points

            raw = bundle["yy_manufacturing_reviewed_holdings_calculation"]
            if _read(raw["input_archive"], root or archive_root()) != raw["inputs"]:
                raise ValueError("R5_YYHR_INPUT_ARCHIVE")
            replay_snapshot_points(bundle, dimensions, root=root or archive_root())
        points = [k for k, v in dimensions.items() if v["candidate"] is not None]
        ranges = [
            k
            for k, v in dimensions.items()
            if v["candidate"] is None and (Decimal(v["lower"]) > 0 or Decimal(v["upper"]) < 100)
        ]
        point_weight = sum(_number(weights[k]) for k in points)
        range_weight = sum(_number(weights[k]) for k in ranges)
        known = sum(_number(weights[k]) * _number(dimensions[k]["candidate"]) / 100 for k in points)
        lower = sum(_number(weights[k]) * _number(dimensions[k]["lower"]) / 100 for k in KEYS)
        upper = sum(_number(weights[k]) * _number(dimensions[k]["upper"]) / 100 for k in KEYS)
        return {
            "dimensions": dimensions,
            "weights": weights,
            "point_weight": str(point_weight),
            "interval_weight": str(range_weight),
            "missing_weight": str(100 - point_weight - range_weight),
            "point_dimensions": len(points),
            "interval_dimensions": len(ranges),
            "partial_score": str(known * 100 / point_weight) if point_weight else None,
            "lower": str(lower),
            "upper": str(upper),
            "has_full_informative_interval": point_weight + range_weight == 100,
            "status": "SCORED" if full else "PARTIAL" if point_weight > 0 else "NOT_RATED",
            "score": full["score"] if full else None,
            "grade": None,
        }


def _display(value) -> str:
    return f"{_number(value):.2f}"


def public_evidence(bundle: dict, calculated: dict) -> dict:
    """只传基金业务结果；文件路径、方法号、审批与调度细节不进入页面。"""
    coverage = {
        "pointWeight": calculated["point_weight"],
        "intervalWeight": calculated["interval_weight"],
        "missingWeight": calculated["missing_weight"],
        "pointDimensions": calculated["point_dimensions"],
        "intervalDimensions": calculated["interval_dimensions"],
    }
    all_interval = calculated["has_full_informative_interval"]
    source = {
        "title": bundle["report_title"],
        "url": bundle["report_url"],
        "publishedOn": bundle["report_published_on"],
    }
    dimensions = []
    for key, value in calculated["dimensions"].items():
        meaningful = value["candidate"] is None and (Decimal(value["lower"]) > 0 or Decimal(value["upper"]) < 100)
        dimensions.append(
            {
                "key": KEYS[key],
                "name": NAMES[key],
                "grade": None,
                "gradeLabel": None,
                "score": _display(value["candidate"]) if value["candidate"] is not None else None,
                "scoreRange": {"lower": _display(value["lower"]), "upper": _display(value["upper"])}
                if meaningful
                else None,
                "explanation": bundle.get("public_explanations", {}).get(key, PUBLIC_EXPLANATIONS[key]),
                # 全市场复核可携带同批已核原始事实；旧研究未提供时保持原响应。
                "metrics": bundle.get("public_metrics", {}).get(key, []),
                "sources": bundle.get("dimension_sources", {}).get(key, [source]),
            }
        )
    point_count = calculated["point_dimensions"]
    summary = (
        f"八个维度均已有分数或有效范围；其中{point_count}个维度得分确定，覆盖原权重{calculated['point_weight']}%。"
        "尚未给出单点总分和等级。"
        if all_interval
        else f"已有{point_count}个维度可确定评分，覆盖原权重{calculated['point_weight']}%；"
        "其余资料或方法依据仍不足，暂不提供完整总分和等级。"
    )
    complete = calculated["status"] == "SCORED"
    if complete:
        summary = "八个维度已按真实资料完成综合评分；等级尚未确定。"
        if bundle.get("evidence_v1_calculation"):
            summary = "已按近一年口径完成八维综合评分。"
    return {
        "score": _display(calculated["score"]) if complete else None,
        "scoreRange": {"lower": _display(calculated["lower"]), "upper": _display(calculated["upper"])}
        if all_interval and not complete
        else None,
        "partialScore": _display(calculated["partial_score"])
        if calculated["partial_score"] is not None and not complete
        else None,
        "scoreCoverage": coverage,
        "summary": summary,
        "dimensions": dimensions,
        "comparison": {
            "category": bundle["public_name"],
            "currency": "CNY",
            "productCount": bundle.get("comparison_product_count", 1),
            "scope": "按基金类型使用适用公开资料；部分评分不作为不同覆盖率基金的综合排名。",
            "period": f"截至{bundle['as_of_date']}",
        },
        "strengths": [],
        "weaknesses": [],
        "dataDates": bundle.get("data_dates")
        or {
            "nav": bundle["as_of_date"],
            "holdings": bundle["holding_date"],
            "manager": bundle["as_of_date"],
            "fees": bundle["as_of_date"],
        },
        "limitations": bundle.get("public_limitations")
        or [
            "部分评分仅对已确定维度折算；分数范围保留尚不确定部分，未取中点。",
            "评价公开披露的信息；不代表全部底层数据已齐全、逐日合规或未来赎回保证。",
            "部分专用方法和完整独立验证仍待完成；具体缺项见各维度说明。",
        ],
        "previousRating": None,
    }


def verify_result(result: RatingResult, batch: RatingBatch, *, replay: bool = True) -> None:
    """发布时完整复算；读取时只核验不可变结果与冻结输入，不重复读取原件。"""
    config = batch.input_snapshot["method"]
    members = batch.input_snapshot["members"]
    member = next(m for m in members if m["fund_code"] == result.fund_code)
    bundle = member["bundle"]
    if config.get("approved_artifacts", {}).get(result.fund_code) != bundle["research_archive"]["sha256"]:
        raise ValueError("RATING_SNAPSHOT_REVIEW_SCOPE")
    if digest(bundle) != member["evidence_hash"]:
        raise ValueError("RATING_SNAPSHOT_BUNDLE_HASH")
    if not replay:
        if (
            result.grade is not None
            or not result.dimension_values
            or set(result.dimension_values) != set(KEYS.values())
        ):
            raise ValueError("RATING_SNAPSHOT_RESULT_MISMATCH")
        return
    calculated = calculate(bundle)
    expected = {KEYS[k]: v["candidate"] for k, v in calculated["dimensions"].items()}
    if (
        result.status != calculated["status"]
        or (Decimal(result.score) if result.score is not None else None)
        != (_number(calculated["score"]) if calculated["score"] is not None else None)
        or result.grade is not None
        or result.dimension_values != expected
        or result.public_evidence != public_evidence(bundle, calculated)
    ):
        raise ValueError("RATING_SNAPSHOT_RESULT_MISMATCH")


def publish_reviewed(session: Session, dossier: dict, reason: str, *, root: Path | None = None) -> dict:
    """同一事务追加目录、来源、方法、批次和当前指针；失败整体回滚。"""
    from app.repositories.fund_rating import lock_category, persist_input, publish

    if not reason.strip() or dossier.get("basis") != BASIS or not dossier.get("entries"):
        raise ValueError("RATING_SNAPSHOT_REVIEW_REQUIRED")
    root = (root or archive_root()).resolve()
    now = datetime.now(UTC)
    out = []
    codes = [e["fund_code"] for e in dossier["entries"]]
    if len(codes) != len(set(codes)) or len(codes) > 30:
        raise ValueError("RATING_SNAPSHOT_REVIEW_SCOPE")
    # 此名单已由完整原目录实证为40项发现后按哈希只取35，遗漏候选没有准入
    # 去向。仅阻止它继续作为新发布的已核准参照；历史输入与历史计算保持可读。
    # 完整范围重建后必须使用新的来源包，不修改旧文件或把数量上限改大冒充修复。
    for entry in dossier["entries"]:
        component = entry.get("enhanced_performance_calculation") or {}
        selection = component.get("inputs", {}).get("cohort", {}).get("selection_source", {})
        if selection.get("sha256") == "1f8728dadf879cc5900bee7c7dd1e3053a9b75de727c1bd1946ddd5b0abb65c0":
            raise ValueError("RATING_PUBLICATION_COHORT_TRUNCATED_REBUILD_REQUIRED")
    revision = "reviewed-partial-20261023"
    v1_entries = [entry for entry in dossier["entries"] if entry.get("evidence_v1_calculation")]
    if v1_entries:
        from app.services.fund_market_review import rating_review_required
        from app.services.fund_rating_evidence_preview import REVISION

        if len(v1_entries) != len(codes) or any(rating_review_required(code) for code in codes):
            raise ValueError("RATING_EVIDENCE_V1_PUBLICATION_SCOPE")
        revision = REVISION
    config = {
        "basis": BASIS,
        "revision": revision,
        "minimum_products": 1,
        "local_only": True,
        "independent_product_validation": False,
        "acceptance": "FROZEN_RESEARCH_SOURCE_ARITHMETIC_AND_EXPLICIT_USER_REVIEW",
        "approved_artifacts": {e["fund_code"]: e["research_archive"]["sha256"] for e in dossier["entries"]},
    }
    method_id = digest(config)
    if session.get(RatingMethodology, method_id) is None:
        session.add(
            RatingMethodology(
                methodology_id=method_id,
                family="MULTITYPE_REVIEWED",
                config=config,
                active=False,
                validation=None,
                created_at=now,
            )
        )
        session.flush()
    categories = []
    for entry in dossier["entries"]:
        code = entry["fund_code"]
        if not isinstance(code, str) or len(code) != 6 or not code.isascii() or not code.isdigit():
            raise ValueError("RATING_SNAPSHOT_CODE")
        bundle = deepcopy(entry)
        bundle["basis"] = BASIS
        doc = _read(entry["research_archive"], root)
        bundle["reviewed_document"] = _case(doc, code, entry["research_format"])
        if entry.get("money_input_archive"):
            bundle["money_inputs"] = _read(entry["money_input_archive"], root)
        validate_sources(bundle, now, root=root)
        computed = calculate(bundle, root=root)
        identity = entry["identity"]
        master_id = uuid5(NAMESPACE_URL, identity["manager_name"] + "/" + identity["legal_name"])
        catalog_binding = entry.get("existing_catalog_binding")
        if catalog_binding:
            # 早期目录可能用基金简称建立产品。以已核份额代码绑定原产品，
            # 保留其主键及名称；不能为修正显示名称重建产品或改动关注、持仓关系。
            existing_share = session.get(FundShareClass, code)
            existing_master = session.get(FundMaster, existing_share.fund_master_id) if existing_share else None
            if (
                not existing_master
                or str(existing_master.fund_master_id) != catalog_binding.get("fund_master_id")
                or existing_master.fund_name != catalog_binding.get("catalog_legal_name")
                or re.sub(r"\s+", "", existing_master.manager_name) != identity["manager_name"]
            ):
                raise ValueError("RATING_SNAPSHOT_CATALOG_BINDING_MISMATCH")
            master_id = existing_master.fund_master_id
        category = digest({"basis": BASIS, "product_id": str(master_id), "share": code})
        categories.append({"category": category, "eligible": True})
        lock_category(session, category)
        master = (
            existing_master
            if catalog_binding
            else session.scalar(
                select(FundMaster).where(
                    FundMaster.manager_name == identity["manager_name"], FundMaster.fund_name == identity["legal_name"]
                )
            )
        )
        if master is None:
            master = FundMaster(
                fund_master_id=master_id,
                fund_name=identity["legal_name"],
                manager_name=identity["manager_name"],
                fund_type=entry["family"],
                status="ACTIVE",
            )
            session.add(master)
            session.flush()
        share = session.get(FundShareClass, code)
        if share is None:
            share = FundShareClass(
                fund_code=code,
                fund_master_id=master.fund_master_id,
                share_class=entry["share_class"],
                fund_name=entry["public_name"],
                fund_type=entry["family"],
                status="ACTIVE",
                source_code="PUBLIC_DISCLOSURE",
                source_fund_code=code,
            )
            session.add(share)
            session.flush()
        elif share.fund_master_id != master.fund_master_id:
            raise ValueError("RATING_SNAPSHOT_EXISTING_PRODUCT_MISMATCH")
        cid = digest({"basis": BASIS, "code": code, "product": str(master.fund_master_id)})
        # 切换评分方法不等于改变法律产品分类。复用评价日已生效的分类，
        # 不新建重叠区间、不撤销旧身份，也不使历史评级失效。
        session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key,0))"), {"key": "rating-class:" + code})
        as_of = date.fromisoformat(entry["as_of_date"])
        existing_classes = session.scalars(
            select(RatingClassification).where(
                RatingClassification.fund_code == code,
                RatingClassification.revoked.is_(False),
                RatingClassification.effective_from <= as_of,
                or_(RatingClassification.effective_to.is_(None), RatingClassification.effective_to >= as_of),
            )
        ).all()
        if existing_classes:
            if len(existing_classes) != 1 or existing_classes[0].currency != "CNY":
                raise ValueError("RATING_SNAPSHOT_EXISTING_CLASSIFICATION_MISMATCH")
            cid = existing_classes[0].classification_id
        if session.get(RatingClassification, cid) is None:
            session.add(
                RatingClassification(
                    classification_id=cid,
                    fund_code=code,
                    product_id=str(master.fund_master_id),
                    category_code=category,
                    family=entry["family"],
                    currency="CNY",
                    eligibility="PUBLIC_LOCAL_ANALYSIS",
                    effective_from=date.fromisoformat(entry["as_of_date"]),
                    source_refs=[entry["original_report_archive"]["sha256"]],
                    verified_at=now,
                    revoked=False,
                )
            )
            session.flush()
        key = digest(bundle)
        if session.get(RatingEvidence, key) is None:
            session.add(
                RatingEvidence(
                    evidence_hash=key,
                    fund_code=code,
                    classification_id=cid,
                    as_of_date=date.fromisoformat(entry["as_of_date"]),
                    payload=bundle,
                    acquired_at=now,
                    revoked=False,
                )
            )
            session.flush()
        member = {"fund_code": code, "bundle": bundle, "evidence_hash": key}
        snapshot = {"method": config, "members": [member], "as_of_date": entry["as_of_date"]}
        batch_id, path = persist_input(root, snapshot)
        expiry = min(
            datetime.fromisoformat(entry["verified_at"]) + timedelta(days=31),
            datetime.combine(date.fromisoformat(entry["holding_date"]), datetime.min.time(), UTC) + timedelta(days=210),
        )
        if expiry <= now:
            raise ValueError("RATING_SNAPSHOT_EXPIRED")
        batch = session.get(RatingBatch, batch_id)
        if batch is None:
            batch = RatingBatch(
                batch_id=batch_id,
                category_code=category,
                methodology_id=method_id,
                as_of_date=date.fromisoformat(entry["as_of_date"]),
                frozen_at=now,
                input_hash=batch_id,
                sample_hash=digest([code]),
                input_path=path,
                input_snapshot=snapshot,
                member_count=1,
                product_count=1,
                status="COMPLETE",
                completed_at=now,
                withdrawn=False,
            )
            session.add(batch)
            session.flush()
            session.add(
                RatingMember(
                    batch_id=batch_id,
                    fund_code=code,
                    product_id=str(master.fund_master_id),
                    representative=True,
                    admitted=True,
                    reasons=[],
                    evidence_hash=key,
                )
            )
            session.add(
                RatingResult(
                    batch_id=batch_id,
                    fund_code=code,
                    status=computed["status"],
                    raw_score=computed["partial_score"],
                    score=computed["score"],
                    grade=None,
                    dimension_values={KEYS[k]: v["candidate"] for k, v in computed["dimensions"].items()},
                    public_evidence=public_evidence(bundle, computed),
                    valid_until=expiry,
                )
            )
            session.flush()
        out.append(
            {
                "fundCode": code,
                "batch": batch_id,
                "status": computed["status"],
                "partialScore": computed["partial_score"] if computed["status"] != "SCORED" else None,
                "pointCoverage": computed["point_weight"],
                "completeScore": computed["score"],
                "grade": None,
            }
        )
    method = session.get(RatingMethodology, method_id)
    if method is None:
        # 方法必须先于引用它的批次存在；调用端先建受控空方法记录，再完成真实类别列表。
        raise ValueError("RATING_SNAPSHOT_METHOD_REQUIRED")
    method.validation = {
        "review": reason,
        "validated_at": now.isoformat(),
        "report": {"categories": categories},
        "independent_product_validation": False,
    }
    method.active = True
    session.flush()
    for row in out:
        # 新规则明确切换只移动当前指针；旧批次仍可通过历史引用读取。
        # 必须匹配调用前冻结的批次，且原类别只含该份额，避免影响其他基金。
        for previous_id in dossier.get("replace_current_batches", {}).get(row["fundCode"], []):
            previous = session.get(RatingBatch, previous_id)
            if not previous:
                raise ValueError("RATING_SWITCH_PREVIOUS_MISSING")
            pointer = session.get(RatingCurrent, previous.category_code, with_for_update=True)
            if pointer is None:
                continue  # 相同切换重复执行时，旧指针已移除，保留幂等。
            if pointer.batch_id != previous_id:
                raise ValueError("RATING_SWITCH_PREVIOUS_CHANGED")
            previous_codes = set(
                session.scalars(select(RatingResult.fund_code).where(RatingResult.batch_id == previous_id))
            )
            if previous_codes != {row["fundCode"]} or previous_id == row["batch"]:
                raise ValueError("RATING_SWITCH_PREVIOUS_SCOPE")
            lock_category(session, previous.category_code)
            session.delete(pointer)
            session.add(
                RatingAudit(
                    audit_id=uuid5(NAMESPACE_URL, previous_id + row["batch"]).hex,
                    category_code=previous.category_code,
                    batch_id=previous_id,
                    action="METHOD_SWITCH",
                    reason=reason,
                    created_at=now,
                )
            )
            session.flush()
        publish(session, session.get(RatingBatch, row["batch"]))
    return {"methodology_id": method_id, "results": out, "independent_product_validation": False}
