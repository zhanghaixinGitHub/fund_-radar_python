"""新评分的可重复计算、逐份额发布与补件台账；所有输入来自正式不可变归档。

资料更新入口登记带摘要的资料清单，月度/单只任务消费清单并重算；没有新期
资料时保留原评价日期和缺项，不改核验时间制造更新。查询接口只读已发布批次。
"""

# 原始响应统一归档外接盘；历史路径只作为数据库别名，不生成C盘源文件。
import json
import re
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from time import monotonic

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.services import material_content as _sources
from app.services.fund_exposure_common import save
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_reviewed_snapshot import BASIS, calculate, validate_sources
from app.services.material_content import (
    content_digest,
    location_exists,
    same_content,
    source_bytes,
    source_exists,
    source_size,
    source_text,
)

logger = get_logger(__name__)


def archive_json(root: Path, value: dict) -> dict:
    """按真实UTF-8字节摘要追加归档；重复内容复用原文件，不覆盖已有来源。"""
    content = canonical(value).encode("utf-8")
    key = content_digest(content)
    relative = f"sources/pipeline/{key}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if location_exists(path):
        if not same_content(source_bytes(path), content):
            raise ValueError("RATING_PIPELINE_ARCHIVE_CHANGED")
    else:
        _sources.write_source(path, content)
    return {"file": relative, "sha256": key, "kind": "json"}


def _load(root: Path, ref: dict) -> dict:
    from app.services.fund_rating_reviewed_snapshot import _read

    value = _read(ref, root)
    if not isinstance(value, dict):
        raise ValueError("RATING_PIPELINE_JSON_REQUIRED")
    return value


def register_update(root: Path, dossier: dict) -> dict:
    """将已核新资料登记到有界队列；不发布、不推导缺失值、不接受外部页面分数。"""
    if dossier.get("basis") != BASIS or len(dossier.get("entries", [])) != 1:
        raise ValueError("RATING_PIPELINE_SINGLE_DOSSIER_REQUIRED")
    entry = dossier["entries"][0]
    code = entry["fund_code"]
    if len(code) != 6 or not code.isascii() or not code.isdigit():
        raise ValueError("RATING_PIPELINE_CODE")
    bundle = {**entry, "basis": BASIS}
    from app.services.fund_rating_reviewed_snapshot import _case

    bundle["reviewed_document"] = _case(_load(root, entry["research_archive"]), code, entry["research_format"])
    validate_sources(bundle, datetime.now(UTC), root=root)
    calculate(bundle, root=root)
    ref = archive_json(root, dossier)
    pointer = root / "pipeline" / "updates" / f"{code}.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    content = {"fund_code": code, "as_of_date": entry["as_of_date"], "dossier": ref}
    if location_exists(pointer):
        previous = json.loads(source_text(pointer, encoding="utf-8"))
        if previous.get("fund_code") != code:
            raise ValueError("RATING_PIPELINE_POINTER_IDENTITY")
        if date.fromisoformat(previous["as_of_date"]) > date.fromisoformat(entry["as_of_date"]):
            return {**content, "queued": False, "reason": "已有更新考察期资料，旧期只归档，不覆盖较新更新指针"}
    from app.services.fund_exposure_common import _publish

    _publish(pointer, canonical(content).encode("utf-8"), replace=True)
    return content


def staged_codes(root: Path, codes: list[str]) -> set[str]:
    """只探测本次真实目录，不扫描来源大目录或接受任意基金文件名。"""
    return {code for code in codes if source_exists(root / "pipeline" / "updates" / f"{code}.json")}


def current_inputs(session: Session, codes: list[str]) -> dict:
    rows = session.execute(
        text("""
        SELECT DISTINCT ON (r.fund_code) r.fund_code,b.batch_id,b.input_snapshot,b.as_of_date
        FROM fund_rating_current c JOIN fund_rating_batch b USING(batch_id)
        JOIN fund_rating_result r USING(batch_id)
        WHERE r.fund_code=ANY(:codes) AND b.status='PUBLISHED'
        ORDER BY r.fund_code,b.as_of_date DESC,b.completed_at DESC
    """),
        {"codes": codes},
    ).mappings()
    return {row["fund_code"]: dict(row) for row in rows}


def prepare_health_stock_liquidity_interval(session: Session, request: dict, *, root: Path) -> dict:
    """按本C份额原件准备流动性区间；保留其余七维、有效期及未知容量。"""
    from app.services.fund_rating_health_stock_liquidity_interval import prepare

    return prepare(session, request, root=root)


def prepare_gf_battery_liquidity_interval(session: Session, request: dict, *, root: Path) -> dict:
    """准备013180原件约束区间；只补最后一维，保留七维点值及原有效期。"""
    from app.services.fund_rating_gf_battery_liquidity_interval import prepare

    return prepare(session, request, root=root)


def prepare_active_portfolio_components(session: Session, request: dict, *, root: Path) -> dict:
    """仅准备004605已核原件的流动性；待复核的公司质量分档不随包采用。"""
    from app.services.fund_rating_active_portfolio_components import prepare

    return prepare(session, request, root=root)


def prepare_japan_active_execution(session: Session, request: dict, *, root: Path) -> dict:
    """按日本主动QDII本C份额的原合同、完整持仓与公开事项准备目标执行。"""
    from app.services.fund_rating_japan_active_execution import prepare

    return prepare(session, request, root=root)


def prepare_japan_return(session: Session, request: dict, *, root: Path) -> dict:
    """仅准备019449已核准日本主动参照的收益分项，其余维度原样保留。"""
    from app.services.fund_rating_japan_return import prepare

    return prepare(session, request, root=root)


def prepare_huaan_qdii_relative(session: Session, request: dict, *, root: Path) -> dict:
    """从同路线真实独立产品补华安本C收益及稳定性，保留风险和跟踪等缺项。"""
    from app.services.fund_rating_huaan_qdii_relative import prepare

    return prepare(session, request, root=root)


def prepare_huaan_qdii_benchmark(session: Session, request: dict, *, root: Path) -> dict:
    """从官方指数、实际汇率及定报精度核验补华安本C风险和目标执行。"""
    from app.services.fund_rating_huaan_qdii_benchmark import prepare

    return prepare(session, request, root=root)


def prepare_fund_013853_execution(session: Session, request: dict, *, root: Path) -> dict:
    """核三年持有期本A公开合同执行；费用场景、其余分项及有效期独立保留。"""
    from app.services.fund_rating_013853_execution import prepare

    return prepare(session, request, root=root)


def prepare_money_aggregate_components(session: Session, request: dict, *, root: Path) -> dict:
    """将货币自然日聚合原件接入正式部分包；不填缺日、不刷新旧五维有效期。"""
    from copy import deepcopy

    from app.services.fund_rating_money_aggregate_components import prepare_dossier
    from app.services.fund_rating_pipeline_prepare import canonical_source

    code = request.get("fund_code")
    previous = current_inputs(session, [code]).get(code) if code == "000009" else None
    if (
        previous is None
        or str(previous["as_of_date"]) != request.get("as_of_date")
        or previous["batch_id"] != request.get("expected_current_ref")
    ):
        raise ValueError("MONEY_AGGREGATE_CURRENT_PERIOD_OR_REF")
    raw = deepcopy(request["inputs"])
    if raw.get("fund_code") != code or raw.get("as_of_date") != request["as_of_date"]:
        raise ValueError("MONEY_AGGREGATE_REQUEST_IDENTITY")
    raw["sources"] = {key: canonical_source(root, ref) for key, ref in raw["sources"].items()}
    component_request = {
        "fund_code": code,
        "expected_current_ref": previous["batch_id"],
        "input_archive": archive_json(root, raw),
    }
    bundle = next(row["bundle"] for row in previous["input_snapshot"]["members"] if row["fund_code"] == code)
    return prepare_dossier(
        component_request,
        current_bundle=bundle,
        current_ref=previous["batch_id"],
        root=root,
        frozen=datetime.now(UTC),
        calculate_snapshot=calculate,
        validate_snapshot_sources=validate_sources,
    )


def blockers(bundle: dict, computed: dict) -> list[dict]:
    """按维列出当前实际不足；不把来源暂不可达解释为永远不存在。"""
    from app.services.fund_rating_reviewed_snapshot import KEYS, NAMES, PUBLIC_EXPLANATIONS

    out = []
    for key, value in computed["dimensions"].items():
        if value["candidate"] is not None:
            continue
        interval = Decimal(value["lower"]) > 0 or Decimal(value["upper"]) < 100
        out.append(
            {
                "dimension": KEYS[key],
                "name": NAMES[key],
                "state": "INTERVAL" if interval else "MISSING",
                "reason": bundle.get("public_explanations", {}).get(key, PUBLIC_EXPLANATIONS[key]),
                "sources": bundle.get("dimension_sources", {}).get(key, []),
                "recovery": "按本份额、本期资料补齐该项必要原始观察及适配参照后自动重算",
            }
        )
    return out


def update_new_rules(
    codes: list[str], *, engine, root: Path, as_of: date, progress=None, eligible_codes: set[str] | None = None
) -> dict:
    """逐只事务消费新资料并重算当前依据；单只异常保留旧结果并继续其余基金。

    已发布同一清单只核验、不重新发布；新月没有对应包时记录具体缺期，旧期不
    冒充新月。该函数由已有全局评级锁内调用，不新建定时循环或绕过事项复核。
    """
    from app.services.fund_market_review import rating_review_required
    from app.services.fund_rating_reviewed_snapshot import publish_reviewed

    if len(codes) > 5000:
        raise ValueError("RATING_PIPELINE_SCOPE_LIMIT")
    root = root.resolve()
    with Session(engine) as session:
        existing = current_inputs(session, codes)
    items, failures = [], []
    started = monotonic()
    for index, code in enumerate(codes, 1):
        item = {"fundCode": code, "status": "WAITING_DATA", "published": False, "blockers": []}
        try:
            # 本人30只的真实原件核验在现场超过三分钟，旧上限会让排序靠后
            # 的基金每轮都没有执行机会。保留十分钟界限，不跳过原件重算。
            if monotonic() - started > 600:
                item["blockers"].append({"state": "TASK_LIMIT", "reason": "本次处理达到时限，下次继续核验"})
                items.append(item)
                continue
            previous = existing.get(code)
            pointer = root / "pipeline" / "updates" / f"{code}.json"
            dossier = None
            if (not previous or previous["as_of_date"] <= as_of) and source_exists(pointer):
                if source_size(pointer) > 4096:
                    raise ValueError("RATING_PIPELINE_POINTER_SIZE")
                ref = json.loads(source_text(pointer, encoding="utf-8"))
                if ref.get("fund_code") != code:
                    raise ValueError("RATING_PIPELINE_POINTER_IDENTITY")
                observed = date.fromisoformat(ref["as_of_date"])
                if observed >= as_of or (observed.year, observed.month) == (as_of.year, as_of.month):
                    candidate_dossier = _load(root, ref["dossier"])
                    if len(candidate_dossier["entries"]) != 1 or candidate_dossier["entries"][0]["fund_code"] != code:
                        raise ValueError("RATING_PIPELINE_DOSSIER_IDENTITY")
                    if _period_covered(candidate_dossier["entries"][0], observed, as_of):
                        dossier = candidate_dossier
            if rating_review_required(code):
                item.update(
                    status="STALE",
                    blockers=[
                        {"state": "EVENT_REVIEW", "reason": "新增公告尚待核准对本基金的实际影响，保留原研究和历史分"}
                    ],
                )
            elif eligible_codes is not None and code not in eligible_codes:
                item["blockers"].append(
                    {"state": "CURRENT_MATERIAL_CHECK", "reason": "本次公告或持仓披露检查未完成，暂不追加评分"}
                )
            elif previous and previous["as_of_date"] > as_of:
                # 月度任务不拿旧月份的队列清单重做已发布的更新考察期。
                item.update(_evaluate(previous, code, root, as_of))
            elif (
                dossier is not None
                and previous
                and previous["as_of_date"] > date.fromisoformat(dossier["entries"][0]["as_of_date"])
            ):
                item.update(_evaluate(previous, code, root, as_of))
            elif dossier is not None:
                entry = dossier["entries"][0]
                # 比较实际冻结输入；相同资料不改核验日期、不制造新批次。
                previous_member = (
                    next((m for m in previous["input_snapshot"]["members"] if m["fund_code"] == code), None)
                    if previous
                    else None
                )
                old_bundle = previous_member.get("bundle", {}) if previous_member else {}
                from app.services.fund_rating_reviewed_snapshot import _case

                candidate = {
                    **entry,
                    "basis": BASIS,
                    "reviewed_document": _case(_load(root, entry["research_archive"]), code, entry["research_format"]),
                }
                if entry.get("money_input_archive"):
                    candidate["money_inputs"] = _load(root, entry["money_input_archive"])
                # 即使分数表未变，新增来源/身份/日期也必须核验，不能只比较研究文件。
                validate_sources(candidate, datetime.now(UTC), root=root)
                if old_bundle.get("evidence_v1_calculation") and not candidate.get("evidence_v1_calculation"):
                    from app.services.fund_rating_evidence_publication import dossier_for

                    dossier = dossier_for([candidate], root)
                    entry = dossier["entries"][0]
                    candidate = {**entry, "basis": BASIS, "reviewed_document": _load(root, entry["research_archive"])}
                calculate(candidate, root=root)
                same = old_bundle == candidate
                if not same:
                    with Session(engine) as session, session.begin():
                        session.execute(text("SET LOCAL statement_timeout='15s'"))
                        receipt = publish_reviewed(
                            session, dossier, "按用户授权的新评分流程核验真实原始资料并追加发布", root=root
                        )
                    item.update(published=True, publication=receipt["results"][0])
                    with Session(engine) as session:
                        previous = current_inputs(session, [code]).get(code)
                item.update(_evaluate(previous, code, root, as_of))
            elif previous:
                item.update(_evaluate(previous, code, root, as_of))
            else:
                item["blockers"].append({"state": "NO_INPUT", "reason": "尚无本份额已核准的新评分原始资料包"})
        except Exception as error:
            logger.exception(
                "fund_rating_pipeline.update_new_rules   >>>   单基金更新失败，保留历史, fund_code=%s", code
            )
            failures.append(code)
            item.update(
                status="FAILED",
                errorCode=str(error)[:180],
                blockers=[
                    {
                        "state": "VALIDATION_FAILED",
                        "reason": "本份额资料完整性或计算核验未通过，保留旧批次，失败证据已留档",
                    }
                ],
            )
        items.append(item)
        if progress:
            progress(index, len(codes), code, "已核验基金新评分和具体缺项")
    report = {
        "target": len(codes),
        "items": items,
        "failures": failures,
        "rated": sum(x["status"] in {"RATED", "SCORED"} for x in items),
        "partial": sum(x["status"] == "PARTIAL" for x in items),
        "notRated": sum(x["status"] not in {"RATED", "SCORED"} for x in items),
        "published": sum(x["published"] for x in items),
        "pending": sum(x["status"] == "STALE" for x in items),
    }
    # 任务台账是检查回执，独立于不可变评分输入；不会作为未来评分原件使用。
    report["checked_at"] = datetime.now(UTC).isoformat()
    receipt = root / "pipeline" / "runs" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f") + ".json")
    save(receipt, report)
    return report


def _period_covered(bundle: dict, observed: date, requested: date) -> bool:
    """自然月末休市时接受已核实的最后境内估值日；较新结果也不倒退。

    原评价日期、原输入和分数均不变。其他日期差异、境外估值政策和缺失年度
    仍须补齐，不能把任意同月旧数据当作月末资料。
    """
    if observed >= requested:
        return True
    if (
        (requested + timedelta(days=1)).day != 1
        or observed.month != requested.month
        or observed.year != requested.year
        or bundle.get("family") not in {"INDEX", "ACTIVE_EQUITY"}
        or re.search(r"QDII|海外|日本|越南|纳斯达克|原油", bundle.get("identity", {}).get("legal_name", ""), re.I)
    ):
        return False
    from app.services.direction_1d_protocol import calendar

    sessions, _ = calendar()
    if requested.year > sessions[-1].year:
        return False
    return max((day for day in sessions if day <= requested), default=None) == observed


def _evaluate(previous: dict | None, code: str, root: Path, as_of: date) -> dict:
    if previous is None:
        raise ValueError("RATING_PIPELINE_PUBLICATION_NOT_READ_BACK")
    member = next(m for m in previous["input_snapshot"]["members"] if m["fund_code"] == code)
    bundle = member["bundle"]
    if bundle.get("basis") != BASIS:
        return {
            "status": "WAITING_DATA",
            "blockers": [{"state": "NEW_INPUT_REQUIRED", "reason": "当前仍是早期量尺，需要新规则原始观察包后重算"}],
        }
    validate_sources(bundle, datetime.now(UTC), root=root)
    result = calculate(bundle, root=root)
    gaps = blockers(bundle, result)
    covered = _period_covered(bundle, previous["as_of_date"], as_of)
    if not covered:
        gaps.insert(
            0,
            {
                "state": "PERIOD_INPUT_MISSING",
                "reason": f"当前依据截至{previous['as_of_date']}，缺少{as_of}对应的新期资料，不沿用旧日期冒充本期",
            },
        )
    return {
        "status": result["status"] if covered else "WAITING_DATA",
        "requestedAsOfDate": str(as_of),
        "periodStatus": "NEWER_INPUT_ALREADY_PUBLISHED"
        if previous["as_of_date"] > as_of
        else "VALUATION_END_VERIFIED"
        if covered
        else "PERIOD_INPUT_MISSING",
        "ratingRef": previous["batch_id"],
        "asOfDate": str(previous["as_of_date"]),
        "score": result["score"],
        "pointWeight": result["point_weight"],
        "blockers": gaps,
    }
