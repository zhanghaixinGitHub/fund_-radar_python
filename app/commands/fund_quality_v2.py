"""质量V2本机原件盘点、转换、资料集冻结及隔离试算；没有正式写库或激活入口。"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from pypdf.errors import PdfReadError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.fund_quality_v2_contract import metric_contract
from app.services.fund_quality_v2_materials import (
    build_cohort,
    convert_archived,
    convert_bundle,
    inspect_sources,
    metric_gaps,
)
from app.services.fund_rating_inputs import catalog, digest, provisional_family
from app.services.fund_rating_quality_v2 import VERSION, assess


def partial_materials(root: Path, as_of: str, frozen: datetime) -> dict:
    """复用已有核验清单；只承认原件目录，不拿旧分项或旧缺口作V2结论。"""
    pointer = root / "market-materials-current.json"
    if not pointer.exists():
        return {}
    ref = json.loads(pointer.read_text(encoding="utf-8"))
    path = (root / ref["file"]).resolve()
    if not path.is_relative_to(root.resolve() / "assessments") or path.stat().st_size > 20_000_000:
        raise ValueError("QUALITY_V2_PARTIAL_PATH_SIZE")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("QUALITY_V2_PARTIAL_HASH")
    data = json.loads(raw)
    if data["as_of_date"] != as_of:
        return {}
    if len(data["records"]) > 5000:
        raise ValueError("QUALITY_V2_PARTIAL_LIMIT")
    return {
        code: item
        for code, item in data["records"].items()
        if item["as_of_date"] == as_of and datetime.fromisoformat(item["verified_at"]) <= frozen
    }


def audit(
    *,
    output: Path,
    as_of: date | None = None,
    additional_source_roots: tuple[Path, ...] = (),
    structure_references: dict | None = None,
) -> dict:
    """同一只读事务查询运行时目录、正式结果和资料，不以旧快照字段判断原件覆盖。"""
    settings, frozen = get_settings(), now()
    if settings.app_environment != "local" or make_url(settings.ai_database_url).host not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("QUALITY_AUDIT_LOCAL_DATABASE_REQUIRED")
    engine = create_engine(
        settings.ai_database_url,
        isolation_level="REPEATABLE READ",
        connect_args={
            "options": "-c default_transaction_read_only=on -c statement_timeout=15000",
            "connect_timeout": 5,
        },
    )
    root = Path(settings.fund_insight_directory) / "ratings"
    try:
        with Session(engine) as session, session.begin():
            rows = catalog(session)
            formal = (
                session.execute(
                    text("""
                SELECT r.fund_code,r.status,r.grade,r.score,b.as_of_date,b.batch_id
                FROM fund_rating_current c JOIN fund_rating_batch b USING(batch_id)
                JOIN fund_rating_result r USING(batch_id) WHERE b.status='PUBLISHED'
            """)
                )
                .mappings()
                .all()
            )
            if not formal and as_of is None:
                raise ValueError("QUALITY_V2_AS_OF_REQUIRED_WITHOUT_FORMAL_BATCH")
            cutoff = str(as_of or max(r["as_of_date"] for r in formal))
            evidence = (
                session.execute(
                    text("""
                SELECT DISTINCT ON (fund_code) fund_code,evidence_hash,payload FROM fund_rating_evidence
                WHERE NOT revoked AND as_of_date=:cutoff ORDER BY fund_code,acquired_at DESC,evidence_hash
                LIMIT 5001
            """),
                    {"cutoff": cutoff},
                )
                .mappings()
                .all()
            )
        if len(formal) > 5000 or len(evidence) > 5000:
            raise ValueError("QUALITY_AUDIT_SCOPE_LIMIT")
        originals = {r["fund_code"]: r for r in evidence}
        current = {r["fund_code"]: dict(r) for r in formal}
        partial = partial_materials(root, cutoff, frozen)
        items = []
        for row in rows:
            code = row["fund_code"]
            original = originals.get(code)
            material = partial.get(code, {})
            # 粗分类仅决定盘点模板；不会据此认定产品、基准或新方法已支持。
            kind = {"ACTIVE_EQUITY": "ACTIVE_EQUITY", "INDEX": "PASSIVE_INDEX"}.get(provisional_family(row))
            bundle = original["payload"] if original else None
            refs = bundle["raw_archive"] if bundle else material.get("sources", [])
            admission_gaps = []
            additional_nav_refs = []
            additional_disclosure_refs = []
            additional_manager_refs = []
            if additional_source_roots and bundle:
                from app.services.fund_quality_v2_materials import HISTORICAL_SUPPLEMENT_ROLES, MANAGER_SUPPLEMENT_ROLES
                from app.services.fund_quality_v2_source_reuse import reuse_database_nav, reuse_originals

                # 旧完整来源包原样保留；规范净值、历史披露作为V2附件独立冻结。
                original_hashes = {ref["sha256"] for ref in bundle["raw_archive"]}
                original_labels = {ref["sha256"]: ref.get("label") for ref in bundle["raw_archive"]}
                current_primary_hashes = {bundle["report_hash"], bundle["summary_hash"]}
                for source_root in additional_source_roots:
                    try:
                        additional_nav_refs += reuse_database_nav(
                            code, source_root, root, frozen, date.fromisoformat(cutoff)
                        )
                        reused = reuse_originals(
                            code,
                            row["fund_name"],
                            source_root,
                            root,
                            date.fromisoformat(cutoff),
                            frozen,
                            admission_gaps,
                        )
                        # 日期证明齐全的新招募书作为历史附件补读，旧V1原件包不变。
                        # source_documents仅在原包缺书时使用，并重读目录和公告索引。
                        reused = [
                            {**ref, "label": "historical-prospectus"}
                            if ref.get("label") == "prospectus" and ref.get("publication_evidence") else ref
                            for ref in reused
                        ]
                        additional_disclosure_refs += [
                            ref
                            for ref in reused
                            if ref.get("label") in HISTORICAL_SUPPLEMENT_ROLES
                            and (
                                ref["sha256"] not in original_hashes
                                or not original_labels[ref["sha256"]]
                                and ref["sha256"] not in current_primary_hashes
                            )
                        ]
                        additional_manager_refs += [
                            ref for ref in reused
                            if ref.get("label") in MANAGER_SUPPLEMENT_ROLES and ref["sha256"] not in original_hashes
                        ]
                    except (ValueError, KeyError, OSError, TypeError) as exc:
                        admission_gaps.append(
                            {
                                "state": "SOURCE_ADMISSION_FAILED",
                                "source_root": str(source_root),
                                "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                            }
                        )
                additional_nav_refs = list({ref["sha256"]: ref for ref in additional_nav_refs}.values())
                additional_disclosure_refs = list({ref["sha256"]: ref for ref in additional_disclosure_refs}.values())
                additional_manager_refs = list({ref["sha256"]: ref for ref in additional_manager_refs}.values())
                refs = [*refs, *additional_nav_refs, *additional_disclosure_refs, *additional_manager_refs]
            if additional_source_roots and not bundle:
                from app.services.fund_quality_v2_source_reuse import reuse_originals

                # 只追加已有原件，主报告/概要仍以原有较新来源为先；不改其他会话文件。
                merged = {r["sha256"]: r for r in refs}
                for source_root in additional_source_roots:
                    try:
                        reused = reuse_originals(
                            code,
                            row["fund_name"],
                            source_root,
                            root,
                            date.fromisoformat(cutoff),
                            frozen,
                            admission_gaps,
                        )
                        for ref in reused:
                            merged.setdefault(ref["sha256"], ref)
                    except (ValueError, KeyError, OSError, TypeError) as exc:
                        admission_gaps.append(
                            {
                                "state": "SOURCE_ADMISSION_FAILED",
                                "source_root": str(source_root),
                                "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                            }
                        )
                refs = list(merged.values())
            sources = inspect_sources(refs, root, frozen)
            item = {
                "fund_code": code,
                "fund_name": row["fund_name"],
                "as_of_date": cutoff,
                "formal": current.get(code),
                "classification_state": "REQUIRES_ORIGINAL_VERIFICATION",
                "provisional_kind": kind,
                "product_id": None,
                "sources": sources,
                "source_admission_gaps": admission_gaps,
                "metrics": metric_gaps(kind, sources, bundle),
                "status": "RAW_MATERIALS_INCOMPLETE" if kind else "METHOD_UNSUPPORTED_OR_UNVERIFIED",
                # audit不消费校准包，不评定试算资格；None区别于实际核验后的0只。
                "trial_ready": None,
                "trial_readiness": "NOT_EVALUATED_NO_CALIBRATION",
                "raw_metrics_complete": False,
                "calibration_state": "REAL_PRODUCT_CALIBRATION_AND_INDEPENDENT_VALIDATION_REQUIRED",
            }
            if original:
                try:
                    if digest(bundle) != original["evidence_hash"]:
                        raise ValueError("QUALITY_V2_DATABASE_EVIDENCE_HASH")
                    packet = convert_bundle(
                        bundle,
                        root,
                        frozen,
                        additional_nav_refs=additional_nav_refs,
                        additional_disclosure_refs=additional_disclosure_refs,
                        additional_manager_refs=additional_manager_refs,
                        structure_reference=(structure_references or {}).get(code, {}),
                    )
                    path = output.parent / "packets" / (packet["hash"] + ".json")
                    versioned_save(path, packet)
                    item.update(
                        classification_state="REBUILT_FROM_ORIGINALS",
                        kind=packet["kind"],
                        product_id=packet["product_id"],
                        packet_path=str(path.resolve()),
                        packet_hash=packet["hash"],
                        sources=list(packet["sources"].values()),
                        metrics=packet["inventory"],
                        raw_metrics_complete=packet["raw_metrics_complete"],
                        benchmark_evidence=packet["benchmark_evidence"],
                        parsed_facts=packet.get("parsed_facts", {}),
                        document_errors=packet.get("document_errors", {}),
                    )
                except (ValueError, KeyError, OSError, TypeError, PdfReadError) as exc:
                    item["conversion_failure"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            elif refs:
                try:
                    packet = convert_archived(code, row["fund_name"], refs, root, date.fromisoformat(cutoff), frozen)
                    path = output.parent / "packets" / (packet["hash"] + ".json")
                    versioned_save(path, packet)
                    item.update(
                        metrics=packet["inventory"],
                        parsed_facts=packet["parsed_facts"],
                        document_errors=packet["document_errors"],
                        kind=packet["kind"],
                        product_id=packet["product_id"],
                        raw_metrics_complete=packet["raw_metrics_complete"],
                        packet_path=str(path.resolve()),
                        packet_hash=packet["hash"],
                    )
                except (ValueError, KeyError, OSError, TypeError, PdfReadError) as exc:
                    item["conversion_failure"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            item["status"] = "RAW_READY_CALIBRATION_REQUIRED" if item["raw_metrics_complete"] else item["status"]
            if "packet_hash" in item:
                item["classification_state"] = (
                    "REBUILT_FROM_ORIGINALS" if item.get("kind") else "METHOD_CURRENTLY_UNSUPPORTED"
                )
                if item.get("kind") is None:
                    item["status"] = "METHOD_CURRENTLY_UNSUPPORTED"
            items.append(item)
            print(
                json.dumps(
                    {
                        "fund_code": code,
                        "verified_metrics": sum(m["state"] == "VERIFIED_METRIC" for m in item["metrics"].values()),
                        "conversion_failure": item.get("conversion_failure"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return {
            "version": VERSION,
            "checked_at": frozen.isoformat(),
            "as_of_date": cutoff,
            "formal_ratings_changed": 0,
            "database_mode": "REPEATABLE_READ_READ_ONLY",
            "count": len(items),
            "formal_status_counts": dict(Counter((r.get("formal") or {}).get("status", "NO_RESULT") for r in items)),
            "formal_dates": sorted({str(r["as_of_date"]) for r in formal}),
            "trial_ready": None,
            "trial_readiness": "NOT_EVALUATED_NO_CALIBRATION",
            "raw_metrics_complete": sum(item["raw_metrics_complete"] for item in items),
            "reconstructed_products": len({i["product_id"] for i in items if i["product_id"]}),
            "metric_state_counts": dict(Counter(m["state"] for i in items for m in i["metrics"].values())),
            "contract": metric_contract(),
            "items": items,
        }
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "convert", "contract", "cohort", "experiment", "trial"))
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--packets", type=Path, help="资料集输入目录")
    parser.add_argument("--assignments", type=Path, help="已审核产品ID到calibration/validation的冻结映射")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--structure-references", type=Path, help="已归档指数及行业原件的份额到参考附件映射；仍逐件重读核验"
    )
    parser.add_argument(
        "--additional-source-root", type=Path, action="append", default=[], help="复用已存在的原件目录，不联网采集"
    )
    args = parser.parse_args()
    root = Path(get_settings().fund_insight_directory) / "ratings"
    references = read(args.structure_references) if args.structure_references else {}
    if not isinstance(references, dict) or len(references) > 5000:
        parser.error("结构参考附件清单必须是至多5000份额的对象")
    if args.action == "audit":
        result = audit(
            output=args.output,
            as_of=args.as_of,
            additional_source_roots=tuple(args.additional_source_root),
            structure_references=references,
        )
    elif args.action == "contract":
        result = metric_contract()
    elif args.action == "convert":
        if not args.bundle:
            parser.error("转换需要有原件目录的规范化事实包 --bundle")
        bundle = read(args.bundle)
        result = convert_bundle(bundle, root, now(), structure_reference=references.get(bundle["fund_code"], {}))
    elif args.action in {"cohort", "experiment"}:
        if not args.packets or not args.assignments:
            parser.error("资料集需要 --packets 和 --assignments；不能自动伪造产品分组")
        paths = sorted(args.packets.glob("*.json"))
        if len(paths) > 5000:
            parser.error("指标包范围超过5000")
        packets, assignments = [read(p) for p in paths], read(args.assignments)
        if args.action == "cohort":
            result = build_cohort(packets, assignments, root)
        else:
            if not args.as_of:
                parser.error("隔离校准需要明确 --as-of 校准截止日，验证产品须位于其后")
            from app.services.fund_quality_v2_research import experiment

            result = experiment(packets, assignments, args.as_of, root, now())
    else:
        if not args.packet:
            parser.error("试算需要可从原件重建的指标包 --packet")
        result = assess(read(args.packet), read(args.calibration) if args.calibration else None, root=root)
    versioned_save(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {"items", "source_bundle", "sources", "metrics", "inventory", "contract", "partial_observations"}
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
