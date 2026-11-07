"""本地迁移的真实业务验收；禁止把提取成功当作评分或预测已经验证。"""

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from app.db.session import get_engine
from app.services import material_content as materials
from app.services.fund_exposure_common import read as read_packet
from app.services.fund_rating import read as read_rating
from app.services.material_content import PROJECT, checksum, encoded, repository
from app.services.prediction_generation import current_predictions


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def code_hash():
    """验证和删除之间代码变化时必须重新验收，不能沿用旧凭据。"""
    digest = hashlib.sha256()
    paths = [
        p
        for name in ("app", "alembic", "tests")
        for p in (PROJECT / name).rglob("*")
        if p.is_file() and p.suffix in {".py", ".json"}
    ]
    for path in sorted(paths):
        digest.update(str(path.relative_to(PROJECT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validated_proof(directory):
    """校验记录和代码版本的完整性；回归记录本身不代表独立提取质量或删除资格。"""
    proof = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
    identity = checksum(encoded(proof["evidence"]))
    evidence = proof["evidence"]
    if identity != proof.get("verification") or evidence.get("passed") is not True:
        raise ValueError("MATERIAL_MIGRATION_NOT_VERIFIED")
    if evidence.get("code_unchanged") is not True or evidence.get("code_hash") != code_hash():
        raise ValueError("MATERIAL_CODE_CHANGED_REVERIFY_REQUIRED")
    tests = json.loads((directory / "pytest-result.json").read_text(encoding="utf-8"))
    if (
        tests.get("exit_code") != 0
        or not tests.get("code_unchanged")
        or tests.get("code_hash") != evidence["code_hash"]
    ):
        raise ValueError("MATERIAL_TESTS_NOT_VERIFIED_FOR_CURRENT_CODE")
    return proof


def certify(directory):
    """评分/预测回归凭据不再授权删除；独立提取质量验收标准另行审核。"""
    raise ValueError("MATERIAL_INDEPENDENT_EXTRACTION_REVIEW_REQUIRED")


def audit_original_access(blocked):
    """验收专用系统级读文件拦截，防止遗漏的旧读取绕过适配层偷偷使用原件。"""

    def audit(event, args):
        if event != "open" or not materials._database_only.get():
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        if materials.managed(path) and materials.original_location(path) and path.suffix.lower() not in {".py", ".pyc"}:
            blocked.append(str(path))
            raise PermissionError("MATERIAL_VALIDATION_ORIGINAL_OPEN_FORBIDDEN")

    sys.addaudithook(audit)


def _verify(directory):
    """先复算并留存必要页面，再在禁止原件访问的条件下重读并比较结果。"""
    initial_code_hash = code_hash()
    tests = json.loads((directory / "pytest-result.json").read_text(encoding="utf-8"))
    if tests.get("exit_code") != 0 or not tests.get("code_unchanged") or tests.get("code_hash") != initial_code_hash:
        raise ValueError("MATERIAL_TESTS_NOT_VERIFIED_FOR_CURRENT_CODE")
    blocked_opens = []

    audit_original_access(blocked_opens)
    evidence = {
        "at": datetime.now(UTC).isoformat(),
        "passed": False,
        "purpose": "STORAGE_REGRESSION_ONLY",
        "authorizes_deletion": False,
        "rating_reads": [],
        "prediction_reads": [],
        "historical_reads": [],
        "errors": [],
        "blocked_original_opens": blocked_opens,
    }
    with get_engine().connect() as connection:
        codes = (
            connection.execute(
                text(
                    "SELECT DISTINCT fund_code FROM fund_rating_result WHERE batch_id IN "
                    "(SELECT batch_id FROM fund_rating_current) ORDER BY fund_code"
                )
            )
            .scalars()
            .all()
        )
        rows = connection.execute(
            text(
                "SELECT fund_code,batch_id,status,score,grade,dimension_values,public_evidence "
                "FROM fund_rating_result ORDER BY fund_code,batch_id"
            )
        ).mappings()
        ratings = {(r["fund_code"], r["batch_id"]): canonical(dict(r)) for r in rows}
        rows = connection.execute(text("SELECT prediction_id,content_hash FROM fund_prediction_record")).all()
        predictions = {str(r[0]): r[1] for r in rows}
        prediction_codes = (
            connection.execute(text("SELECT DISTINCT fund_code FROM fund_prediction_record ORDER BY fund_code"))
            .scalars()
            .all()
        )
        history = connection.execute(
            text(
                "SELECT b.batch_id,min(r.fund_code) AS fund_code FROM fund_rating_batch b "
                "JOIN fund_rating_result r USING(batch_id) WHERE b.status='PUBLISHED' "
                "AND r.status IN ('RATED','SCORED','PARTIAL') AND b.batch_id NOT IN "
                "(SELECT batch_id FROM fund_rating_current) GROUP BY b.batch_id ORDER BY min(r.fund_code),b.batch_id"
            )
        )
        historical_cases = [dict(row) for row in history.mappings()]
    baseline = json.loads((directory / "rating-database-before.json").read_text(encoding="utf-8"))
    evidence["historical_ratings_preserved"] = all(
        ratings.get((r["fund_code"], r["batch_id"])) == canonical(r) for r in baseline
    )
    baseline = json.loads((directory / "prediction-database-before.json").read_text(encoding="utf-8"))
    evidence["historical_predictions_preserved"] = all(
        predictions.get(str(r["prediction_id"])) == r["content_hash"] for r in baseline
    )
    for index, code in enumerate(codes):
        try:
            with materials.original_materials():
                before = read_rating([code])
            # 单独执行资料库预读，捕获现有算法实际使用的页面排版，再关闭原件访问。
            prepared = read_rating([code])
            with materials.database_only():
                after = read_rating([code])
            if before != prepared or before != after:
                raise ValueError("RATING_DATABASE_ONLY_DIFFERENCE")
            item = after["items"][0]
            evidence["rating_reads"].append(
                {
                    "fund_code": code,
                    "status": item["status"],
                    "matched": True,
                    "result_hash": checksum(canonical(before).encode("utf-8")),
                }
            )
        except Exception as error:
            evidence["errors"].append(
                {
                    "fund_code": code,
                    "stage": "rating",
                    "error": type(error).__name__,
                    "reason": (str(error).splitlines() or [""])[0][:180],
                }
            )
        if index % 10 == 0:
            print(
                json.dumps(
                    {"rating_checked": index + 1, "rating_total": len(codes), "errors": len(evidence["errors"])}
                ),
                flush=True,
            )
        (directory / "verification-progress.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for code in prediction_codes:
        try:
            before = current_predictions(code)
            with materials.database_only():
                after = current_predictions(code)
            if before != after:
                raise ValueError("PREDICTION_DATABASE_ONLY_DIFFERENCE")
            evidence["prediction_reads"].append(
                {
                    "fund_code": code,
                    "records": len(after["predictions"]),
                    "matched": True,
                    "result_hash": checksum(canonical(after).encode("utf-8")),
                }
            )
        except Exception as error:
            evidence["errors"].append({"fund_code": code, "stage": "prediction", "error": type(error).__name__})
    # 每个历史批次会独立验证全部成员、八维值、总分及冻结证据；相同输入仅复算一次。
    # 这里以已发布的原计算结果为基准，不重新发布或更新历史记录。
    for index, case in enumerate(historical_cases):
        try:
            before = read_rating([case["fund_code"]], rating_ref=case["batch_id"])
            with materials.database_only():
                after = read_rating([case["fund_code"]], rating_ref=case["batch_id"])
            if before != after:
                raise ValueError("HISTORICAL_RATING_DATABASE_ONLY_DIFFERENCE")
            evidence["historical_reads"].append(
                {**case, "matched": True, "result_hash": checksum(canonical(after).encode("utf-8"))}
            )
        except Exception as error:
            evidence["errors"].append(
                {
                    **case,
                    "stage": "historical_rating",
                    "error": type(error).__name__,
                    "reason": (str(error).splitlines() or [""])[0][:180],
                }
            )
        if index % 20 == 0:
            print(
                json.dumps(
                    {
                        "historical_checked": index + 1,
                        "historical_total": len(historical_cases),
                        "errors": len(evidence["errors"]),
                    }
                ),
                flush=True,
            )
        (directory / "verification-progress.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # 校验实际交接的正文资料包，不生成或覆盖任何基金预测。
    material_count = 0
    handoffs = PROJECT / "data/fund-insights/market-workflow/comprehensive-inputs"
    from app.services.fund_comprehensive_materials import checked_documents

    for path in handoffs.glob("*.json"):
        try:
            handoff = read_packet(path)
            candidate = Path(handoff["materials_path"])
            before = read_packet(candidate)
            repository().ingest(candidate)
            with materials.database_only():
                after = read_packet(candidate)
                checked_documents(datetime.now(UTC), handoff["fund_code"], after)
            if before != after:
                raise ValueError("PREDICTION_MATERIAL_DIFFERENCE")
            material_count += 1
        except Exception as error:
            evidence["errors"].append(
                {
                    "stage": "prediction_material",
                    "path": str(path),
                    "error": type(error).__name__,
                    "reason": (str(error).splitlines() or [""])[0][:180],
                }
            )
    evidence["prediction_materials_checked"] = material_count
    evidence["rating_states"] = dict(Counter(r["status"] for r in evidence["rating_reads"]))
    evidence["passed"] = evidence["historical_ratings_preserved"] and evidence["historical_predictions_preserved"]
    evidence["passed"] = evidence["passed"] and not evidence["errors"] and not blocked_opens
    evidence["code_hash"] = code_hash()
    memo = materials._verification_cache.get()
    evidence["calculation_replays"] = {"executions": memo["executions"], "cache_hits": memo["hits"]} if memo else {}
    evidence["code_unchanged"] = evidence["code_hash"] == initial_code_hash
    evidence["passed"] = evidence["passed"] and evidence["code_unchanged"]
    proof = {"evidence": evidence, "verification": checksum(encoded(evidence))}
    (directory / "verification.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "passed": evidence["passed"],
        "ratings": len(evidence["rating_reads"]),
        "predictions": len(evidence["prediction_reads"]),
        "materials": material_count,
        "errors": evidence["errors"][:10],
        "verification": proof["verification"],
    }


def verify(directory):
    with materials.bulk_verification():
        return _verify(directory)


def verify_after_delete(directory):
    """新进程关闭原件补提后复算；与删除前原件对照摘要逐项相等才算迁移完成。"""
    proof = validated_proof(directory)
    evidence, errors = proof["evidence"], []
    blocked = []
    audit_original_access(blocked)
    checked = Counter()
    with materials.bulk_verification(), materials.database_only():
        for stage, cases in (
            ("rating", evidence["rating_reads"]),
            ("historical_rating", evidence["historical_reads"]),
            ("prediction", evidence["prediction_reads"]),
        ):
            for index, case in enumerate(cases):
                try:
                    if stage == "prediction":
                        value = current_predictions(case["fund_code"])
                    else:
                        value = read_rating(
                            [case["fund_code"]], **({"rating_ref": case["batch_id"]} if "batch_id" in case else {})
                        )
                    if checksum(canonical(value).encode("utf-8")) != case["result_hash"]:
                        raise ValueError("MATERIAL_POST_DELETE_RESULT_DIFFERENCE")
                    checked[stage] += 1
                except Exception as error:
                    errors.append(
                        {
                            "stage": stage,
                            "fund_code": case["fund_code"],
                            "error": type(error).__name__,
                            "reason": (str(error).splitlines() or [""])[0][:180],
                        }
                    )
                if index % 20 == 0:
                    print(
                        json.dumps({"post_delete_stage": stage, "checked": index + 1, "errors": len(errors)}),
                        flush=True,
                    )
    return {
        "passed": not errors and not blocked,
        "checked": dict(checked),
        "errors": errors,
        "blocked_original_opens": blocked,
        "verification": proof["verification"],
    }
