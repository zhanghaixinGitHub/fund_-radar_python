"""开发期续研的来源时间审查：隔离冲突报告，禁止生成或读取2026目标标签。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from pypdf import PdfReader

from scripts import fund_002112_existing_data_inputs_v1 as old
from scripts import fund_002112_holding_impact_fit_v1 as impact
from scripts import fund_002112_holding_impact_v1 as event
from scripts import fund_002112_next_day_search_v1 as previous

io = event.io
ROOT = io.RESEARCH / "development-only-optimization/20260930-v1"
LAST_LABEL_DATE = "2025-12-31"
ZONE = ZoneInfo("Asia/Shanghai")


def dev_lines(path: Path, field: str = "target") -> list[dict]:
    """先检查日期字面量，再解析该行；2026输入行不进入开发对象。"""
    result = []
    pattern = re.compile(r'"' + re.escape(field) + r'"\s*:\s*"(\d{4}-\d{2}-\d{2})"')
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = pattern.search(line)
            if not match:
                raise ValueError("ROW_DATE_NOT_FOUND")
            if match.group(1) <= LAST_LABEL_DATE:
                result.append(json.loads(line))
    return result


def restricted_nav(raw: dict) -> dict:
    """从混合年代的原始快照隔离开发期值；不生成2026标签或输出其净值。"""
    return {day: value for day, value in raw.items() if day <= LAST_LABEL_DATE}


def labels(rows: list[dict], nav: dict) -> list[str]:
    """标签入口硬限制日期，参数违规先拒绝，不能仅依靠调用者约定。"""
    if any(r["target"] > LAST_LABEL_DATE or r["base"] > LAST_LABEL_DATE for r in rows):
        raise ValueError("POST_2025_LABEL_ACCESS_FORBIDDEN")
    result = []
    for row in rows:
        before, after = Decimal(nav[row["base"]]["unit_nav"]), Decimal(nav[row["target"]]["unit_nav"])
        result.append("UP" if after > before else "DOWN" if after < before else "FLAT")
    return result


def cms_bound(report: dict) -> str:
    """CMS晚时间未获版本等价证明时，不解释成无害迁移；按晚约束隔离旧期使用。"""
    candidates = [report["available_at"]]
    for name in ("catalog_activation_ms", "catalog_publish_ms"):
        value = report.get("source", {}).get(name)
        if value:
            candidates.append(datetime.fromtimestamp(int(value) / 1000, ZONE).isoformat())
    return max(candidates)


def normalize_date(value: str) -> str:
    return value if "-" in value else f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def raw_quote_rows(path: Path) -> list[dict]:
    """只保留2025年底及以前行情；发布、修订时间缺失时不补造来源证据。"""
    payload = io.read(path)["data"]
    fields = payload["fields"]
    idx = fields.index("trade_date")
    return [
        dict(zip(fields, item, strict=True))
        for item in payload["items"]
        if normalize_date(item[idx]) <= LAST_LABEL_DATE
    ]


def audit(root: Path = ROOT) -> dict:
    """保全旧结果，在独立目录生成使用清单、来源证据、隔离输入及开发期标签快照。"""
    if (root / "timing-audit.json").exists():
        return io.read(root / "timing-audit.json")
    manifest = io.read(old.DEFAULT_ROOT / "source-manifest.json")
    by_hash = {s["sha256"]: old.DEFAULT_ROOT / s["snapshot_path"] for s in manifest["sources"]}
    source_checks = {}

    def checked(sha: str) -> Path:
        path = by_hash[sha]
        if sha not in source_checks:
            if io.sha(path) != sha:
                raise ValueError("SOURCE_HASH_CHANGED")
            source_checks[sha] = str(path)
        return path

    bundle = io.read(old.DEFAULT_ROOT / "snapshot/normalized-inputs.json")
    bundle["nav"] = restricted_nav(bundle["nav"])
    bundle["targets"] = [d for d in bundle["targets"] if d <= LAST_LABEL_DATE]
    provenance = io.read(old.PACKAGE / "input-provenance.json")
    accepted = io.read(old.DEFAULT_ROOT / "handoff/data/revisions/v2/inputs.json")
    original = {r["target_date"]: r for r in accepted["rows"] if r["target_date"] <= LAST_LABEL_DATE}
    rows = dev_lines(previous.ROOT / "inputs.jsonl")
    semantic = {r["target"]: r for r in dev_lines(event.ROOT / "feature-rows.jsonl")}
    io.save(
        root / "scope.json",
        {
            "at": io.now(),
            "phase": "SOURCE_AUDIT_THEN_BOUNDED_DEVELOPMENT_ONLY",
            "label_end": LAST_LABEL_DATE,
            "prediction_origin": "D_08:00_TO_NEXT_TRADING_DATE_U",
            "max_new_candidates": 36,
            "max_actual_fits_including_failures_and_repeats": 120,
            "no_new_collection": True,
            "no_2026_model_scores_or_predictions_read": True,
            "no_old_batch_overwrite": True,
            "no_model_activation": True,
        },
    )
    report_evidence = []
    reports = {}
    for report in bundle["reports"]:
        if report["published_date"] > LAST_LABEL_DATE:
            continue
        sha = report["raw"]["sha256"]
        reader = PdfReader(checked(sha))
        report_evidence.append(
            {
                "raw_sha256": sha,
                "raw_path": str(by_hash[sha]),
                "report_end": report["report_end"],
                "declared_publication": report["published_date"],
                "old_available_at": report["available_at"],
                "conservative_version_bound": cms_bound(report),
                "source": report["source"],
                "pdf_created": reader.metadata.get("/CreationDate"),
                "pdf_modified": reader.metadata.get("/ModDate"),
                "cover_publication_excerpt": reader.pages[0].extract_text()[-180:],
                "interpretation": "CMS_MIGRATION_POSSIBLE_BUT_SAME_VERSION_AT_OLD_DATE_UNPROVEN",
                "metadata_is_not_publication_proof": True,
            }
        )
        reports[sha] = report
    io.save(root / "report-time-evidence.json", report_evidence)

    # 净值值与既有官方原件逐项核对；公告日期来源仍是接入时的元数据，不冒称首次抓取历史。
    nav_sources = {}
    for day, value in bundle["nav"].items():
        sha = value["raw_sha256"]
        if sha not in nav_sources:
            nav_sources[sha] = {r["date"]: r for r in io.read(checked(sha))["dataList"] if r["date"] <= LAST_LABEL_DATE}
        raw = nav_sources[sha][day]
        assert raw["fundcode"] == "002112"
        assert Decimal(str(raw["netvalue"])) == Decimal(value["unit_nav"])
        assert value["available_at"] == old.available_at(value["ann_date"], value["source_published_at"])

    # 大盘及行业索引逐日还原原始响应，检查不同原件是否存在冲突版本。
    quote_evidence, conflicts = {}, []
    for category in ("market", "sector"):
        sources = (
            [src for meta in provenance["market"].values() for src in meta["sources"]]
            if category == "market"
            else io.read(old.PACKAGE / "sector-source-inputs.json")["sources"]
        )
        raw_map = {}
        seen = set()
        for source in sources:
            sha = source.get("raw_sha256") or source["sha256"]
            if sha in seen:
                continue
            seen.add(sha)
            for raw in raw_quote_rows(checked(sha)):
                key = (raw["ts_code"], normalize_date(raw["trade_date"]))
                fields = ("close", "pre_close", "pct_chg") if category == "market" else ("close", "pre_close")
                values = tuple(str(raw[k]) for k in fields)
                if key in raw_map and tuple(Decimal(v) for v in raw_map[key]) != tuple(Decimal(v) for v in values):
                    conflicts.append({"category": category, "key": key, "hash": sha})
                raw_map[key] = values
        normalized = bundle["markets" if category == "market" else "sectors"]
        checks = 0
        for code, days in normalized.items():
            for day, value in days.items():
                if day > LAST_LABEL_DATE:
                    continue
                fields = ("close", "pre_close", "pct_chg") if category == "market" else ("close", "pre_close")
                expected = tuple(Decimal(str(value[k])) for k in fields)
                assert expected == tuple(Decimal(v) for v in raw_map[(code, day)])
                checks += 1
        quote_evidence[category] = {
            "raw_sources_checked": len(seen),
            "rows_checked": checks,
            "historical_published_timestamp_present": False,
            "availability_assumption": "CLOSE_AVAILABLE_NEXT_TRADING_DATE_08:00",
            "revision_history_complete": False,
        }
    if conflicts:
        io.save(root / "raw-version-conflicts.json", conflicts)
        raise ValueError("MARKET_OR_SECTOR_VERSION_CONFLICT_REQUIRES_SEPARATE_QUARANTINE")

    stock_checks = 0
    for day, values in bundle["stocks"].items():
        if day > LAST_LABEL_DATE:
            continue
        meta = provenance["stock_dates"][day]
        raw = {r["ts_code"]: r for r in raw_quote_rows(checked(meta["raw_sha256"]))}
        for code, value in values.items():
            assert all(Decimal(str(value[k])) == Decimal(str(raw[code][k])) for k in value)
            stock_checks += 1
    quote_evidence["stock"] = {
        "rows_checked": stock_checks,
        "historical_publication_verified": False,
        "availability_assumption": "CLOSE_AVAILABLE_NEXT_TRADING_DATE_08:00",
        "decision": "H_QUARANTINED_BY_HOLDING_REPORT_VERSION_CONFLICT",
    }
    lineage, output = [], []
    counts_kept = [i for i, name in enumerate(impact.legacy_features.STATS) if not name.endswith("_related_20")]
    for row in rows:
        raw = original[row["base"]]
        assert row["as_of"] == raw["as_of"] == row["base"] + "T08:00:00+08:00"
        assert row["input_row_target"] == row["base"] < row["target"]
        cutoff = old.at0800(row["base"])
        end, lag, n = old.choose_nav_window(bundle["sessions"], bundle["nav"], row["base"])
        assert n is not None and np.allclose(n, row["groups"]["N"], rtol=0, atol=1e-13)
        idx = bundle["sessions"].index(row["base"])
        m, industry = [], []
        for code in old.MARKETS:
            ds = bundle["sessions"][idx - 5 : idx]
            q = [bundle["markets"][code][d] for d in ds]
            assert all(old.quote_available(v, d, bundle["sessions"], cutoff) for d, v in zip(ds, q, strict=True))
            returns = [float(v["pct_chg"]) / 100 for v in q]
            m.extend([returns[-1], math.prod(1 + v for v in returns) - 1])
        for code in old.SECTORS:
            for span in (1, 5, 20):
                ds = bundle["sessions"][idx - span : idx]
                q = [bundle["sectors"][code][d] for d in ds]
                assert all(old.quote_available(v, d, bundle["sessions"], cutoff) for d, v in zip(ds, q, strict=True))
                industry.append(float(math.prod(Decimal(v["close"]) / Decimal(v["pre_close"]) for v in q) - 1))
        assert np.allclose(m, row["groups"]["M"], rtol=0, atol=1e-13)
        assert np.allclose(industry, row["groups"]["I"], rtol=0, atol=1e-13)
        report = reports[raw["report"]["raw_sha256"]]
        # 当前开发期每行选用报告都有未解决的晚CMS时间，逐行验证后统一隔离依赖。
        assert cms_bound(report) > row["as_of"]
        independent = semantic[row["base"]]["morning"]["global"]
        out = {k: row[k] for k in ("target", "base", "as_of", "session_index", "label_mature_at")}
        out["groups"] = {
            "N": n,
            "M": m,
            "I": industry,
            "COUNTS": [row["groups"]["COUNTS"][i] for i in counts_kept],
            "GLOBAL": independent,
        }
        output.append(out)
        lineage.append(
            {
                "target": row["target"],
                "origin": row["base"],
                "as_of": row["as_of"],
                "nav_window_end": end,
                "nav_lag_sessions": lag,
                "last_market_and_stock_date": bundle["sessions"][idx - 1],
                "report_sha256": report["raw"]["sha256"],
                "report_bound": cms_bound(report),
                "removed_groups": ["H", "F", "HOLDING_CONTEXT", "WEIGHTED_SEMANTICS", "RELATED_COUNTS"],
                "nmi_recomputed_equal": True,
            }
        )
    io.save_lines(root / "input-time-lineage.jsonl", lineage)
    io.save_lines(root / "inputs.jsonl", output)
    io.save(root / "nav-through-2025.json", bundle["nav"])
    io.save(
        root / "source-checks.json",
        {
            "sources": source_checks,
            "quotes": quote_evidence,
            "nav_rows": len(bundle["nav"]),
            "raw_quote_conflicts": conflicts,
        },
    )
    inventory = {
        "N": {
            "columns": accepted["group_columns"]["N"],
            "used": True,
            "limitation": "ANN_DATE_RECONSTRUCTION_FIRST_SEEN_UNPROVEN",
        },
        "H": {
            "columns": accepted["group_columns"]["H"],
            "used": False,
            "reason": "LATE_CMS_REPORT_VERSION_NOT_HISTORICALLY_VERIFIED",
        },
        "M": {
            "columns": accepted["group_columns"]["M"],
            "used": True,
            "limitation": "CURRENT_VENDOR_REVISION_HISTORY_UNAVAILABLE",
        },
        "I": {
            "columns": accepted["group_columns"]["I"],
            "used": True,
            "limitation": "CURRENT_VENDOR_REVISION_HISTORY_UNAVAILABLE_AND_RESEARCH_SELECTED_SECTORS",
        },
        "F": {
            "columns": accepted["group_columns"]["F"],
            "used": False,
            "reason": "REPORT_DERIVED_STATE_SHARES_REPORT_TIME_CONFLICT",
        },
        "G_OLD": {
            "used": False,
            "reason": "OLD_OPTIONAL_CATALOG_WINDOW_FIELDS_DISABLED_NOT_A_BAN_ON_NEW_EVENT_FEATURES",
        },
        "COUNTS": {
            "columns": [impact.legacy_features.STATS[i] for i in counts_kept],
            "used": True,
            "removed": [v for v in impact.legacy_features.STATS if v.endswith("_related_20")],
            "limitation": "OBSERVED_CORPUS_COUNTS_NOT_COMPLETE_REAL_WORLD_EVENT_COUNTS",
        },
        "GLOBAL": {
            "columns": impact.GLOBAL_COLUMNS,
            "used": True,
            "limitation": "UNWEIGHTED_SOURCE_GROUNDED_FACTS_BODY_ONLY_TO_2023",
        },
        "OTHER_EXISTING": {
            "not_used": [
                "RAW_QUOTE_VOLUME_AND_LEVELS_BEYOND_REGISTERED_TRANSFORMS",
                "OTHER_FUNDS_REPORTS_AND_PEER_TRANSFER_EXPERIMENTS",
                "UNVERIFIED_FINANCIAL_AND_EVENT_PERIOD_FIELDS",
                "RAW_TITLES_AS_DENSE_TEXT_FEATURES",
            ],
            "reasons": [
                "BOUNDED_STAGE_NOT_ALL_POSSIBLE_TRANSFORMS",
                "NO_SHARED_POINT_IN_TIME_CROSS_FUND_TRAINING_CONTRACT_IN_THIS_STAGE",
                "SEMANTIC_OR_TIME_ADMISSION_INCOMPLETE",
                "NOT_IN_FIXED_DEVELOPMENT_HYPOTHESES_NO_NEW_LLM_EXTRACTION",
            ],
        },
    }
    io.save(root / "feature-use-inventory.json", inventory)
    result = {
        "at": io.now(),
        "rows": len(output),
        "report_versions_reviewed": len(report_evidence),
        "affected_input_rows": len(lineage),
        "old_input_strict_time_gate": "FAILED_REPORT_VERSION_CONFLICT",
        "repaired_input_gate": "PASS_FOR_RECONSTRUCTED_DEVELOPMENT_ONLY_AFTER_QUARANTINE",
        "strict_historical_first_seen_proven": False,
        "old_results_preserved": True,
        "nmi_recomputed_equal": True,
        "no_post_2025_labels_generated": True,
        "no_new_2026_scores": True,
        "available_groups": list(output[0]["groups"]),
        "body_nonzero_days_by_year": dict(Counter(r["target"][:4] for r in output if any(r["groups"]["GLOBAL"]))),
    }
    io.save(root / "timing-audit.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
