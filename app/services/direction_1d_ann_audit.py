"""一日历史净值公告语义审计：本人范围公开数据只读、有限原始响应核对、零训练。"""

from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from time import monotonic, sleep

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine
from app.integrations.tushare import TushareFundClient, _to_fund_nav
from app.services import direction_1d_activity_study as activity
from app.services.direction_1d_protocol import ZONE, digest, input_days
from app.services.direction_1d_training import read, write_new
from app.services.direction_training_artifacts import file_hash
from app.services.tushare_fund_sync import _to_nav_daily_upsert

FIELDS = "ts_code,ann_date,nav_date,unit_nav,accum_nav,accum_div,net_asset,total_netasset,adj_nav"
START, END = "2024-09-27", "2024-10-09"
MAX_REQUESTS = 2
sector = activity.sector


def fingerprint() -> dict:
    result = activity.fingerprint()
    project = Path(__file__).resolve().parents[2]
    for name in (
        "app/services/direction_1d_ann_audit.py",
        "scripts/direction_1d_ann_audit.py",
        "app/integrations/tushare.py",
        "app/services/tushare_fund_sync.py",
        "app/repositories/fund_sync.py",
    ):
        result[name] = file_hash(project / name)
    return result


def validate_scope(scope: dict) -> list[str]:
    """只接受已核对本人账户的短时有效清单；不默认管理员，不扩大到市场目录。"""
    codes = scope["codes"]
    if (
        not 1 <= len(codes) <= 500
        or codes != sorted(set(codes))
        or any(len(c) != 6 or not c.isdigit() for c in codes)
        or scope["scope_hash"] != digest(codes)
        or scope["identity_basis"] != "EXACT_USER_CONFIRMED_ACTIVE_ACCOUNT"
    ):
        raise ValueError("ANN_AUDIT_SCOPE_INVALID")
    if not timedelta(0) <= datetime.now(ZONE) - datetime.fromisoformat(scope["verified_at"]) <= timedelta(hours=1):
        raise ValueError("ANN_AUDIT_SCOPE_STALE")
    return codes


def validate_source(source: dict) -> None:
    if (
        source["source_code"] != "TUSHARE_PRO_FUND"
        or not source["enabled"]
        or not source["authorization_verified_at"]
        or "fund_nav" not in source["authorized_api_names"]
        or source["retention_days"] <= 0
        or source["rate_limit_per_minute"] <= 0
    ):
        raise ValueError("ANN_AUDIT_SOURCE_UNAVAILABLE")


def collect(scope: dict) -> dict:
    """按本人全部关注、指定来源、2021—2024日期边界读取；事务和语句超时均显式限制。"""
    codes = validate_scope(scope)
    with get_engine().connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        c.execute(text("SET LOCAL statement_timeout='20s'"))
        source = dict(
            c.execute(
                text("""SELECT source_id,source_code,enabled,authorization_verified_at,
          authorized_api_names,retention_days,rate_limit_per_minute FROM source_registry
          WHERE source_code='TUSHARE_PRO_FUND'""")
            )
            .mappings()
            .one()
        )
        validate_source(source)
        params = {"codes": codes, "source": source["source_id"]}
        funds = [
            dict(r)
            for r in c.execute(
                text("""SELECT fund_code,fund_name,fund_type,source_code,source_fund_code
          FROM fund_share_class WHERE fund_code=ANY(:codes) ORDER BY fund_code"""),
                params,
            ).mappings()
        ]
        if [r["fund_code"] for r in funds] != codes:
            raise ValueError("ANN_AUDIT_FUND_IDENTITY_MISSING")
        navs = [
            dict(r)
            for r in c.execute(
                text("""SELECT fund_code,nav_date,ann_date,unit_nav,accumulated_nav,
          accumulated_dividend,net_asset,total_net_asset,adjusted_nav,content_hash,source_published_at,created_at,updated_at
          FROM nav_daily WHERE source_id=:source AND fund_code=ANY(:codes)
          AND nav_date BETWEEN '2021-01-01' AND '2024-12-31' ORDER BY fund_code,nav_date"""),
                params,
            ).mappings()
        ]
        versions = [
            dict(r)
            for r in c.execute(
                text("""SELECT fund_code,business_date,content_hash,received_at,stored_at,
          payload FROM direction_1d_source_version WHERE source_id=:source AND fund_code=ANY(:codes) AND kind='NAV'
          AND business_date BETWEEN '2021-01-01' AND '2024-12-31' ORDER BY fund_code,business_date,received_at"""),
                params,
            ).mappings()
        ]
        at = c.execute(text("SELECT clock_timestamp()")).scalar_one()
    return {
        "observed_at": at,
        "source": source,
        "scope": scope,
        "funds": funds,
        "navs": navs,
        "historical_versions": versions,
        "database_writes": 0,
        "protected_years_read": [],
        "api_calls": 0,
    }


def initialize(scope_path: Path, base: Path, plan: Path, root: Path, probes: list[str]) -> dict:
    scope = read(scope_path)
    codes = validate_scope(scope)
    if len(probes) != MAX_REQUESTS or len(set(probes)) != MAX_REQUESTS or not set(probes) <= set(codes):
        raise ValueError("ANN_AUDIT_PROBE_SCOPE_INVALID")
    snapshot = collect(scope)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "database.json", snapshot)
    activity.tree.copy_file(plan, root / "plan.md")
    # 原冻结历史与考题只复制必要证据，原研究目录及其指纹不变。
    files = (
        "baseline/history.json",
        "baseline/dataset.json",
        "baseline/models.json",
        "baseline/predictions.json",
        "mapping.json",
        "prices.json",
        "common-exam.json",
        "study.json",
    )
    for name in files:
        activity.tree.copy_file(base / name, root / "base" / name)
    inputs = {"base/" + n: file_hash(root / "base" / n) for n in files}
    inputs.update({n: file_hash(root / n) for n in ("database.json", "plan.md")})
    spec = {
        "at": datetime.now(ZONE).isoformat(),
        "kind": "ANNOUNCEMENT_SEMANTICS_AUDIT_ONLY",
        "fingerprint": fingerprint(),
        "input_files": inputs,
        "scope_hash": scope["scope_hash"],
        "probe_codes": probes,
        "probe_start": START,
        "probe_end": END,
        "max_requests": MAX_REQUESTS,
        "max_fits": 0,
        "database_writes": 0,
        "model_released": False,
        "source_expires_at": (
            datetime.fromisoformat(str(snapshot["observed_at"])) + timedelta(days=snapshot["source"]["retention_days"])
        ).isoformat(),
    }
    # 复用历史的留存期沿用原来源最早期限，不能因审计重新开始计时。
    spec["source_expires_at"] = min(
        datetime.fromisoformat(spec["source_expires_at"]),
        datetime.fromisoformat(read(base / "study.json")["source"]["source_expires_at"]),
    ).isoformat()
    write_new(root / "audit-spec.json", spec)
    write_new(root / "audit-spec-receipt.json", {"hash": digest(spec)})
    return {
        "scope_count": len(codes),
        "nav_rows": len(snapshot["navs"]),
        "historical_versions": len(snapshot["historical_versions"]),
        "max_requests": MAX_REQUESTS,
        "max_fits": 0,
    }


def verify_inputs(root: Path) -> dict:
    spec = read(root / "audit-spec.json")
    if (
        digest(spec) != read(root / "audit-spec-receipt.json")["hash"]
        or spec["fingerprint"] != fingerprint()
        or spec["max_fits"] != 0
        or spec["max_requests"] != MAX_REQUESTS
        or spec["probe_start"] != START
        or spec["probe_end"] != END
        or spec["model_released"] is not False
    ):
        raise ValueError("ANN_AUDIT_SPEC_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source_expires_at"]):
        raise ValueError("ANN_AUDIT_SOURCE_EXPIRED")
    for name, expected in spec["input_files"].items():
        f = root / name
        if not f.resolve().is_relative_to(root.resolve()) or f.is_symlink() or file_hash(f) != expected:
            raise ValueError("ANN_AUDIT_INPUT_CHANGED")
    return spec


def validate_probe(rows: list[dict], code: str, source_code: str) -> list[dict]:
    """保存原始字段后才沿既有适配器映射；拒绝错基金、越界、重复或不完整响应。"""
    if not source_code.startswith(code + ".") or not rows or len(rows) >= 20:
        raise ValueError("ANN_AUDIT_PROBE_IDENTITY_OR_LIMIT")
    result, seen = [], set()
    for raw in rows:
        nav = _to_fund_nav(raw)
        if (
            nav.ts_code != source_code
            or not START <= str(nav.nav_date) <= END
            or not nav.unit_nav.is_finite()
            or nav.unit_nav <= 0
        ):
            raise ValueError("ANN_AUDIT_PROBE_OUT_OF_SCOPE")
        if nav.nav_date in seen:
            raise ValueError("ANN_AUDIT_PROBE_DUPLICATE")
        seen.add(nav.nav_date)
        result.append(asdict(_to_nav_daily_upsert(code, nav)))
    return sorted(result, key=lambda r: r["nav_date"])


def client() -> TushareFundClient:
    settings = get_settings()
    return TushareFundClient(
        token=settings.tushare_token.get_secret_value(),
        api_url=settings.tushare_api_url,
        connect_timeout_seconds=5,
        read_timeout_seconds=20,
        max_retries=0,
        catalog_max_rows_per_query=20,
        nav_max_rows_per_query=20,
    )


def probe(root: Path) -> dict:
    spec = verify_inputs(root)
    if (root / "probe-completion.json").exists():
        verify_probes(root, spec)
        return {"status": "ALREADY_COMPLETED", "new_api_calls": 0}
    snapshot = read(root / "database.json")
    if datetime.now(ZONE) - datetime.fromisoformat(snapshot["observed_at"]) > timedelta(hours=1):
        raise ValueError("ANN_AUDIT_SOURCE_CHECK_STALE")
    # 实际出站前重新确认现有来源的启用、接口与留存，未购买新权限。
    with get_engine().connect() as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        current = dict(
            c.execute(
                text("""SELECT source_id,source_code,enabled,authorization_verified_at,
          authorized_api_names,retention_days,rate_limit_per_minute FROM source_registry
          WHERE source_code='TUSHARE_PRO_FUND'""")
            )
            .mappings()
            .one()
        )
    validate_source(current)
    if str(current["source_id"]) != snapshot["source"]["source_id"]:
        raise ValueError("ANN_AUDIT_SOURCE_ID_CHANGED")
    funds = {f["fund_code"]: f for f in snapshot["funds"]}
    requests = 0
    with client() as api:
        last = 0.0
        for code in spec["probe_codes"]:
            reserved, output = root / (code + "-reserved.json"), root / (code + "-probe.json")
            if reserved.exists() or output.exists():
                raise ValueError("ANN_AUDIT_REQUEST_ALREADY_RESERVED")
            fund = funds[code]
            source_code = fund["source_fund_code"]
            if fund["source_code"] != "TUSHARE_PRO_FUND" or not source_code or not source_code.startswith(code + "."):
                raise ValueError("ANN_AUDIT_SOURCE_FUND_CODE_UNKNOWN")
            sleep(
                max(
                    0,
                    60 / min(current["rate_limit_per_minute"], snapshot["source"]["rate_limit_per_minute"])
                    - (monotonic() - last),
                )
            )
            reservation = {
                "at": datetime.now(ZONE).isoformat(),
                "api": "fund_nav",
                "fields": FIELDS,
                "params": {
                    "ts_code": source_code,
                    "start_date": START.replace("-", ""),
                    "end_date": END.replace("-", ""),
                },
            }
            write_new(reserved, reservation)
            last = monotonic()
            requests += 1
            try:
                raw = list(api._query("fund_nav", params=reservation["params"], fields=FIELDS))
                mapped = validate_probe(raw, code, source_code)
            except Exception as error:
                write_new(
                    output,
                    {
                        "reservation": reservation,
                        "finished_at": datetime.now(ZONE).isoformat(),
                        "status": "FAILED",
                        "error_type": type(error).__name__,
                    },
                )
                raise ValueError("ANN_AUDIT_PROBE_FAILED_NO_RETRY") from None
            write_new(
                output,
                {
                    "reservation": reservation,
                    "finished_at": datetime.now(ZONE).isoformat(),
                    "status": "DOWNLOADED",
                    "raw_rows": raw,
                    "mapped_rows": mapped,
                },
            )
    result = {
        "at": datetime.now(ZONE).isoformat(),
        "api_calls": requests,
        "max_fits": 0,
        "files": {p.name: file_hash(p) for p in root.glob("*-probe.json")},
    }
    write_new(root / "probe-completion.json", result)
    return {"status": "COMPLETED", "new_api_calls": requests}


def verify_probes(root: Path, spec: dict) -> None:
    completion = read(root / "probe-completion.json")
    if completion["api_calls"] != MAX_REQUESTS or set(completion["files"]) != {
        c + "-probe.json" for c in spec["probe_codes"]
    }:
        raise ValueError("ANN_AUDIT_PROBE_BUDGET_CHANGED")
    if {p.name for p in root.glob("*-reserved.json")} != {c + "-reserved.json" for c in spec["probe_codes"]}:
        raise ValueError("ANN_AUDIT_PROBE_RESERVATION_CHANGED")
    for code in spec["probe_codes"]:
        name = code + "-probe.json"
        result, reserved = read(root / name), read(root / (code + "-reserved.json"))
        if (
            file_hash(root / name) != completion["files"][name]
            or result["reservation"] != reserved
            or result["status"] != "DOWNLOADED"
            or not datetime.fromisoformat(spec["at"])
            <= datetime.fromisoformat(reserved["at"])
            <= datetime.fromisoformat(result["finished_at"])
            <= datetime.fromisoformat(completion["at"])
        ):
            raise ValueError("ANN_AUDIT_PROBE_CHANGED")
        if digest(validate_probe(result["raw_rows"], code, reserved["params"]["ts_code"])) != digest(
            result["mapped_rows"]
        ):
            raise ValueError("ANN_AUDIT_MAPPING_CHANGED")


def is_quarter_end(d: str) -> bool:
    day = date.fromisoformat(d)
    return day.month in (3, 6, 9, 12) and (day + timedelta(days=1)).month != day.month


def summary_rows(rows: list[dict]) -> dict:
    delays = [
        (date.fromisoformat(r["ann_date"]) - date.fromisoformat(r["nav_date"])).days for r in rows if r["ann_date"]
    ]
    late = [
        r
        for r in rows
        if r["ann_date"] and (date.fromisoformat(r["ann_date"]) - date.fromisoformat(r["nav_date"])).days > 7
    ]
    return {
        "nav_count": len(rows),
        "ann_null": sum(r["ann_date"] is None for r in rows),
        "ann_before_nav": sum(d < 0 for d in delays),
        "ann_delay_gt_7_days": len(late),
        "late_at_quarter_end": sum(is_quarter_end(r["nav_date"]) for r in late),
        "late_with_net_asset": sum(r["net_asset"] is not None for r in late),
        "source_published_count": sum(r["source_published_at"] is not None for r in rows),
        "max_delay_days": max(delays, default=None),
    }


def exam_impact(base: Path) -> dict:
    """只量化旧公告规则的覆盖损失；不补题评分、不训练，也不宣称另一假设已证明。"""
    history, dataset, predictions = (
        read(base / n) for n in ("baseline/history.json", "baseline/dataset.json", "baseline/predictions.json")
    )
    lookup = {f["fund_code"]: {r["date"]: r for r in f["rows"]} for f in history["funds"]}
    by_key = {(r["fund_code"], r["t"]): r for r in dataset}
    available = sector.comparison.availability(dataset, history)
    rows, details = [], []
    for p in predictions:
        row = by_key[p["fund_code"], p["t"]]
        deadline = datetime.combine(date.fromisoformat(row["u"]), time(8), ZONE)
        if available[row["fund_code"], row["t"]] > deadline:
            blocked = [
                lookup[row["fund_code"]][str(d)]
                for d in input_days(date.fromisoformat(row["t"]))
                if (lookup[row["fund_code"]][str(d)]["ann_date"] or str(d)) > row["u"]
            ]
            details.append(
                {
                    "fund_code": row["fund_code"],
                    "t": row["t"],
                    "u": row["u"],
                    "blocking_nav": [{"nav_date": r["date"], "ann_date": r["ann_date"]} for r in blocked],
                }
            )
        rows.append(row)
    mapped, _ = sector.attach_specific(rows, read(base / "mapping.json")["funds"], read(base / "prices.json")["prices"])
    mapped_keys = {(r["fund_code"], r["t"]) for r in mapped}
    blocked25 = [r for r in details if (r["fund_code"], r["t"]) in mapped_keys]
    inherited = read(base / "common-exam.json")
    assert len(mapped) - len(blocked25) == len(inherited)
    day_counts = Counter(r["u"] for r in mapped)
    excluded_counts = Counter(r["u"] for r in blocked25)
    return {
        "original_29_exam": len(rows),
        "original_29_ann_rejected": len(details),
        "original_29_after_ann_gate": len(rows) - len(details),
        "mapped_25_before_ann_gate": len(mapped),
        "mapped_25_ann_rejected": len(blocked25),
        "mapped_25_inherited_exam": len(inherited),
        "fully_excluded_target_dates": sorted(d for d, n in day_counts.items() if excluded_counts[d] == n),
        "per_fund_rejected": dict(sorted(Counter(r["fund_code"] for r in details).items())),
        "blocked_questions": details,
        "new_predictions_created": 0,
        "new_fits": 0,
    }


def analyze(root: Path) -> dict:
    spec = verify_inputs(root)
    verify_probes(root, spec)
    snapshot = read(root / "database.json")
    navs = snapshot["navs"]
    current = {(r["fund_code"], r["nav_date"]): r for r in navs}
    if len(current) != len(navs):
        raise ValueError("ANN_AUDIT_DUPLICATE_DB_ROW")
    history = read(root / "base/baseline/history.json")
    mismatches = []
    frozen_count = 0
    for fund in history["funds"]:
        for old in fund["rows"]:
            frozen_count += 1
            now = current.get((fund["fund_code"], old["date"]))
            if (
                now is None
                or old["source_hash"] != now["content_hash"]
                or old["ann_date"] != now["ann_date"]
                or float(old["nav"]) != float(now["unit_nav"])
            ):
                mismatches.append({"fund_code": fund["fund_code"], "date": old["date"]})
    comparisons = []
    for code in spec["probe_codes"]:
        output = read(root / (code + "-probe.json"))
        for row in output["mapped_rows"]:
            db = current.get((code, row["nav_date"]))
            comparisons.append(
                {
                    "fund_code": code,
                    "nav_date": row["nav_date"],
                    "ann_date": row["ann_date"],
                    "unit_nav": row["unit_nav"],
                    "net_asset": row["net_asset"],
                    "same_database_content_hash": db is not None and row["content_hash"] == db["content_hash"],
                    "first_publication_proven": False,
                }
            )
    per_fund = {code: summary_rows([r for r in navs if r["fund_code"] == code]) for code in snapshot["scope"]["codes"]}
    return {
        "kind": "DATA_TIME_SEMANTICS_AUDIT_NOT_MODEL_OPTIMIZATION",
        "scope_count": len(per_fund),
        "scope_hash": spec["scope_hash"],
        "overall": summary_rows(navs),
        "per_fund": per_fund,
        "frozen_history_rows_checked": frozen_count,
        "frozen_db_mismatches": mismatches,
        "historical_version_count": len(snapshot["historical_versions"]),
        "probe_comparisons": comparisons,
        "exam_impact": exam_impact(root / "base"),
        "quarter_end_examples": [
            {k: r[k] for k in ("fund_code", "nav_date", "ann_date", "unit_nav", "net_asset", "source_published_at")}
            for r in navs
            if r["nav_date"] == "2024-09-30"
        ],
        "first_publication_proven": False,
        "raw_ann_date_rewritten": False,
        "database_writes": 0,
        "new_fits": 0,
        "new_forecasts": 0,
        "api_calls": MAX_REQUESTS,
        "model_released": False,
    }


def finish(root: Path) -> dict:
    result = analyze(root)
    write_new(root / "audit-result.json", result)
    write_new(root / "audit-result-receipt.json", {"at": datetime.now(ZONE).isoformat(), "hash": digest(result)})
    return {
        k: result[k]
        for k in (
            "scope_count",
            "overall",
            "frozen_history_rows_checked",
            "frozen_db_mismatches",
            "historical_version_count",
            "probe_comparisons",
            "new_fits",
            "api_calls",
        )
    }


def verify(root: Path) -> dict:
    result = analyze(root)
    if result != read(root / "audit-result.json") or digest(result) != read(root / "audit-result-receipt.json")["hash"]:
        raise ValueError("ANN_AUDIT_RESULT_CHANGED")
    return {
        "verified": True,
        "scope_count": result["scope_count"],
        "api_calls_during_verify": 0,
        "new_fits": 0,
        "first_publication_proven": False,
    }
