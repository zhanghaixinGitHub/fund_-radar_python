"""未来输入只读适配：近期净值/分红来自既有库，公共指数每代码只请求一次并留原文。"""

import json
import re
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import urlsplit

import httpx
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_nav_preview_engine
from app.repositories import direction_1d as repo
from app.services import direction_1d_independent as protocol
from app.services import direction_1d_independent_audit as audit
from app.services.direction_1d_data import classify
from app.services.direction_1d_protocol import canonical, digest

FIELDS = ["ts_code", "trade_date", "close", "pre_close", "amount"]


def fetch_index(index: str, start: date, end: date) -> bytes:
    """单次有界index_daily查询；仅当前截止前的短窗口，不携带Token保存请求或错误。"""
    if not re.fullmatch(r"\d{6}\.(SH|SZ|CSI)", index) or not 0 <= (end - start).days <= 61:
        raise ValueError("INDEX_QUERY_RANGE_INVALID")
    if start.year < 2026 or end > protocol.now().date():
        raise ValueError("INDEX_QUERY_OUTSIDE_FORWARD_INPUT_RANGE")
    settings = get_settings()
    endpoint = urlsplit(settings.tushare_api_url)
    if (endpoint.scheme, endpoint.hostname, endpoint.path) != ("https", "api.tushare.pro", "") or any(
        (endpoint.query, endpoint.username, endpoint.password, endpoint.fragment, endpoint.port)
    ):
        raise ValueError("OFFICIAL_ENDPOINT_REQUIRED")
    token = settings.tushare_token.get_secret_value()
    if not token:
        raise ValueError("PROVIDER_TOKEN_MISSING")
    began = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream(
            "POST",
            settings.tushare_api_url,
            json={
                "api_name": "index_daily",
                "token": token,
                "params": {
                    "ts_code": index,
                    "start_date": start.strftime("%Y%m%d"),
                    "end_date": end.strftime("%Y%m%d"),
                },
                "fields": ",".join(FIELDS),
            },
        ) as response:
            if response.status_code != 200:
                raise ValueError("INDEX_PROVIDER_HTTP_FAILED")
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > 65536 or monotonic() - began > 30:
                    raise ValueError("INDEX_RESPONSE_LIMIT")
    if token.encode() in body:
        raise ValueError("INDEX_RESPONSE_CONTAINS_CREDENTIAL")
    payload = json.loads(body)
    if type(payload.get("code")) is not int or payload["code"] != 0:
        raise ValueError("INDEX_PROVIDER_BUSINESS_FAILED")
    return bytes(body)


def parse_index(raw: bytes, index: str, wanted: list[str]) -> list[dict]:
    payload = json.loads(raw)
    if payload.get("code") != 0 or payload.get("data", {}).get("fields") != FIELDS:
        raise ValueError("INDEX_RESPONSE_SCHEMA_CHANGED")
    found = {}
    for item in payload["data"]["items"]:
        if len(item) != len(FIELDS):
            raise ValueError("INDEX_RESPONSE_ROW_INVALID")
        row = dict(zip(FIELDS, item, strict=True))
        if not re.fullmatch(r"\d{8}", str(row["trade_date"])):
            raise ValueError("INDEX_RESPONSE_DATE_INVALID")
        day = row["trade_date"][:4] + "-" + row["trade_date"][4:6] + "-" + row["trade_date"][6:]
        if row["ts_code"] != index or day not in wanted or day in found:
            raise ValueError("INDEX_RESPONSE_IDENTITY_INVALID")
        if row["close"] is None or (index == "000300.SH" and row["amount"] is None):
            raise ValueError("INDEX_RESPONSE_VALUE_MISSING")
        found[day] = {
            "date": day,
            "close": str(row["close"]),
            "pre_close": str(row["pre_close"]),
            "amount": str(row["amount"]) if row["amount"] is not None else None,
        }
    if set(found) != set(wanted):
        raise ValueError("INDEX_HISTORY_INCOMPLETE")
    rows = [found[d] for d in wanted]
    for earlier, later in zip(rows, rows[1:], strict=False):
        if Decimal(earlier["close"]) != Decimal(later["pre_close"]):
            raise ValueError("INDEX_PRE_CLOSE_DISCONTINUITY")
    return rows


def current_source() -> dict:
    with get_nav_preview_engine().connect() as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        source = dict(repo.source(c))
        if not {"fund_nav", "fund_div", "index_daily"}.issubset(source["authorized_api_names"]):
            raise ValueError("SOURCE_CAPABILITY_MISSING")
        if not source["authorization_verified_at"] or source["rate_limit_per_minute"] <= 0:
            raise ValueError("SOURCE_NOT_VERIFIED")
    return json.loads(
        canonical(
            {
                k: source[k]
                for k in (
                    "source_id",
                    "enabled",
                    "retention_days",
                    "rate_limit_per_minute",
                    "authorization_verified_at",
                )
            }
        )
    )


def collect_market(root: Path, draft: dict, codes: list[str], target: str, *, query=fetch_index) -> dict:
    """同一目标日复用每个指数的首个成功响应；失败槽也留档，不无界自动重试。

    该方法由有界诊断调用。正式按时采集任务须在完整日历和运行验收通过后另行启用。
    """
    source = current_source()
    if source["source_id"] != draft["source"]["source_id"]:
        raise ValueError("SOURCE_CHANGED")
    if not 1 <= len(codes) <= 100 or not set(codes).issubset(draft["members"]):
        raise ValueError("MARKET_COLLECTION_SCOPE_INVALID")
    sessions = draft["sessions"]
    base, opened, deadline = protocol.bounds(sessions, target)
    if not opened <= protocol.now() < deadline:
        raise ValueError("MARKET_COLLECTION_WINDOW_CLOSED")
    i = sessions.index(base)
    indexes = sorted({"000300.SH", *[draft["members"][c]["index_code"] for c in codes]})
    series, received, persisted = {}, [], []
    for index in indexes:
        wanted = sessions[i - (20 if index == "000300.SH" else 5) : i + 1]
        request_path = root / (index + "-request.json")
        response_path = root / (index + "-response.json")
        if not response_path.exists():
            protocol.seal(
                request_path,
                {
                    "api": "index_daily",
                    "index": index,
                    "start": wanted[0],
                    "end": wanted[-1],
                    "at": protocol.now().isoformat(),
                },
            )
            try:
                raw = query(index, date.fromisoformat(wanted[0]), date.fromisoformat(wanted[-1]))
                at = protocol.now()
                rows = parse_index(raw, index, wanted)
                protocol.seal(
                    response_path,
                    {
                        "raw": raw.decode("utf-8"),
                        "received_at": at.isoformat(),
                        "rows": rows,
                        "amount_unit": "THOUSAND_CNY",
                        "unit_source": "https://tushare.pro/document/2?doc_id=95",
                    },
                )
                # 新读回完成后的真实时刻，不把请求发起时间当成接收/持久化证据。
                restored = protocol.unseal(response_path)
                if restored["rows"] != rows:
                    raise ValueError("MARKET_RESTORE_MISMATCH")
                protocol.seal(
                    root / (index + "-receipt.json"),
                    {"response_hash": digest(restored), "persisted_at": protocol.now().isoformat()},
                )
            except Exception as exc:
                protocol.seal(
                    root / (index + "-failure.json"),
                    {
                        "at": protocol.now().isoformat(),
                        "error_type": type(exc).__name__,
                        "status": "REQUEST_FAILED_NO_AUTOMATIC_RETRY",
                    },
                )
                raise
            sleep(max(0.31, 60 / source["rate_limit_per_minute"]))
        response = protocol.unseal(response_path)
        receipt = protocol.unseal(root / (index + "-receipt.json"))
        if response["amount_unit"] != "THOUSAND_CNY":
            raise ValueError("MARKET_AMOUNT_UNIT_CHANGED")
        if (
            digest(response) != receipt["response_hash"]
            or parse_index(response["raw"].encode(), index, wanted) != response["rows"]
        ):
            raise ValueError("MARKET_RESPONSE_CHANGED")
        series[index] = response["rows"]
        received.append(protocol.instant(response["received_at"]))
        persisted.append(protocol.instant(receipt["persisted_at"]))
    return {
        "series": series,
        "received_at": max(received).isoformat(),
        "persisted_at": max(persisted).isoformat(),
        "expires_at": (
            min(received) + timedelta(days=min(source["retention_days"], draft["source"]["retention_days"]))
        ).isoformat(),
        "amount_unit": "THOUSAND_CNY",
        "api_calls_reserved": len(indexes),
    }


def collect_nav(member: dict, draft: dict, target: str) -> dict:
    base, _, deadline = protocol.bounds(draft["sessions"], target)
    i = draft["sessions"].index(base)
    wanted = draft["sessions"][i - 60 : i + 1]
    with get_nav_preview_engine().connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        source = dict(repo.source(c))
        if "fund_div" not in source["authorized_api_names"] or str(source["source_id"]) != member["source_id"]:
            raise ValueError("NAV_SOURCE_UNAVAILABLE")
        profile = repo.profiles(c, [member["fund_code"]])
        if (
            len(profile) != 1
            or classify(profile[0])["product_family_id"] != member["family"]
            or classify(profile[0])["group_id"] != member["group"]
            or profile[0]["profile_hash"] != member["profile_hash"]
        ):
            raise ValueError("CURRENT_PROFILE_CHANGED")
        rows = repo.navs(
            c, member["fund_code"], source["source_id"], date.fromisoformat(wanted[0]), date.fromisoformat(base)
        )
        versions = [
            dict(r)
            for r in c.execute(
                text(
                    "SELECT content_hash,received_at,expires_at FROM direction_1d_source_version "
                    "WHERE fund_code=:code AND source_id=:source AND kind='NAV' "
                    "AND business_date BETWEEN :start AND :end"
                ),
                {
                    "code": member["fund_code"],
                    "source": source["source_id"],
                    "start": date.fromisoformat(wanted[0]),
                    "end": date.fromisoformat(base),
                },
            ).mappings()
        ]
        events = [
            dict(r)
            for r in c.execute(
                text(
                    "SELECT ex_date,nav_ex_date,content_hash FROM fund_dividend WHERE fund_code=:code "
                    "AND source_id=:source AND (ex_date=:target OR nav_ex_date=:target) ORDER BY content_hash LIMIT 101"
                ),
                {"code": member["fund_code"], "source": source["source_id"], "target": date.fromisoformat(target)},
            ).mappings()
        ]
        at = c.execute(text("SELECT clock_timestamp()")).scalar_one()
        if abs((protocol.now() - at).total_seconds()) > 5:
            raise ValueError("CLOCK_SKEW")
        if len(events) > 100 or at >= deadline:
            raise ValueError("EVENT_LIMIT_OR_DEADLINE")
        expires, version_evidence = nav_retention(
            rows,
            versions,
            str(source["source_id"]),
            min(source["retention_days"], draft["source"]["retention_days"]),
            at,
        )
    return json.loads(
        canonical(
            {
                "fund_code": member["fund_code"],
                "family": member["family"],
                "group": member["group"],
                "base": base,
                "target": target,
                "observed_at": at,
                "nav": [
                    {
                        "date": r["nav_date"],
                        "unit_nav": r["unit_nav"],
                        "ann_date": r["ann_date"],
                        "content_hash": r["content_hash"],
                        "updated_at": r["updated_at"],
                    }
                    for r in rows
                ],
                "source": {
                    "source_id": source["source_id"],
                    "enabled": source["enabled"],
                    "expires_at": expires,
                },
                "nav_source_versions": version_evidence,
                "events": events,
                "events_status": "KNOWN_EVENT" if events else "UNKNOWN",
                "market": None,
                "event_limitations": "仅已有授权现金分红记录；无记录不证明不存在拆分或其他事件。",
                "first_provider_publication_claimed": False,
            }
        )
    )


def nav_retention(rows: list[dict], versions: list[dict], source_id: str, days: int, at) -> tuple:
    """只读沿用已登记相同净值版本的首次留存起点；没有旧版本才记当前观察。

    不调用会INSERT的repo.observe，也不因复制到新试验目录而续期。后续正式采集
    还需用本地版本索引复用本阶段首次观察，本函数目前用于一次性的P0现场诊断。
    """
    known = {v["content_hash"]: v for v in versions}
    evidence = []
    for row in rows:
        h = digest(
            {
                "nav_date": str(row["nav_date"]),
                "unit_nav": str(row["unit_nav"]),
                "ann_date": str(row["ann_date"]) if row["ann_date"] else None,
                "upstream_hash": row["content_hash"],
                "source_id": source_id,
            }
        )
        previous = known.get(h)
        if previous and protocol.instant(previous["received_at"]) > at:
            raise ValueError("NAV_PREVIOUS_VERSION_TIME_INVALID")
        expires = (
            min(
                protocol.instant(previous["expires_at"]),
                protocol.instant(previous["received_at"]) + timedelta(days=days),
            )
            if previous
            else at + timedelta(days=days)
        )
        if expires <= at:
            raise ValueError("NAV_PREVIOUS_VERSION_EXPIRED")
        evidence.append(
            {
                "content_hash": h,
                "received_at": previous["received_at"] if previous else at,
                "expires_at": expires,
                "basis": "EXISTING_VERSION_NO_RENEWAL" if previous else "CURRENT_READ_ONLY_OBSERVATION",
            }
        )
    return min([at + timedelta(days=days), *[e["expires_at"] for e in evidence]]), evidence


def probe(root: Path, out: Path, market_evidence: Path | None = None) -> dict:
    """固定最小现场响应：首只共同基金和所需公共指数；不生成真实试验预测/成绩。"""
    state = audit.verify(root)
    draft = state["draft"]
    if out.exists():
        raise ValueError("PROBE_OUTPUT_ALREADY_EXISTS")
    live = audit.live_context(Path(draft["identity_evidence_path"]), Path(draft["core_env_path"]))
    if live["owner_binding_hash"] != draft["owner_binding_hash"] or live["scope_hash"] != draft["scope_hash"]:
        raise ValueError("OWNER_SCOPE_CHANGED_REAUDIT_REQUIRED")
    code = sorted(draft["members"])[0]
    target = draft["schedule"]["first_target"]
    out.mkdir(parents=True, exist_ok=False)
    market = collect_market(market_evidence or out / "market", draft, [code], target)
    snapshot = collect_nav(draft["members"][code], draft, target)
    snapshot["market"] = {**market, "mapping_hash": draft["members"][code]["mapping_hash"]}
    result = protocol.answers(snapshot, draft["sessions"], draft["members"][code], state["models"], protocol.now())
    protocol.seal(out / "input.json", snapshot)
    restored = protocol.answers(
        protocol.unseal(out / "input.json"), draft["sessions"], draft["members"][code], state["models"], protocol.now()
    )
    if restored != result:
        raise ValueError("DIAGNOSTIC_RESTORE_MISMATCH")
    evidence = {
        "status": "DIAGNOSTIC_ONLY_NOT_TRIAL_PREDICTION",
        "at": protocol.now().isoformat(),
        "fund_code": code,
        "target": target,
        "nav_rows": len(snapshot["nav"]),
        "index_rows": {k: len(v) for k, v in market["series"].items()},
        "input_hash": digest(snapshot),
        "answers": result,
        "new_fits": 0,
        "database_writes": 0,
        "trial_prediction_count": 0,
    }
    protocol.seal(out / "result.json", evidence)
    return {k: v for k, v in evidence.items() if k != "answers"}
