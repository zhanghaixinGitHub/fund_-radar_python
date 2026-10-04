"""002112 的 2024 年以来输入补证；只新增本批次文件，不修改业务代码和数据库。

只读查询净值日期、有效性和摘要，不查询或生成方向标签。所有外部读取先记请求，
原始响应独立留存；已有回执复用，失败请求不隐身。历史时间证据不足不提前日期。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx
from app.core.config import get_settings
from app.services.direction_1d_protocol import ZONE, digest
from sqlalchemy import create_engine, text

PY = Path(__file__).resolve().parents[1]
WEB = Path("C:/WebStormProject/workSpace05")
JAVA = Path("C:/ideaProject/workSpace12")
ROOT = PY / ".local-runs/fund-exposure-002112"
OUT = ROOT / "recent-input-completion/20260930-v1"
SECTORS = ("399998.SZ", "399986.SZ", "980017.SZ", "000819.SH", "980030.SZ")
NAV_DATES = ("2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31")


def now():
    return datetime.now(ZONE).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(value, dict) and set(value) == {"hash", "payload"}:
        if digest(value["payload"]) != value["hash"]:
            raise ValueError("INPUT_HASH_MISMATCH")
        return value["payload"]
    return value


def save(path, value):
    """排他新增，禁止覆盖原文件和本批次已完成步骤。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)


def engine():
    e = create_engine(
        get_settings().ai_database_url,
        hide_parameters=True,
        connect_args={
            "connect_timeout": 5,
            "options": "-c default_transaction_read_only=on -c statement_timeout=15000",
        },
    )
    if e.url.host not in {"localhost", "127.0.0.1"}:
        raise ValueError("LOCAL_READONLY_DATABASE_REQUIRED")
    return e


def runtime():
    """只读取模型摘要、日期和锁；不读取预测内容及封存答案。"""
    e = engine()
    with e.connect() as c:

        def rows(sql):
            return [dict(r) for r in c.execute(text(sql)).mappings()]

        result = {
            "at": now(),
            "locks": rows("select classid,objid,mode from pg_locks where locktype='advisory' and granted"),
            "active_jobs": rows(
                "select kind,state,count(*) as count from direction_1d_job "
                "where state in ('QUEUED','RUNNING') group by kind,state"
            ),
            "models": rows("select model_id::text,content_hash from direction_1d_model order by model_id"),
            "nav": rows(
                "select max(nav_date)::text as latest,count(*) as count from nav_daily where fund_code='002112'"
            ),
            "source": rows(
                "select source_code,enabled,authorized_api_names,retention_days,"
                "authorization_verified_at from source_registry where source_code='TUSHARE_PRO_FUND'"
            ),
        }
    e.dispose()
    return result


def protect():
    """备份三仓所有现有 Git 改动，冻结引用文件摘要及旧拟合/请求账本。"""
    if (OUT / "protection-before.json").exists():
        return read(OUT / "protection-before.json")
    files, repos = {}, []
    for name, repo in (("web", WEB), ("python", PY), ("java", JAVA)):

        def git(*args, repo=repo):
            return subprocess.check_output(["git", "-C", str(repo), *args])

        paths = set(git("diff", "--name-only", "-z").split(b"\0"))
        paths |= set(git("diff", "--cached", "--name-only", "-z").split(b"\0"))
        paths |= set(git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"))
        for rel in paths:
            if not rel:
                continue
            p = repo / rel.decode("utf-8")
            if p.is_file() and p != Path(__file__) and OUT not in p.parents:
                dest = OUT / "workspace-backup" / name / p.relative_to(repo)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                files[str(p)] = {"sha256": sha(p), "backup": str(dest)}
                if sha(dest) != files[str(p)]["sha256"]:
                    raise ValueError("WORKSPACE_CHANGED_DURING_BACKUP")
        repos.append(
            {
                "name": name,
                "path": str(repo),
                "head": git("rev-parse", "HEAD").decode().strip(),
                "status": git("status", "--porcelain=v1", "-uall").decode("utf-8"),
            }
        )
    ledger = ROOT / "information-research/20260928-v1"
    budget = read(ledger / "budget-authorized.json")
    fits = list((ledger / "fit-ledger").glob("*.json"))
    # 只做字节摘要；旧模型研究、原始资料和可能含标签的文件均不解析。
    protected = {}
    for folder in (ROOT / "reports", ROOT / "stock-days", ROOT / "indices", ledger / "fit-ledger"):
        for p in folder.glob("*.json"):
            protected[str(p)] = sha(p)
    for p in (
        ROOT / "report-result.json",
        ROOT / "supplement/stock-context.json",
        ROOT / "supplement/official-nav.json",
        ledger / "budget-authorized.json",
        ROOT / "data-completion/20260930-v1/acceptance-final-v3.json",
        WEB / "docs_zhx/implementation/fund-radar-implementation-plan-v1-2026-09-29.md",
    ):
        protected[str(p)] = sha(p)
    result = {
        "at": now(),
        "workspaces": repos,
        "dirty_files": files,
        "protected": protected,
        "old_body_requests": len(list((ROOT / "closure/20260929-v1/company-bodies/requests").glob("*.json"))),
        "cumulative_fits": budget["previous_actual_fits"] + len(fits),
        "new_fits": 0,
        "old_data_completion_ledger": str(ROOT / "data-completion/20260930-v1/request-ledger-final-v3.json"),
        "runtime": runtime(),
        "user_scope": "2024 onward source inputs only; no business-code changes",
    }
    save(OUT / "protection-before.json", result)
    save(
        OUT / "collection-contract.json",
        {
            "at": now(),
            "target_start": "2024-01-02",
            "target_end": "2026-09-29",
            "lookback_nav_start": "2023-09-28",
            "sector_codes": SECTORS,
            "maximum_http_requests": 40,
            "maximum_response_bytes": 16_000_000,
            "maximum_attempts_per_identical_request": 2,
            "no_paid_sources": True,
            "no_fit_or_registration": True,
            "old_budget_not_reset": True,
            "no_original_cache_overwrite": True,
            "no_labels_or_outcome_queries": True,
        },
    )
    print(
        json.dumps(
            {
                "protected_dirty_files": len(files),
                "protected_inputs": len(protected),
                "runtime": result["runtime"],
                "fits": result["cumulative_fits"],
            },
            ensure_ascii=False,
            default=str,
        )
    )
    return result


def fetch(url, *, params=None, api=None, fields=None, method="GET"):
    """有界公共读取；Token 只在内存请求体中，不进入磁盘、错误或输出。"""
    contract = read(OUT / "collection-contract.json")
    request = {"url": url, "params": params or {}, "api": api, "fields": fields, "method": method}
    key = digest(request)
    done = OUT / "receipts" / (key + ".json")
    if done.exists():
        r = read(done)
        p = OUT / r["file"]
        if sha(p) != r["sha256"]:
            raise ValueError("REUSED_RAW_HASH_MISMATCH")
        return p.read_bytes(), r
    from urllib.parse import urlparse

    host = urlparse(url).hostname
    allowed = {
        "api.tushare.pro",
        "www.dbfund.com.cn",
        "www.cninfo.com.cn",
        "static.cninfo.com.cn",
        "www.sse.com.cn",
        "query.sse.com.cn",
        "www.szse.cn",
        "disc.static.szse.cn",
        "www.csrc.gov.cn",
        "fund.eastmoney.com",
        "api.fund.eastmoney.com",
        "finance.sina.com.cn",
    }
    if host not in allowed or not url.startswith("https://"):
        raise ValueError("SOURCE_HOST_NOT_ALLOWED")
    if api:
        source = runtime()["source"][0]
        if not source["enabled"] or api not in source["authorized_api_names"] or source["retention_days"] <= 0:
            raise ValueError("EXISTING_PERMISSION_REQUIRED")
    previous = list((OUT / "requests").glob(key + "-*.json"))
    if len(previous) >= contract["maximum_attempts_per_identical_request"]:
        raise ValueError("IDENTICAL_REQUEST_STOPPED")
    if len(list((OUT / "requests").glob("*.json"))) >= contract["maximum_http_requests"]:
        raise ValueError("SCOPED_REQUEST_BUDGET_EXHAUSTED")
    attempt = len(previous) + 1
    save(OUT / "requests" / f"{key}-{attempt}.json", {**request, "at": now(), "attempt": attempt})
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.dbfund.com.cn/"}
    kwargs = {"params": params or {}} if method == "GET" else {"data": params or {}}
    if api:
        kwargs = {
            "json": {
                "api_name": api,
                "token": get_settings().tushare_token.get_secret_value(),
                "params": params or {},
                "fields": fields or "",
            }
        }
    try:
        time.sleep(0.35)
        with httpx.Client(timeout=httpx.Timeout(30, connect=7), follow_redirects=False, trust_env=True) as client:
            with client.stream(method, url, headers=headers, **kwargs) as response:
                status = response.status_code
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > contract["maximum_response_bytes"]:
                        raise ValueError("RESPONSE_TOO_LARGE")
                content_type = response.headers.get("content-type")
                # 重定向不自动跟随；保留目标供人工校验。
                location = response.headers.get("location")
        raw = bytes(body)
        h = hashlib.sha256(raw).hexdigest()
        p = OUT / "raw" / (h + ".bin")
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open("xb") as stream:
                stream.write(raw)
        receipt = {
            **request,
            "at": now(),
            "status": status,
            "content_type": content_type,
            "location": location,
            "file": str(p.relative_to(OUT)),
            "sha256": h,
            "bytes": len(raw),
        }
        save(OUT / "attempt-results" / f"{key}-{attempt}.json", receipt)
        if status != 200:
            raise ValueError("PUBLIC_HTTP_UNSUCCESSFUL")
        save(done, receipt)
        return raw, receipt
    except Exception as exc:
        error = OUT / "errors" / f"{key}-{attempt}.json"
        if not error.exists():
            save(
                error,
                {
                    "at": now(),
                    "request_hash": key,
                    "reason": type(exc).__name__,
                    "known_reason": str(exc) if type(exc) is ValueError else None,
                },
            )
        raise ValueError("PUBLIC_READ_FAILED_SEE_RECORDED_ATTEMPT") from None


def sectors():
    """补最近明确缺口，最多五个接口调用；校验身份、日期、数值和前收盘连续性留到总验收。"""
    if (OUT / "sector-acquisition.json").exists():
        return read(OUT / "sector-acquisition.json")
    result = []
    for code in SECTORS:
        try:
            raw, receipt = fetch(
                "https://api.tushare.pro",
                method="POST",
                api="index_daily",
                params={"ts_code": code, "start_date": "20260916", "end_date": "20260929"},
                fields="ts_code,trade_date,close,pre_close",
            )
            obj = json.loads(raw)
            if obj.get("code") != 0 or obj.get("data", {}).get("fields") != [
                "ts_code",
                "trade_date",
                "close",
                "pre_close",
            ]:
                raise ValueError("SECTOR_SOURCE_SCHEMA_FAILED")
            result.append({"code": code, "receipt": receipt, "rows": len(obj["data"]["items"]), "status": "RECEIVED"})
        except ValueError:
            result.append({"code": code, "status": "FAILED_SEE_REQUEST_LEDGER"})
    save(OUT / "sector-acquisition.json", {"at": now(), "results": result})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["protect", "sectors"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / ".active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "at": now(), "step": args.step}))
    try:
        {"protect": protect, "sectors": sectors}[args.step]()
    finally:
        lock.unlink()
