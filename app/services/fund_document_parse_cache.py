"""公告、新闻、政策的不可变解析缓存；正式002112一日预测和本地验证共用契约。

数据库存解析事实，不存可跨日复用的预测方向。来源、正文、解析契约变化会产生
新键；基金仓位、行情、预测日期与模型名称不参与键，避免每天重复解析同一原文。
SQLite 文件由调用者指定，正式解析库随业务数据备份。校验通过仅说明来源和已覆盖
规则一致，不等于所有语义正确，更不代表允许发布预测。
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CONTRACT = "DOCUMENT_EVENTS_SOURCE_BOUND_V1"


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("PARSE_CACHE_TIMEZONE_REQUIRED")
    return parsed.astimezone(UTC)


def identity(document: dict, contract: str = CONTRACT) -> dict:
    """以实际原文识别版本；恢复截断正文也算输入版本变化，不能冒称已读全文。"""
    if document.get("kind") not in {"ANNOUNCEMENT", "NEWS", "POLICY"}:
        raise ValueError("PARSE_CACHE_DOCUMENT_KIND_INVALID")
    if not document.get("id") or not document.get("source_hash") or not document.get("body"):
        raise ValueError("PARSE_CACHE_SOURCE_IDENTITY_REQUIRED")
    return {
        "document_id": document["id"],
        "kind": document["kind"],
        "source_hash": document["source_hash"],
        "body_sha256": hashlib.sha256(document["body"].encode()).hexdigest(),
        "body_scope": document.get("body_scope"),
        "body_truncated": document.get("body_truncated"),
        "contract": contract,
    }


def source_issues(document: dict, result: dict) -> list[str]:
    """独立校验源身份和连续引句；不让模型自报通过作为缓存准入条件。"""
    issues = []
    if document.get("body_scope") != "FULL_TEXT" or document.get("body_truncated"):
        issues.append("PARTIAL_SOURCE_NOT_FULL_PARSE")
    if not isinstance(result, dict) or result.get("id") != document["id"]:
        return [*issues, "PARSED_DOCUMENT_ID_MISMATCH"]
    events = result.get("events")
    if not isinstance(events, list) or not events:
        return [*issues, "PARSED_EVENTS_REQUIRED"]
    for number, event in enumerate(events):
        if not isinstance(event, dict):
            issues.append(f"E{number}:EVENT_FORMAT_INVALID")
            continue
        if any(
            not isinstance(event.get(k), str) or not event[k]
            for k in ("title", "assessment", "stage", "meaning", "mechanism", "caveat")
        ):
            issues.append(f"E{number}:EVENT_FIELDS_REQUIRED")
        if event.get("assessment") not in {"POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL", "UNKNOWN"}:
            issues.append(f"E{number}:ASSESSMENT_INVALID")
        if event.get("stage") not in {"拟议", "进行中", "已发生", "历史经营", "程序事项", "不明"}:
            issues.append(f"E{number}:STAGE_INVALID")
        quotes = event.get("quotes")
        if not isinstance(quotes, list) or not quotes:
            issues.append(f"E{number}:QUOTES_REQUIRED")
        else:
            for i, quote in enumerate(quotes):
                if not isinstance(quote, str) or not quote or quote not in document["body"]:
                    issues.append(f"E{number}:Q{i}:QUOTE_NOT_IN_THIS_SOURCE")
    return list(dict.fromkeys(issues))


class DocumentParseCache:
    """一份来源版本只保存一次首次尝试和首次结果；失败、未知状态也保留。"""

    def __init__(self, path: Path, contract: str = CONTRACT):
        self.path, self.contract = Path(path), contract
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS parse_attempt (
                    cache_key TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    identity_json TEXT NOT NULL,
                    document_json TEXT NOT NULL,
                    input_cutoff_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    provenance_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS parse_attempt_document ON parse_attempt(document_id);
                CREATE TABLE IF NOT EXISTS parse_result (
                    cache_key TEXT PRIMARY KEY REFERENCES parse_attempt(cache_key),
                    state TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    checks_json TEXT NOT NULL,
                    result_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS parse_review (
                    review_key TEXT PRIMARY KEY,
                    cache_key TEXT NOT NULL REFERENCES parse_attempt(cache_key),
                    input_cutoff_at TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    issues_json TEXT NOT NULL,
                    receipt_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS parse_review_cache ON parse_review(cache_key,input_cutoff_at);
            """)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def key(self, document: dict) -> str:
        return digest(identity(document, self.contract))

    def lookup(self, document: dict, cutoff: str) -> dict:
        """先检查时间和元数据，再读取可复用结果；未来版本不进入历史输入。"""
        key, now = self.key(document), instant(cutoff)
        # 公告日期按北京时间记录；UTC截止时间跨午夜时不能直接截字符串比较日期。
        cutoff_day = now.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
        if instant(document["available_at"]) > now or document["date"] > cutoff_day:
            return {"state": "EXCLUDED", "key": key, "reason": "SOURCE_AFTER_CUTOFF"}
        with self.connection() as connection:
            attempt = connection.execute(
                "SELECT input_cutoff_at,provenance_json FROM parse_attempt WHERE cache_key=?", (key,)
            ).fetchone()
            if attempt is None:
                older = connection.execute(
                    "SELECT 1 FROM parse_attempt WHERE document_id=? LIMIT 1", (document["id"],)
                ).fetchone()
                return {
                    "state": "PENDING",
                    "key": key,
                    "reason": "SOURCE_OR_CONTRACT_CHANGED" if older else "NOT_PARSED",
                }
            if instant(attempt["input_cutoff_at"]) > now:
                return {"state": "EXCLUDED", "key": key, "reason": "PARSE_CONTEXT_AFTER_CUTOFF"}
            row = connection.execute("SELECT * FROM parse_result WHERE cache_key=?", (key,)).fetchone()
            reviews = connection.execute(
                "SELECT issues_json FROM parse_review WHERE cache_key=? AND input_cutoff_at<=?",
                (key, now.isoformat()),
            ).fetchall()
        if row is None:
            return {"state": "BLOCKED", "key": key, "reason": "FIRST_ATTEMPT_RUNNING_OR_OUTCOME_UNKNOWN"}
        if row["state"] != "SOURCE_CHECKED":
            return {"state": "BLOCKED", "key": key, "reason": row["state"], "issues": json.loads(row["checks_json"])}
        if reviews:
            return {
                "state": "BLOCKED",
                "key": key,
                "reason": "INDEPENDENT_REVIEW_WITHDREW_REUSE",
                "issues": list(
                    dict.fromkeys(issue for review in reviews for issue in json.loads(review["issues_json"]))
                ),
            }
        result = json.loads(row["result_json"])
        issues = source_issues(document, result)
        if digest(result) != row["result_hash"]:
            issues.append("CACHE_RESULT_HASH_MISMATCH")
        if issues:
            return {"state": "BLOCKED", "key": key, "reason": "CACHED_RESULT_RECHECK_FAILED", "issues": issues}
        archived_raw = json.loads(row["raw_json"])
        return {
            "state": "REUSE",
            "key": key,
            "reason": "SAME_SOURCE_AND_CONTRACT",
            "result": result,
            "provenance": json.loads(attempt["provenance_json"]),
            "event_selection": archived_raw.get("event_selection") if isinstance(archived_raw, dict) else None,
            "semantic_verified": False,
        }

    def claim(self, document: dict, cutoff: str, provenance: dict) -> bool:
        """唯一键原子抢占；两个进程不能同时为同一材料触发模型请求。"""
        if self.lookup(document, cutoff)["state"] != "PENDING":
            return False
        with self.connection() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO parse_attempt VALUES (?,?,?,?,?,?,?)",
                (
                    self.key(document),
                    document["id"],
                    canonical(identity(document, self.contract)),
                    canonical(document),
                    instant(cutoff).isoformat(),
                    datetime.now(UTC).isoformat(),
                    canonical(provenance),
                ),
            )
            return cursor.rowcount == 1

    def finish(
        self,
        document: dict,
        result: dict | None,
        raw: dict,
        issues: list[str] | None = None,
        failure: str | None = None,
    ) -> dict:
        """只插入首次结果；未知传输结果不重发，已知错误不伪装成可复用事实。"""
        checks = list(issues or [])
        if failure:
            if failure not in {"FAILED", "UNKNOWN"}:
                raise ValueError("PARSE_CACHE_FAILURE_STATE_INVALID")
            state = failure
        else:
            checks.extend(source_issues(document, result))
            state = "REJECTED" if checks else "SOURCE_CHECKED"
        checks = list(dict.fromkeys(checks))
        value = result or {}
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO parse_result VALUES (?,?,?,?,?,?,?)",
                (
                    self.key(document),
                    state,
                    datetime.now(UTC).isoformat(),
                    canonical(value),
                    canonical(raw),
                    canonical(checks),
                    digest(value),
                ),
            )
        return {"state": state, "issues": checks, "semantic_verified": False}

    def release_unstarted(self, document: dict) -> None:
        """仅解除当前尚未发出模型请求的占用，不删除任何首次响应或历史解析。

        配额或配置检查在HTTP请求前拒绝时，材料仍属于未解析的新版本；下次可继续。
        已保存结果及结果未知的调用均不能走此分支，避免失败后偷偷重发。
        """
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM parse_attempt WHERE cache_key=? AND NOT EXISTS "
                "(SELECT 1 FROM parse_result WHERE parse_result.cache_key=parse_attempt.cache_key)",
                (self.key(document),),
            )

    def results_count(self) -> int:
        with self.connection() as connection:
            return connection.execute("SELECT count(*) FROM parse_result").fetchone()[0]

    def reject_after_review(self, document: dict, issues: list[str], cutoff: str, receipt: dict) -> None:
        """追加撤销记录而不改写首次答案；更晚输入的审查不能倒灌旧回放。"""
        if not issues or not receipt:
            raise ValueError("PARSE_CACHE_REVIEW_RECEIPT_REQUIRED")
        key = self.key(document)
        review = {"cache_key": key, "issues": issues, "cutoff": instant(cutoff).isoformat(), "receipt": receipt}
        with self.connection() as connection:
            if not connection.execute("SELECT 1 FROM parse_result WHERE cache_key=?", (key,)).fetchone():
                raise ValueError("PARSE_CACHE_FIRST_RESULT_NOT_FOUND")
            connection.execute(
                "INSERT OR IGNORE INTO parse_review VALUES (?,?,?,?,?,?)",
                (
                    digest(review),
                    key,
                    review["cutoff"],
                    datetime.now(UTC).isoformat(),
                    canonical(issues),
                    canonical(receipt),
                ),
            )
