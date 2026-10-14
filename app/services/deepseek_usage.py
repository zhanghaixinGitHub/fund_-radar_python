"""私人模型调用台账：请求前原子预留每日额度，完成后记录真实token用量。"""

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.core.config import get_settings


class UsageLedger:
    """与资料解析库一起备份；不保存密钥、请求正文或个人持仓。"""

    def __init__(self, path: Path | None = None):
        self.path = path or Path(get_settings().fund_parse_cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS private_deepseek_call (
                request_key TEXT PRIMARY KEY, call_day TEXT NOT NULL, target_date TEXT NOT NULL,
                fund_code TEXT NOT NULL CHECK(fund_code='002112'), horizon INTEGER NOT NULL CHECK(horizon=1),
                stage TEXT NOT NULL, model TEXT NOT NULL, input_bytes_upper INTEGER NOT NULL,
                output_limit INTEGER NOT NULL, state TEXT NOT NULL, reserved_cny REAL NOT NULL,
                prompt_tokens INTEGER, completion_tokens INTEGER, cache_hit_tokens INTEGER,
                estimated_peak_cny REAL, response_id TEXT, started_at TEXT NOT NULL, finished_at TEXT
            )""")

    def reserve(self, key: str, scope, settings, stage: str, input_bytes: int, output_limit: int) -> None:
        """字节仅作保守费用预留，不声称是token；并发请求共享同一天的额度。"""
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        reserved = (input_bytes * 2 + output_limit * 8) / 1_000_000
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM private_deepseek_call WHERE request_key=?", (key,)).fetchone():
                raise ValueError("ANALYSIS_STAGE_FAILED")
            count, cost = connection.execute(
                "SELECT count(*),coalesce(sum(coalesce(estimated_peak_cny,reserved_cny)),0) "
                "FROM private_deepseek_call WHERE call_day=?",
                (str(now.date()),),
            ).fetchone()
            if count >= settings.max_calls_per_day or cost + reserved > settings.max_daily_cny:
                raise ValueError("DEEPSEEK_DAILY_BUDGET")
            connection.execute(
                """INSERT INTO private_deepseek_call
                (request_key,call_day,target_date,fund_code,horizon,stage,model,input_bytes_upper,
                 output_limit,state,reserved_cny,started_at) VALUES(?,?,?,?,?,?,?,?,?,'RUNNING',?,?)""",
                (
                    key,
                    str(now.date()),
                    scope.target_nav_date,
                    "002112",
                    1,
                    stage,
                    settings.model,
                    input_bytes,
                    output_limit,
                    reserved,
                    now.isoformat(),
                ),
            )

    def finish(self, key: str, usage: dict | None, response_id: str | None, state: str) -> dict:
        """按接口返回usage记录；无用量或结果未知时保留预留，不能伪造零费用。"""
        usage = usage or {}
        prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        hit = usage.get("prompt_cache_hit_tokens", 0)
        valid = all(isinstance(x, int) and x >= 0 for x in (prompt, output, hit)) and hit <= prompt
        # 当前高峰官方单价：命中输入0.04、其他输入2、输出8元/百万token。
        cost = ((prompt - hit) * 2 + hit * 0.04 + output * 8) / 1_000_000 if valid else None
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute(
                """UPDATE private_deepseek_call SET state=?,prompt_tokens=?,completion_tokens=?,
                cache_hit_tokens=?,estimated_peak_cny=?,response_id=?,finished_at=? WHERE request_key=?""",
                (
                    state,
                    prompt if valid else None,
                    output if valid else None,
                    hit if valid else None,
                    cost,
                    str(response_id or "")[:160],
                    datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    key,
                ),
            )
        return {
            "prompt_tokens": prompt if valid else None,
            "completion_tokens": output if valid else None,
            "cache_hit_tokens": hit if valid else None,
            "estimated_peak_cny": cost,
            "state": state,
        }
