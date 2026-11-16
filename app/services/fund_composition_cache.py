"""公共构成落库；源内容变化立即失效，同版本不重复解析正文或核对关系。"""

import hashlib
import json
from contextvars import ContextVar
from pathlib import Path
from threading import Lock

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine

_context = ContextVar("composition_sources", default=None)
_lock = Lock()
VERSION = "COMPOSITION_DB_V1"


def active():
    return _context.get() is not None


def _identity(path):
    from app.services.material_content import source_bytes

    try:
        raw = source_bytes(Path(path))
    except FileNotFoundError:
        return None
    return hashlib.sha256(raw).hexdigest()


def observe(path):
    context = _context.get()
    key = str(Path(path).resolve())
    if context is not None and key not in context["sources"]:
        context["sources"][key] = _identity(key)


def read(path):
    from app.services.fund_exposure_common import read as verified_read

    context = _context.get()
    if context is None:
        return verified_read(path)
    key = str(Path(path).resolve())
    observe(path)
    if key not in context["values"]:
        context["values"][key] = verified_read(path)
    return context["values"][key]


def _valid(sources):
    # 比较实际字节摘要，不能只靠修改时间或“持仓没变”判断资料仍然有效。
    return bool(sources) and all(_identity(path) == identity for path, identity in sources.items())


def batch(codes):
    from app.services import fund_composition

    if (
        not codes
        or len(codes) > 100
        or len(set(codes)) != len(codes)
        or any(not isinstance(code, str) or not fund_composition.re.fullmatch(r"[0-9]{6}", code) for code in codes)
    ):
        raise ValueError("COMPOSITION_FUND_CODES_INVALID")
    settings = get_settings()
    scope = [
        VERSION,
        str(Path(settings.fund_material_directory).resolve()),
        str(Path(settings.fund_insight_directory).resolve()),
        sorted(codes),
    ]
    key = hashlib.sha256(json.dumps(scope).encode()).hexdigest()
    # 合并并发的首次构建；不缓存个人持仓、金额或收益。
    with _lock:
        engine = get_engine()
        with engine.connect() as connection:
            stored = connection.execute(
                text("SELECT payload FROM fund_composition_cache WHERE cache_key=:key"), {"key": key}
            ).scalar()
        if isinstance(stored, str):
            stored = json.loads(stored)
        if stored is not None and _valid(stored["sources"]):
            result = stored["result"]
            by_code = {row["fundCode"]: row for row in result["funds"]}
            return {**result, "funds": [by_code[code] for code in codes]}
        context = {"sources": {}, "values": {}}
        token = _context.set(context)
        try:
            root = Path(settings.fund_insight_directory) / "market-workflow"
            observe(fund_composition.PROJECT / "app/data/fund_security_themes_v1.json")
            for code in codes:
                observe(root / "comprehensive-inputs" / f"{code}.json")
                observe(root / "materials" / f"{code}.json")
            # 分类表也绑定实际版本，不能继续使用进程启动后的旧分类。
            fund_composition._theme_registry.cache_clear()
            result = fund_composition.batch(codes)
            if not _valid(context["sources"]):
                raise ValueError("COMPOSITION_SOURCES_CHANGED_DURING_READ")
            stored = {"sources": context["sources"], "result": result}
            with engine.begin() as connection:
                payload_value = "CAST(:payload AS jsonb)" if engine.dialect.name == "postgresql" else ":payload"
                connection.execute(
                    text(f"""INSERT INTO fund_composition_cache(cache_key,payload) VALUES(:key,{payload_value})
                    ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload,created_at=CURRENT_TIMESTAMP"""),
                    {"key": key, "payload": json.dumps(stored, ensure_ascii=False)},
                )
            return result
        finally:
            _context.reset(token)
