"""登记已逐项审阅的官方年历，并导出Java独立核验副本；绝不推测未公告休市日。"""

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from app.services import trading_calendar as calendars
from app.services.direction_1d_protocol import canonical, digest


def register(definition_file, evidence_file, manifest=None):
    manifest = manifest or calendars.EXTENSION_MANIFEST
    definition = calendars.CalendarDefinition.model_validate_json(Path(definition_file).read_bytes())
    calendar = calendars.build_calendar(definition)
    evidence = json.loads(Path(evidence_file).read_text(encoding="utf-8"))
    if evidence.get("reviewed_closed_ranges") != definition.model_dump(mode="json")["years"][0]["closed_ranges"]:
        raise ValueError("CALENDAR_CLOSED_RANGES_NOT_REVIEWED")
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if definition.reviewed_on > today or definition.years[0].announced_on > definition.reviewed_on:
        raise ValueError("CALENDAR_REVIEW_DATE_INVALID")
    if len(evidence.get("sources", [])) != 2:
        raise ValueError("TWO_OFFICIAL_ORIGINALS_REQUIRED")
    for url, original in zip(definition.years[0].sources, evidence["sources"], strict=True):
        path = Path(original["path"]).resolve()
        if path.drive.upper() != "D:" or original["url"] != url or not original.get("review_note", "").strip():
            raise ValueError("CALENDAR_ORIGINAL_IDENTITY_REQUIRED")
        if hashlib.sha256(path.read_bytes()).hexdigest() != original["sha256"]:
            raise ValueError("CALENDAR_ORIGINAL_HASH_MISMATCH")
    hosts = [urlparse(url).hostname or "" for url in definition.years[0].sources]
    if not any(h == "sse.com.cn" or h.endswith(".sse.com.cn") for h in hosts) or not any(
        h == "szse.cn" or h.endswith(".szse.cn") for h in hosts
    ):
        raise ValueError("CALENDAR_OFFICIAL_SOURCES_REQUIRED")
    # 独立目录也使用同一加载器，登记前验证现有链和官方域名。
    previous = calendars.EXTENSION_MANIFEST
    try:
        calendars.EXTENSION_MANIFEST = manifest
        chain = calendars.load_live_calendars()
        if definition.coverage_start != chain[-1].definition.coverage_end + timedelta(days=1):
            raise ValueError("CALENDAR_EXTENSION_NOT_CONTIGUOUS")
        root = manifest.parent
        root.mkdir(parents=True, exist_ok=True)
        name = definition.version.lower() + ".json"
        java_name = definition.version.lower() + "-java.json"
        for path in (root / name, root / java_name):
            if path.exists():
                raise ValueError("CALENDAR_IMMUTABLE_FILE_EXISTS")
        version = digest([c.content_hash for c in chain] + [calendar.content_hash])
        java_raw = canonical(
            {"version": version, "sessions": [str(d) for c in (*chain, calendar) for d in c.sessions]}
        ).encode()
        entry = {
            "definition_file": name,
            "definition_hash": calendar.content_hash,
            "calendar_version": version,
            "java_file": java_name,
            "java_sha256": hashlib.sha256(java_raw).hexdigest(),
            "evidence": evidence,
        }
        current = json.loads(manifest.read_bytes()) if manifest.exists() else {"extensions": []}
        (root / name).write_text(definition.model_dump_json(), encoding="utf-8")
        # 先只验证候选定义，再发布清单；验证失败不会把它加入运行日历。
        candidate = root / ".candidate-calendar-manifest.json"
        candidate.write_text(
            json.dumps({"extensions": current["extensions"] + [entry]}, ensure_ascii=False), encoding="utf-8"
        )
        calendars.EXTENSION_MANIFEST = candidate
        calendars.load_live_calendars()
        (root / java_name).write_bytes(java_raw)
        # 最后才原子切换清单，运行中的两端不会读到写了一半的日历目录。
        candidate.replace(manifest)
        return {"status": "REGISTERED", "year": definition.coverage_start.year, "calendar_version": version}
    finally:
        calendars.EXTENSION_MANIFEST = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.definition, args.evidence), ensure_ascii=False))


if __name__ == "__main__":
    main()
