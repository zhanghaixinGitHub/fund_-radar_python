"""持续资料的覆盖层与版本留存；历史研究文件和训练协议只读复用。"""

from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, read, save

LIVE = ROOT / "materials-live"


def source_path(root: Path, relative: str) -> Path:
    """优先持续采集版本，不存在时读取已经核验的历史成果。"""
    live = root / "materials-live" / relative
    return live if live.exists() else root / relative


def source_files(root: Path, relative: str) -> list[Path]:
    files = {p.name: p for p in (root / relative).glob("*.json")}
    files.update({p.name: p for p in (root / "materials-live" / relative).glob("*.json")})
    return list(files.values())


def versioned_save(path: Path, value: dict) -> str:
    """当前索引可替换；替换前后完整值均按哈希留档，撤回也不删除原事实。"""
    for item in ([read(path)] if path.exists() else []) + [value]:
        # 请求缓存名本身已是 64 位哈希，不能再用它作目录叠加第二个哈希（Windows 超长路径）。
        evidence = {"source_file": path.name, "value": item}
        archive = path.parent / "versions" / (digest(evidence) + ".json")
        if not archive.exists():
            save(archive, evidence)
    return save(path, value, replace=True)


def merge_catalog(previous: list[dict], incoming: list[dict], key: str, date_key: str, windows: list) -> list:
    """仅完整重查窗口内的消失项标为目录暂未列出，不能据此断言法律意义上的撤回。"""
    received = {str(r[key]): r for r in incoming}
    result = {}
    for row in previous:
        identity = str(row[key])
        covered = any(start <= str(row.get(date_key, ""))[:10] <= end for start, end in windows)
        result[identity] = (
            {**row, "listing_status": "NOT_LISTED_ON_RECHECK"} if covered and identity not in received else row
        )
    for identity, row in received.items():
        old = result.get(identity, {})
        metadata = ("adjunctUrl", "adjunctSize", "announcementTitle", "announcementTime")
        changed = bool(old) and any(old.get(k) != row.get(k) for k in metadata)
        result[identity] = {
            **row,
            "listing_status": "LISTED",
            "source_changed": changed or old.get("source_changed", False),
        }
    return list(result.values())
