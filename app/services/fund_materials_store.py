"""持续资料的覆盖层与版本留存；历史研究文件和训练协议只读复用。"""

from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import PROJECT, ROOT, read, save
from app.services.material_content import location_exists, source_glob

LIVE = ROOT / "materials-live"


def source_path(root: Path, relative: str) -> Path:
    """优先持续采集版本，不存在时读取已经核验的历史成果。"""
    live = root / "materials-live" / relative
    return live if location_exists(live) else root / relative


def source_files(root: Path, relative: str) -> list[Path]:
    files = {p.name: p for p in source_glob(root / relative, "*.json")}
    files.update({p.name: p for p in source_glob(root / "materials-live" / relative, "*.json")})
    return list(files.values())


def versioned_save(path: Path, value: dict) -> str:
    """当前索引可替换；替换前后完整值均按哈希留档，撤回也不删除原事实。"""
    for item in ([read(path)] if location_exists(path) else []) + [value]:
        # 请求缓存名本身已是 64 位哈希，不能再用它作目录叠加第二个哈希（Windows 超长路径）。
        evidence = {"source_file": path.name, "value": item}
        archive = path.parent / "versions" / (digest(evidence) + ".json")
        if len(str(archive.resolve())) > 210:
            # 新输入版本目录自身已有64位摘要，再套一层版本归档及原子临时名
            # 会超过Windows路径上限。仅将过长的历史副本放入短内容寻址目录，
            # 权威当前路径不变；完整来源路径参与摘要，避免同名基金文件混淆。
            evidence["source_path"] = str(path.resolve())
            archive = PROJECT / ".local-runs" / "material-version-archives" / (digest(evidence) + ".json")
        if not location_exists(archive):
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
