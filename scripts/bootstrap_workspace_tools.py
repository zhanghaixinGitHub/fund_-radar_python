"""离线复制本项目需要的既有运行工具到D盘；不下载、不改写或删除C盘共享安装。"""

import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]


def imports(path):
    """读取PE导入表，只复制Python标准扩展实际需要的非系统DLL，避免复制整套Anaconda。"""
    raw = path.read_bytes()
    if raw[:2] != b"MZ":
        return []
    pe = struct.unpack_from("<I", raw, 0x3C)[0]
    count, optional_size = struct.unpack_from("<H", raw, pe + 6)[0], struct.unpack_from("<H", raw, pe + 20)[0]
    optional = pe + 24
    directory = optional + (112 if struct.unpack_from("<H", raw, optional)[0] == 0x20B else 96)
    rva = struct.unpack_from("<I", raw, directory + 8)[0]
    sections = optional + optional_size

    def offset(address):
        for index in range(count):
            size, start, length, base = struct.unpack_from("<IIII", raw, sections + index * 40 + 8)
            if start <= address < start + max(size, length):
                return base + address - start
        raise ValueError("PE_IMPORT_RVA_INVALID")

    if not rva:
        return []
    position, result = offset(rva), []
    while any(raw[position:position + 20]):
        name = offset(struct.unpack_from("<I", raw, position + 12)[0])
        result.append(raw[name:raw.index(b"\x00", name)].decode("ascii").casefold())
        position += 20
    return result


def main():
    if WORKSPACE.drive.upper() != "D:" or not WORKSPACE.is_dir():
        raise ValueError("WORKSPACE_MUST_BE_EXISTING_D_DRIVE")
    device = WORKSPACE.stat().st_dev
    records = []

    def copy(old, new):
        if WORKSPACE.stat().st_dev != device:
            raise OSError("WORKSPACE_DEVICE_CHANGED")
        if not new.is_relative_to(WORKSPACE):
            raise ValueError("TOOL_COPY_TARGET_OUTSIDE_WORKSPACE")
        new.parent.mkdir(parents=True, exist_ok=True)
        if not new.exists():
            with old.open("rb") as left, new.open("xb") as right:
                shutil.copyfileobj(left, right, 1024 * 1024)
                right.flush()
                os.fsync(right.fileno())
        def digest(path):
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        identity = digest(old)
        if digest(new) != identity:
            raise ValueError("WORKSPACE_TOOL_COPY_CHANGED:" + str(new))
        records.append({"file": str(new.relative_to(WORKSPACE)), "sha256": identity, "size": old.stat().st_size})
        if len(records) % 500 == 0:
            print(json.dumps({"copied_tool_files_verified": len(records)}), flush=True)

    python_source, python_target = Path("C:/anaconda3"), WORKSPACE / "03-python/.tools/python313"
    for path in python_source.iterdir():
        if path.is_file() and re.fullmatch(r"python.*\.(exe|dll|zip)|vcruntime.*\.dll|LICENSE.*", path.name, re.I):
            copy(path, python_target / path.name)
    for directory in ("Lib", "DLLs"):
        for folder, names, files in os.walk(python_source / directory):
            names[:] = [name for name in names if name not in {"site-packages", "__pycache__", "test", "tests"}]
            for name in files:
                if not name.endswith((".pyc", ".pyo")):
                    old = Path(folder) / name
                    copy(old, python_target / old.relative_to(python_source))
    available = {}
    for folder in (python_source, python_source / "DLLs", python_source / "Library/bin"):
        for path in folder.glob("*.dll"):
            available.setdefault(path.name.casefold(), path)
    queue = list(python_target.glob("*.exe")) + list(python_target.glob("*.dll"))
    queue += list((python_target / "DLLs").glob("*.pyd"))
    seen = set()
    while queue:
        for name in imports(queue.pop()):
            if name in seen or name not in available:
                continue
            seen.add(name)
            old = available[name]
            target = python_target / ("" if old.parent == python_source else "DLLs") / old.name
            copy(old, target)
            queue.append(target)
    node_source, node_target = Path("C:/nvm4w/nodejs"), WORKSPACE / ".tools/node"
    for name in ("node.exe", "npm", "npm.cmd", "npx", "npx.cmd", "nodevars.bat", "LICENSE"):
        if (node_source / name).is_file():
            copy(node_source / name, node_target / name)
    for folder, _, files in os.walk(node_source / "node_modules/npm"):
        for name in files:
            old = Path(folder) / name
            copy(old, node_target / old.relative_to(node_source))
    maven_source, maven_target = Path("C:/maven-3.9.11"), WORKSPACE / ".tools/maven"
    for folder, _, files in os.walk(maven_source):
        for name in files:
            old = Path(folder) / name
            copy(old, maven_target / old.relative_to(maven_source))
    settings = maven_target / "conf/settings.xml"
    text = settings.read_text(encoding="utf-8-sig")
    text = text.replace("<localRepository>C:\\maven-repository</localRepository>",
                        "<localRepository>" + (WORKSPACE / "02-java/.m2/repository").as_posix() + "</localRepository>")
    settings.write_text(text, encoding="utf-8")
    report = {"tool_files_independently_verified": len(records), "bytes": sum(row["size"] for row in records),
              "C_shared_installations_unchanged": True, "python": str(python_target), "node": str(node_target),
              "maven": str(maven_target), "records": records,
              "maven_settings_local_repository_retargeted_after_copy": True}
    output = WORKSPACE / "03-python/.local-runs/mixed-reading-20261107"
    (output / "workspace-tools-copy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
