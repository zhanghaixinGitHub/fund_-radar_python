"""整合工作区的运行目录和历史资料别名；历史路径只用于数据库身份兼容。"""

import os
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PYTHON_ROOT.parent
JAVA_ROOT = WORKSPACE_ROOT / "02-java"
VUE_ROOT = WORKSPACE_ROOT / "01-vue"
SOURCE_ROOT = WORKSPACE_ROOT / "04-SourceFile"

# 这些是已核验的迁移前位置，只允许按完整目录边界映射，不能进行任意字符串替换。
LEGACY_PYTHON_ROOT = Path("C:/pythonProject/workSpace06")
LEGACY_SOURCE_ROOT = Path("D:/fundRadarSourceFile")


def material_location_aliases(path, *, project_root=PYTHON_ROOT, archive_root=SOURCE_ROOT):
    """返回数据库查询候选，精确路径优先，另一个候选仅兼容已知的迁移前身份。

    不打开、创建或改写原件，不批量更新历史记录。新位置已有新版本时优先新位置，
    显式查询旧位置仍保留旧版本；缺少精确别名时才复用迁移前后同一相对路径。
    """
    # 只作词法规范化，先消除..再判断目录边界，不查询或打开历史实体。
    path = Path(os.path.abspath(path))
    candidates = [str(path)]
    for current, previous in ((Path(project_root), LEGACY_PYTHON_ROOT), (Path(archive_root), LEGACY_SOURCE_ROOT)):
        for left, right in ((current, previous), (previous, current)):
            if path.is_relative_to(left):
                candidates.append(str(right / path.relative_to(left)))
                return tuple(dict.fromkeys(candidates))
    return tuple(candidates)


def runtime_material_path(path, *, project_root=PYTHON_ROOT, archive_root=SOURCE_ROOT):
    """把历史计划中的已知资料路径转到当前实体，未知目录不推测、不访问旧工程。"""
    path = Path(os.path.abspath(path))
    for previous, current in ((LEGACY_PYTHON_ROOT, project_root), (LEGACY_SOURCE_ROOT, archive_root)):
        if path.is_relative_to(previous):
            return Path(current) / path.relative_to(previous)
    return path
