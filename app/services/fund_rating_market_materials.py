"""市场资料核验的部分成果；只由手动任务加载，不将缺项资料准入为评分样本。

每个事实保留原件引用。已发布批次冻结完整说明，普通查询仅读取该批次。
"""

import hashlib
import json
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

from app.services.fund_rating_inputs import digest


def load(root: Path, as_of: date, frozen: datetime) -> dict:
    """读取本机维护命令保存的有界不可变清单；不存在清单时沿用原有空态。"""
    pointer = root / "market-materials-current.json"
    if not pointer.exists():
        return {}
    reference = json.loads(pointer.read_text(encoding="utf8"))
    path = (root / reference["file"]).resolve()
    if not path.is_relative_to(root.resolve() / "assessments") or path.stat().st_size > 20_000_000:
        raise ValueError("RATING_MARKET_MATERIALS_PATH_SIZE")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("RATING_MARKET_MATERIALS_HASH")
    packet = json.loads(raw)
    if packet["version"] != 1 or packet["as_of_date"] != str(as_of):
        return {}
    if not 1 <= len(packet["records"]) <= 500:
        raise ValueError("RATING_MARKET_MATERIALS_COUNT")
    for code, item in packet["records"].items():
        verified = datetime.fromisoformat(item["verified_at"])
        if (
            code != item["fund_code"]
            or len(code) != 6
            or not code.isascii()
            or not code.isdigit()
            or verified.tzinfo is None
        ):
            raise ValueError("RATING_MARKET_MATERIALS_IDENTITY_TIME")
        if item["source_root"] != str(root.resolve()) or item["as_of_date"] != str(as_of):
            raise ValueError("RATING_MARKET_MATERIALS_ROOT_SCOPE")
        for ref in item["sources"]:
            source = (root / ref["file"]).resolve()
            if (
                not source.is_relative_to(root.resolve() / "sources")
                or source.stat().st_size > 20_000_000
                or hashlib.sha256(source.read_bytes()).hexdigest() != ref["sha256"]
            ):
                raise ValueError("RATING_MARKET_MATERIALS_ORIGINAL")
    # 后补资料不属于较早冻结时点的已知事实；历史复算忽略它，而非使用它或使整批失败。
    return {
        code: item for code, item in packet["records"].items() if datetime.fromisoformat(item["verified_at"]) <= frozen
    }


def attach(member: dict, material: dict) -> None:
    """只显示真实事实及缺口，不设置 admitted、分值、等级或替换权重。"""
    public = deepcopy(member["public"])
    member["preparation"] = material
    member["reasons"] = [b["code"] for b in material["blockers"]] or ["MARKET_MATERIALS_NOT_ADMITTED"]
    public["summary"] = "必要资料尚未核齐，暂无评分。已核实的历史表现和披露资料见下方。"
    for dimension in public["dimensions"]:
        if dimension["key"] in material["dimensions"]:
            dimension.update(material["dimensions"][dimension["key"]])
        gaps = [b["message"] for b in material["blockers"] if b["dimension"] == dimension["key"]]
        if gaps:
            dimension["explanation"] = " ".join(gaps)
        dimension["grade"] = None
        dimension["gradeLabel"] = None
    public["dataDates"].update(material["data_dates"])
    public["comparison"].update(period=material["period"], scope="依据本基金已核验的公开资料；尚未形成完整评价")
    public["limitations"] = [b["message"] for b in material["blockers"]]
    public["limitations"].append("暂无评分不等于偏弱；历史表现及定期持仓不代表未来收益或实时资产。")
    member["public"] = public
    member["material_hash"] = digest(material)
