"""旧JSONB快照的表示兼容；只借同一份已校验资料恢复负零，最终仍要求原始哈希相等。"""

import math
from copy import deepcopy
from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read


def materials_input_identity(material):
    """经济输入身份仅排除用途批准的说明和时间；来源、原件及所有资料字段仍参与。

    重复批准说明不产生新行情或财务证据，不应凭审计元数据变化重复生成预测。
    原包完整摘要仍在交接/快照处核验，本函数不修改原件或降低资料要求。
    """
    return digest({k: v for k, v in material.items() if k not in {"workflow_review_note", "workflow_reviewed_at"}})


def checked_snapshot(payload, expected_hash):
    """PostgreSQL将JSON数值 -0.0 读为 0.0；保留原快照，不修改数据库或预测。

    仅允许用该快照已经绑定、摘要一致的本基金原资料恢复同位置负零。任何其他
    字段变化、丢失原包、错基金或最终原始哈希不一致均拒绝，不能以数值近似放行。
    """
    from app.services.material_packet_storage import unpack

    payload = unpack(payload)
    if digest(payload) == expected_hash:
        return payload
    data = payload if "fund_code" in payload else payload["data"]
    materials = read(Path(data["materials_path"]))
    if digest(materials) != data.get("materials_hash") or materials.get("fund_code") != data.get("fund_code"):
        raise ValueError("SNAPSHOT_ORIGINAL_MATERIALS_MISMATCH")
    restored = deepcopy(payload)
    target = restored if "fund_code" in restored else restored["data"]

    def restore(value, original):
        if isinstance(value, dict) and isinstance(original, dict):
            for key in value.keys() & original.keys():
                value[key] = restore(value[key], original[key])
        elif isinstance(value, list) and isinstance(original, list) and len(value) == len(original):
            for index in range(len(value)):
                value[index] = restore(value[index], original[index])
        elif (
            isinstance(value, float)
            and isinstance(original, float)
            and value == original == 0.0
            and math.copysign(1, original) < 0
        ):
            return original
        return value

    restore(target, materials)
    if digest(restored) != expected_hash:
        raise ValueError("PREDICTION_INPUT_SNAPSHOT_HASH_INVALID")
    return restored
