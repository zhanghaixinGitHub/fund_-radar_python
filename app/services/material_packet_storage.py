"""预测输入的正文引用存储；还原后仍按原逻辑输入摘要校验，不改变日期或数字。"""

from copy import deepcopy

from app.services.material_content import PROJECT, SourceContent, checksum, encoded, repository

FORMAT = "MATERIAL_JSON_REFERENCES_V1"


def text_reference(body):
    """同一正文只保存一次；来源版本与提取摘要都核验，不引用可变的当前资料。"""
    raw = body.encode("utf-8")
    identity = checksum(raw)
    repo = repository()
    value = repo.get(source_hash=identity)
    if value is None:
        extracted = {"text": body}
        value = repo.put(
            PROJECT / "data/fund-materials/text-content" / (identity + ".txt"),
            SourceContent(identity, len(raw), "TEXT", extracted, checksum(encoded(extracted))),
        )
    representation = {}
    if value.kind not in {"TEXT", "JSON", "HTML"} or value.decode() != body:
        # 同一原JSON可能已按语义压缩过空白。审阅正文的字符位置有业务含义，另存
        # 精确文字表示，不能用语义等价文本破坏原引用范围或输入摘要。
        repo.page(identity, "PREDICTION_TEXT_V1", 0, body)
        representation = {"engine": "PREDICTION_TEXT_V1"}
    return {
        "source_hash": value.source_hash,
        "content_hash": value.content_hash,
        "characters": len(body),
        "body_hash": identity,
        **representation,
    }


def pack(value):
    """只抽出大字符串，元数据保持原值；引用路径显式列出，普通业务字段不充当标记。"""
    references = []

    def visit(item, path):
        if isinstance(item, str) and len(item) >= 4096:
            references.append({"path": path, **text_reference(item)})
            return None
        if isinstance(item, dict):
            return {key: visit(body, path + [key]) for key, body in item.items()}
        if isinstance(item, list):
            return [visit(body, path + [index]) for index, body in enumerate(item)]
        return item

    stored = visit(value, [])
    return {"storage_format": FORMAT, "value": stored, "references": references} if references else value


def unpack(value):
    """每次读取独立校验引用；缺失、篡改或路径冲突立即报错，不用最新正文补洞。"""
    if not isinstance(value, dict) or value.get("storage_format") != FORMAT:
        return value
    if set(value) != {"storage_format", "value", "references"}:
        raise ValueError("MATERIAL_PACKET_STORAGE_INVALID")
    restored = deepcopy(value["value"])
    seen = set()
    for reference in value["references"]:
        path = reference["path"]
        identity = tuple(path)
        if not path or identity in seen:
            raise ValueError("MATERIAL_PACKET_REFERENCE_PATH_INVALID")
        seen.add(identity)
        content = repository().get(source_hash=reference["source_hash"])
        if content is None or not content.complete or content.content_hash != reference["content_hash"]:
            raise ValueError("MATERIAL_PACKET_REFERENCE_INVALID")
        body = (
            repository().page(content.source_hash, reference["engine"], 0)
            if reference.get("engine")
            else content.decode()
        )
        if (
            body is None
            or len(body) != reference["characters"]
            or checksum(body.encode("utf-8")) != reference["body_hash"]
        ):
            raise ValueError("MATERIAL_PACKET_REFERENCE_LENGTH_INVALID")
        target = restored
        for key in path[:-1]:
            target = target[key]
        if target[path[-1]] is not None:
            raise ValueError("MATERIAL_PACKET_REFERENCE_PATH_INVALID")
        target[path[-1]] = body
    return restored
