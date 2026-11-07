"""必要栅格图片的共享内容资产；已有JPEG不转码，PNG只做逐像素核验的无损优化。

正文能够完整表达的页面不应调用本模块。这里仅处理仍需保留的图像，存储
格式变化不能改变像素、透明色值或色彩配置。Pillow仅在迁移/独立像素验收时
按需导入，日常底稿读取使用标准库核对内容摘要，不依赖迁移工具目录。
"""

import base64
import hashlib
import io
import json
import zlib

from app.services.material_content import encoded

ASSET_KIND = "RASTER_IMAGE_V1"
ASSET_FORMAT = "RASTER_IMAGE"
ASSET_PREFIX = b"MATERIAL_SHARED_RASTER_V1\x00"
MAX_ENCODE_PIXELS = 40_000_000


def payload(raw, media_type):
    """以完整已编码图片及格式生成不可变资产；跨页面复用相同编码内容。"""
    if media_type not in {"image/png", "image/jpeg", "image/webp", "image/gif", "image/tiff"}:
        raise ValueError("MATERIAL_RASTER_MEDIA_TYPE_UNSUPPORTED")
    body = encoded({"media_type": media_type, "encoded_size": len(raw),
                    "encoded_sha256": hashlib.sha256(raw).hexdigest(),
                    "data_base64": base64.b64encode(raw).decode("ascii")})
    return {"asset_kind": ASSET_KIND,
            "source_hash": hashlib.sha256(ASSET_PREFIX + body).hexdigest(),
            "content_hash": hashlib.sha256(body).hexdigest()}, body


def validate_asset(identity, expected_hash, compressed):
    """读取时核对封装、命名空间与编码字节，不把摘要校验冒充首次像素验收。"""
    body = zlib.decompress(compressed)
    if (hashlib.sha256(body).hexdigest() != expected_hash
            or hashlib.sha256(ASSET_PREFIX + body).hexdigest() != identity):
        raise ValueError("MATERIAL_RASTER_ASSET_HASH_MISMATCH")
    value = json.loads(body)
    raw = base64.b64decode(value["data_base64"], validate=True)
    if len(raw) != value["encoded_size"] or hashlib.sha256(raw).hexdigest() != value["encoded_sha256"]:
        raise ValueError("MATERIAL_RASTER_ENCODED_BYTES_MISMATCH")
    if value["media_type"] not in {"image/png", "image/jpeg", "image/webp", "image/gif", "image/tiff"}:
        raise ValueError("MATERIAL_RASTER_MEDIA_TYPE_UNSUPPORTED")
    return value


def _metadata(value):
    if isinstance(value, bytes):
        return {"binary_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def optimise(raw, media_type):
    """返回编码内容、MIME和可复核凭据；不做降采样、丢帧或有损压缩。

    JPEG/GIF/TIFF直接保持编码字节；PNG只有在模式、帧数和尺寸适合、重解码
    像素及ICC完全一致且结果更小时才替换成无损WebP。原容器附加信息单独保留。
    """
    original_hash = hashlib.sha256(raw).hexdigest()
    audit = {"source_encoded_sha256": original_hash, "source_encoded_size": len(raw),
             "source_media_type": media_type, "preservation": "ENCODED_BYTES_UNCHANGED"}
    if media_type != "image/png":
        return raw, media_type, audit
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        audit.update(width=image.width, height=image.height, mode=image.mode,
                     container_metadata=_metadata(image.info))
        if (getattr(image, "n_frames", 1) != 1 or image.width * image.height > MAX_ENCODE_PIXELS
                or image.mode not in {"RGB", "RGBA", "L"} or "transparency" in image.info):
            return raw, media_type, audit
        image.load()
        expected = hashlib.sha256(image.tobytes()).hexdigest()
        profile = image.info.get("icc_profile", b"")
        output = io.BytesIO()
        image.save(output, format="WEBP", lossless=True, exact=True, quality=60, method=0,
                   icc_profile=profile)
        candidate = output.getvalue()
        with Image.open(io.BytesIO(candidate)) as restored:
            restored.load()
            if (restored.size != image.size or restored.info.get("icc_profile", b"") != profile
                    or hashlib.sha256(restored.convert(image.mode).tobytes()).hexdigest() != expected):
                raise ValueError("MATERIAL_RASTER_LOSSLESS_REENCODE_CHANGED_PIXELS")
        audit.update(pixel_sha256=expected, icc_sha256=hashlib.sha256(profile).hexdigest())
        if len(candidate) < len(raw):
            audit["preservation"] = "LOSSLESS_PIXELS_AND_ICC_VERIFIED"
            return candidate, "image/webp", audit
        return raw, media_type, audit


def verify_reencoding(original, replacement, audit):
    """独立读回时再验证编码等价；不依赖写入端声称已通过的布尔值。"""
    if hashlib.sha256(original).hexdigest() != audit["source_encoded_sha256"]:
        raise ValueError("MATERIAL_RASTER_SOURCE_CHANGED")
    if original == replacement:
        return True
    from PIL import Image

    with Image.open(io.BytesIO(original)) as before, Image.open(io.BytesIO(replacement)) as after:
        before.load()
        after.load()
        if (before.size != after.size or before.info.get("icc_profile", b"") != after.info.get("icc_profile", b"")
                or hashlib.sha256(before.tobytes()).digest()
                != hashlib.sha256(after.convert(before.mode).tobytes()).digest()):
            raise ValueError("MATERIAL_RASTER_REENCODING_NOT_EQUIVALENT")
    return True
