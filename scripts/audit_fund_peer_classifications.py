"""只读核验同类分类原件、身份、摘录及当前目录覆盖，并保留不可变验收快照。"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader
from sqlalchemy import select, text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.db.session import get_engine  # noqa: E402
from app.models.fund import FundShareClass  # noqa: E402
from app.services.fund_peer_classifications import CLASSIFICATION_FILE, PeerSnapshot  # noqa: E402
from app.services.fund_peer_comparison import read_peer_comparison  # noqa: E402


def normalized(value: str) -> str:
    """只消除排版空白和全角差异，不改写证券代码、数值或投资限制。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def audit(output: Path) -> dict:
    """原件读取有界、查询使用只读事务；不发布评级、不采集外部资料、不修改数据库。"""
    raw = CLASSIFICATION_FILE.read_bytes()
    packet = PeerSnapshot.model_validate_json(raw)
    root = (Path(get_settings().fund_insight_directory) / "ratings").resolve()
    documents, verified = {}, []
    for record in packet.records:
        full_texts = []
        for source in record.sources:
            path = (root / source.file).resolve()
            if not path.is_relative_to(root / "sources") or path.stat().st_size > 30_000_000:
                raise ValueError("peer source outside original archive or too large")
            if source.sha256 not in documents:
                original = path.read_bytes()
                if hashlib.sha256(original).hexdigest() != source.sha256:
                    raise ValueError("peer source hash mismatch")
                if path.suffix == ".pdf":
                    reader = PdfReader(path)
                    if len(reader.pages) > 300:
                        raise ValueError("peer original exceeds page limit")
                    value = "\n".join(page.extract_text() for page in reader.pages)
                else:
                    value = json.loads(original)["data"]["notice_content"]
                documents[source.sha256] = normalized(value)
            value = documents[source.sha256]
            if any(normalized(quote) not in value for quote in source.excerpts):
                raise ValueError(f"peer excerpt absent from original: {record.fund_code}")
            full_texts.append(value)
        identity = "".join(full_texts)
        if record.fund_code not in identity or record.product_id.split(":")[-1] not in identity:
            raise ValueError(f"peer product identity absent: {record.fund_code}")
        verified.append({
            "fundCode": record.fund_code, "productId": record.product_id, "shareClass": record.share_class,
            "group": record.name, "criteria": record.criteria, "effectiveFrom": str(record.effective_from),
            "sources": [source.model_dump(mode="json") for source in record.sources],
        })
    results = []
    with Session(get_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        session.execute(text("SET LOCAL statement_timeout='12000ms'"))
        codes = session.scalars(select(FundShareClass.fund_code).where(
            FundShareClass.status == "ACTIVE", FundShareClass.source_code == "TUSHARE_PRO_FUND",
        ).order_by(FundShareClass.fund_code)).all()
        for code in codes:
            result = read_peer_comparison(session, code)
            results.append(result.model_dump(mode="json"))
    report = {
        "checkedAt": datetime.now(UTC).isoformat(), "version": packet.version,
        "snapshotHash": hashlib.sha256(raw).hexdigest(), "originalCount": len(documents),
        "catalogCount": len(codes), "verifiedClassificationCount": len(packet.records),
        "statuses": dict(Counter(result["status"] for result in results)),
        "coverageGaps": packet.gaps, "verified": verified, "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    immutable = output / (report["snapshotHash"] + ".classification.json")
    if not immutable.exists():
        immutable.write_bytes(raw)
    if immutable.read_bytes() != raw:
        raise ValueError("immutable peer snapshot conflict")
    target = output / (datetime.now(UTC).strftime("audit-%Y%m%dT%H%M%S%fZ.json"))
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return {key: report[key] for key in (
        "snapshotHash", "originalCount", "catalogCount", "verifiedClassificationCount", "statuses",
    )} | {"auditFile": str(target.resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".local-runs/peer-comparison-20261026"))
    arguments = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(audit(arguments.output), ensure_ascii=False))
