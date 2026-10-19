"""把已有研究目录中的真实原件登记到隔离评级档案，保留原取得时间和日期。

不调用采集器、不执行研究目录脚本、不复制其评分或核验布尔值。只接纳
给定目录内摘要相符的PDF/DOCX及已有明确适配的原始JSON响应；
正文、API请求身份和日期分别核验，旧原件绝不覆盖。
"""

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from app.services.fund_exposure_common import _publish


def reuse_database_nav(code: str, source_root: Path, rating_root: Path, frozen: datetime, as_of: date) -> list[dict]:
    """接纳同一只读事务冻结的本份额规范净值，明确区别于原始HTTP回执。"""
    source_root = source_root.resolve()
    manifest = source_root / "quality-database-nav.json"
    if not manifest.exists():
        return []
    if manifest.stat().st_size > 1_000_000:
        raise ValueError("DATABASE_NAV_MANIFEST_SIZE")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data = data.get("payload", data)
    if data.get("schema") != "QUALITY_DATABASE_NAV_SNAPSHOTS_V1" or len(data.get("items", [])) > 500:
        raise ValueError("DATABASE_NAV_MANIFEST_SCHEMA")
    result = []
    for item in data["items"]:
        if item.get("fund_code") != code:
            continue
        if item.get("source_kind") != "LOCAL_DATABASE_NAV_SNAPSHOT":
            raise ValueError("DATABASE_NAV_SOURCE_KIND")
        receipt = item["receipt"]
        path = (source_root / receipt["file"]).resolve()
        if not path.is_relative_to(source_root / "raw") or not path.is_file() or path.stat().st_size > 5_000_000:
            raise ValueError("DATABASE_NAV_SOURCE_PATH_SIZE")
        raw = path.read_bytes()
        key = hashlib.sha256(raw).hexdigest()
        if key != receipt["sha256"]:
            raise ValueError("DATABASE_NAV_SOURCE_HASH")
        snapshot = json.loads(raw)
        if snapshot.get("captured_at") != receipt["received_at"]:
            raise ValueError("DATABASE_NAV_CAPTURE_TIME_CONFLICT")
        from app.services.fund_quality_v2_database_nav import database_nav_coverage

        database_nav_coverage(snapshot, fund_code=code, as_of=as_of, frozen=frozen)
        target = rating_root / "sources" / (key + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("DATABASE_NAV_DESTINATION_CONFLICT")
        else:
            _publish(target, raw)
        result.append(
            {
                "file": target.relative_to(rating_root).as_posix(),
                "sha256": key,
                "label": "nav-database-snapshot",
                "url": "local-database:nav_daily",
                "acquired_at": receipt["received_at"],
                "source_kind": "NORMALIZED_DATABASE_NAV_SNAPSHOT_NOT_RAW_HTTP_RESPONSE",
                "source_code": "TUSHARE_PRO_FUND",
                "title": "本份额已有规范净值的只读数据库快照",
                "origin_receipt_file": str(manifest),
            }
        )
    return result


def reuse_regulatory_originals(
    code: str, source_root: Path, rating_root: Path, frozen: datetime, as_of: date
) -> list[dict]:
    """登记公开监管决定原件；基金列表仅作路由，适用对象由转换器与正式管理人核对。"""
    manifest = source_root / "quality-governance-originals.json"
    if not manifest.exists():
        return []
    if manifest.stat().st_size > 1_000_000:
        raise ValueError("GOVERNANCE_SOURCE_MANIFEST_SIZE")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data = data.get("payload", data)
    if (
        data.get("schema") != "QUALITY_GOVERNANCE_SOURCE_RECEIPTS_V1"
        or not isinstance(data.get("items"), list)
        or len(data["items"]) > 500
    ):
        raise ValueError("GOVERNANCE_SOURCE_MANIFEST_SCHEMA")
    result = []
    for item in data["items"]:
        if code not in item.get("fund_codes", []):
            continue
        receipt = item["receipt"]
        url = urlsplit(receipt["url"])
        if (
            item.get("source_kind") != "SAFE_PUBLIC_DECISION"
            or url.scheme != "https"
            or url.hostname != "www.safe.gov.cn"
            or url.path != "/www/illegal/listQuery"
        ):
            raise ValueError("GOVERNANCE_REGULATOR_SOURCE_UNSUPPORTED")
        path = (source_root / receipt["file"]).resolve()
        if not path.is_relative_to(source_root / "raw") or not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError("GOVERNANCE_SOURCE_PATH_SIZE")
        acquired = datetime.fromisoformat(receipt["received_at"])
        if acquired.tzinfo is None or acquired > frozen:
            raise ValueError("GOVERNANCE_SOURCE_RECEIPT_TIME")
        raw = path.read_bytes()
        key = hashlib.sha256(raw).hexdigest()
        if key != receipt["sha256"] or receipt.get("status") != 200:
            raise ValueError("GOVERNANCE_SOURCE_HASH_OR_HTTP_STATUS")
        from app.services.fund_quality_v2_governance import safe_decision

        decision = safe_decision(raw)
        if decision["event_date"] > str(as_of):
            continue
        target = rating_root / "sources" / (key + ".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != key:
                raise ValueError("REUSED_SOURCE_DESTINATION_HASH")
        else:
            _publish(target, raw)
        result.append(
            {
                "file": target.relative_to(rating_root).as_posix(),
                "sha256": key,
                "url": receipt["url"],
                "acquired_at": receipt["received_at"],
                "label": "governance-regulator",
                "source_kind": "SAFE_PUBLIC_DECISION",
                "title": decision["authority"] + decision["document_number"],
                "legal_entity": decision["legal_entity"],
                "decision_date": decision["event_date"],
                "publication_date_status": "NOT_REPORTED_BY_THIS_PAGE",
                "origin_receipt_file": str(manifest),
            }
        )
    return result


def reuse_api_originals(code: str, source_root: Path, rating_root: Path, frozen: datetime) -> list[dict]:
    """接纳显式回执清单中的有界公开任职响应，保留实际字节、身份及取得时刻。

    清单不是核验结论：每次仍检查路径、摘要并重读API字段和份额身份。
    这里只接受既有供应商的fund_manager，不把任意JSON当作可用原件。
    """
    manifest = source_root / "quality-originals.json"
    if not manifest.exists():
        return []
    if manifest.stat().st_size > 1_000_000:
        raise ValueError("SUPPLEMENTAL_ORIGINAL_MANIFEST_SIZE")
    content = json.loads(manifest.read_text(encoding="utf-8"))
    content = content.get("payload", content)
    if (
        content.get("schema") != "QUALITY_SOURCE_RECEIPTS_V1"
        or not isinstance(content.get("items"), list)
        or len(content["items"]) > 500
    ):
        raise ValueError("SUPPLEMENTAL_ORIGINAL_MANIFEST_SCHEMA")
    result = []
    for item in content["items"]:
        if item.get("fund_code") != code or item.get("failure_type"):
            continue
        receipt = item["receipt"]
        url = urlsplit(receipt["url"])
        if (
            item.get("api_name") != "fund_manager"
            or item.get("label") not in {"manager-api", "manager-master-api"}
            or url.scheme != "https"
            or url.hostname != "api.tushare.pro"
        ):
            raise ValueError("SUPPLEMENTAL_ORIGINAL_API_UNSUPPORTED")
        path = (source_root / receipt["file"]).resolve()
        if not path.is_relative_to(source_root / "raw") or not path.is_file() or path.stat().st_size > 5_000_000:
            raise ValueError("SUPPLEMENTAL_ORIGINAL_PATH_SIZE")
        acquired = datetime.fromisoformat(receipt["received_at"])
        if acquired.tzinfo is None or acquired > frozen:
            raise ValueError("SUPPLEMENTAL_ORIGINAL_RECEIPT_TIME")
        raw = path.read_bytes()
        key = hashlib.sha256(raw).hexdigest()
        if key != receipt["sha256"]:
            raise ValueError("SUPPLEMENTAL_ORIGINAL_HASH")
        from app.services.fund_quality_v2_manager_sources import manager_api_records

        source_code = item["ts_code"].split(".")[0]
        if item["label"] == "manager-api" and source_code != code:
            raise ValueError("SUPPLEMENTAL_MANAGER_SHARE_IDENTITY")
        if item["label"] == "manager-master-api" and source_code != item.get("master_code"):
            raise ValueError("SUPPLEMENTAL_MANAGER_MASTER_IDENTITY")
        # 产品级任职史只在转换阶段与正式报告主代码再次比对；收据声明不授予产品关系。
        manager_api_records(json.loads(raw), fund_code=source_code, ts_code=item["ts_code"])
        target = rating_root / "sources" / (key + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != key:
                raise ValueError("REUSED_SOURCE_DESTINATION_HASH")
        else:
            _publish(target, raw)
        result.append(
            {
                "sha256": key,
                "file": target.relative_to(rating_root).as_posix(),
                "url": receipt["url"],
                "acquired_at": receipt["received_at"],
                "label": item["label"],
                "master_code": item.get("master_code"),
                "api_name": "fund_manager",
                "ts_code": item["ts_code"],
                "source_code": item.get("source_code"),
                "origin_receipt_file": str(manifest),
            }
        )
    return result


def reuse_cash_originals(
    code: str, source_root: Path, rating_root: Path, frozen: datetime, as_of: date
) -> list[dict]:
    """接纳显式现金原件清单；清单仅供路由，金额和实施情况仍由原文及日恒等式核验。"""
    manifest = source_root / "quality-cash-originals.json"
    if not manifest.exists():
        return []
    if manifest.stat().st_size > 1_000_000:
        raise ValueError("QUALITY_CASH_ORIGINAL_MANIFEST_SIZE")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value = value.get("payload", value)
    if value.get("schema") != "QUALITY_CASH_ORIGINALS_V1" or len(value.get("items", [])) > 200:
        raise ValueError("QUALITY_CASH_ORIGINAL_MANIFEST_SCHEMA")
    result = []
    for item in value["items"]:
        if item.get("fund_code") != code:
            continue
        label = item.get("label")
        if label not in {"cash-event-snapshot", "cash-dividend-notice"}:
            raise ValueError("QUALITY_CASH_ORIGINAL_ROLE")
        if label == "cash-dividend-notice" and (
            not item.get("published_on") or item["published_on"] > str(as_of)
        ):
            continue
        ref = item["receipt"]
        origin = urlsplit(ref["url"])
        if origin.scheme != "https" or origin.hostname != (
            "www.wjasset.com" if label == "cash-dividend-notice" else "tushare.pro"
        ):
            raise ValueError("QUALITY_CASH_ORIGINAL_SOURCE_HOST")
        acquired = datetime.fromisoformat(ref["received_at"])
        path = (source_root / ref["file"]).resolve()
        if (acquired.tzinfo is None or acquired > frozen or not path.is_relative_to(source_root.resolve() / "raw")
                or not path.is_file() or path.stat().st_size > 8_000_000):
            raise ValueError("QUALITY_CASH_ORIGINAL_TIME_PATH_OR_SIZE")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != ref["sha256"]:
            raise ValueError("QUALITY_CASH_ORIGINAL_BYTES_CHANGED")
        if label == "cash-dividend-notice":
            if not raw.startswith(b"%PDF"):
                raise ValueError("QUALITY_CASH_NOTICE_PDF_REQUIRED")
        else:
            snapshot = json.loads(raw)
            if (snapshot.get("fund_code") != code or not isinstance(snapshot.get("dividend_records"), list)
                    or len(snapshot["dividend_records"]) > 200):
                raise ValueError("QUALITY_CASH_SNAPSHOT_SHARE_OR_SCHEMA")
        target = rating_root / "sources" / (sha + (".pdf" if label == "cash-dividend-notice" else ".json"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("QUALITY_CASH_SOURCE_DESTINATION_CONFLICT")
        else:
            _publish(target, raw)
        result.append({
            "file": target.relative_to(rating_root).as_posix(), "sha256": sha, "url": ref["url"],
            "acquired_at": ref["received_at"], "published_on": item.get("published_on"), "label": label,
            "source_kind": "ISSUER_ORIGINAL_CASH_DIVIDEND_PDF" if label == "cash-dividend-notice"
            else "NORMALIZED_PROVIDER_EVENT_SNAPSHOT_NOT_RAW_HTTP_RESPONSE",
            "origin_receipt_file": str(manifest), "title": item["title"],
        })
    return result


def reuse_originals(
    code: str,
    name: str,
    source_root: Path,
    rating_root: Path,
    as_of: date,
    frozen: datetime,
    admission_gaps: list[dict] | None = None,
) -> list[dict]:
    """根据现存报告/基金公告的来源收据定位原件；A/C只共享真实产品级报告。

    012836使用012835研究档案的报告是明确的既有产品关系，仍由转换器重读
    报告正文核对份额代码与主代码，不因该目录别名直接生成产品核验结论。
    """
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise ValueError("ADDITIONAL_SOURCE_ROOT_NOT_FOUND")
    master = "012835" if code == "012836" else code
    # 已证明产品关系的共同报告可读主代码目录，份额专属概要仍须读自身目录。
    # 两个目录都存在时不因别名路由而丢失C类自己的费用，具体身份由正文再核。
    archive_codes = {master, code}
    files = sorted(
        {
            path
            for archive_code in archive_codes
            for folder in ("reports", "fund-documents")
            for path in (source_root / folder / archive_code).glob("*.json")
        }
    )
    if len(files) > 500:
        raise ValueError("ADDITIONAL_SOURCE_METADATA_LIMIT")
    result = {}
    for metadata_path in files:
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        if item.get("fund_code") not in archive_codes:
            raise ValueError("ADDITIONAL_SOURCE_METADATA_IDENTITY")
        title = item.get("title", "")
        published = item.get("publishedDate") or item.get("published_date")
        if not published or published > str(as_of):
            continue
        receipt = item.get("source_receipt") or item.get("receipt")
        if not receipt:
            continue
        # 提示公告、合同摘要不能当作完整招募书或合同。
        if any(w in title for w in ("提示性公告", "摘要", "提示公告", "更正公告")):
            continue
        if item.get("source_kind") == "ISSUER_CMF_CONTRACT_AMENDMENT_HTML":
            label = "contract-amendment-notice"
        elif "中期报告" in title:
            label = "historical-half"
        elif "年度报告" in title:
            label = "historical-annual"
        elif "季度报告" in title:
            label = "historical-quarter"
        elif "产品资料概要" in title:
            target_class = "C" if name.endswith("-C") else "A" if name.endswith("-A") else None
            if target_class and target_class + "类份额" not in title.replace(" ", ""):
                continue
            label = "summary"
        elif "招募说明书" in title:
            label = "prospectus"
        elif "基金合同" in title:
            label = "historical-contract"
        elif "基金经理" in title and any(word in title for word in ("变更", "增聘")):
            label = "appointment"
        else:
            continue
        raw_path = (source_root / receipt["file"]).resolve()
        if (
            not raw_path.is_relative_to(source_root / "raw")
            or not raw_path.is_file()
            or raw_path.stat().st_size > 20_000_000
        ):
            raise ValueError("ADDITIONAL_SOURCE_PATH_SIZE")
        acquired = datetime.fromisoformat(receipt["received_at"])
        if acquired.tzinfo is None or acquired > frozen:
            raise ValueError("ADDITIONAL_SOURCE_RECEIPT_TIME")
        raw = raw_path.read_bytes()
        key = hashlib.sha256(raw).hexdigest()
        if key != receipt["sha256"]:
            raise ValueError("ADDITIONAL_SOURCE_HASH")
        article = None
        # 完整转载仍从保存的HTTP响应重读证券、公告号、标题和正文；不取元数据body。
        if not raw.startswith((b"%PDF", b"PK\x03\x04")):
            from app.services.fund_rating_public_market import REPRINT_HOST, public_document

            if label == "contract-amendment-notice":
                # 此分支仅复制实际HTML和独立收据；主体、正文、附件行及
                # 日期在转换器重读，不由source_kind字符串或目录标志接纳。
                pass
            elif urlsplit(receipt["url"]).hostname == REPRINT_HOST:
                article = (json.loads(raw).get("data") or {}).get("art_code")
                required = (
                    "产品资料概要"
                    if label == "summary"
                    else "招募说明书"
                    if label == "prospectus"
                    else "基金经理"
                    if label == "appointment"
                    else "报告"
                )
                public_document(raw, {"article_id": article}, code, required)
            else:
                if admission_gaps is not None:
                    admission_gaps.append(
                        {
                            "state": "ORIGINAL_AVAILABLE_PARSER_REQUIRED",
                            "reason": "ISSUER_HTML_ADAPTER_REQUIRED",
                            "sha256": key,
                            "title": title,
                            "published_on": published,
                            "source_url": receipt["url"],
                            "original_file": str(raw_path),
                        }
                    )
                continue
        suffix = (".html" if label == "contract-amendment-notice" else ".pdf" if raw.startswith(b"%PDF")
                  else ".docx" if raw.startswith(b"PK\x03\x04") else ".json")
        target = rating_root / "sources" / (key + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != key:
                raise ValueError("REUSED_SOURCE_DESTINATION_HASH")
        else:
            _publish(target, raw)
        ref = {
            "sha256": key,
            "file": target.relative_to(rating_root).as_posix(),
            "url": receipt["url"],
            "acquired_at": receipt["received_at"],
            "published_on": published,
            "label": label,
            "title": title,
            "report_end": item.get("endDate"),
            "origin_receipt_file": str(metadata_path),
        }
        if label == "contract-amendment-notice":
            proof = item.get("contract_notice_receipt")
            if not isinstance(proof, dict) or set(proof) != {"file", "sha256"}:
                raise ValueError("CONTRACT_NOTICE_DOWNLOAD_RECEIPT_REFERENCE_REQUIRED")
            proof_path = (source_root/proof["file"]).resolve()
            if (not proof_path.is_relative_to(source_root/"raw") or not proof_path.is_file()
                    or proof_path.stat().st_size > 100_000):
                raise ValueError("CONTRACT_NOTICE_RECEIPT_SOURCE_PATH_SIZE")
            proof_raw = proof_path.read_bytes()
            if hashlib.sha256(proof_raw).hexdigest() != proof["sha256"]:
                raise ValueError("CONTRACT_NOTICE_RECEIPT_SOURCE_ACTUAL_HASH")
            target_proof = rating_root/"sources"/(proof["sha256"]+".json")
            if target_proof.exists():
                if target_proof.read_bytes() != proof_raw:
                    raise ValueError("CONTRACT_NOTICE_RECEIPT_DESTINATION_CONFLICT")
            else:
                _publish(target_proof, proof_raw)
            result[proof["sha256"]] = {"label": "contract-notice-download-receipt", "sha256": proof["sha256"],
                "file": target_proof.relative_to(rating_root).as_posix(), "url": receipt["url"],
                "acquired_at": receipt["received_at"], "published_on": published,
                "source_kind": "ORIGINAL_VERSIONED_DOWNLOAD_RECEIPT_NOT_CONTENT_ACCEPTANCE_FLAG"}
            ref.update(contract_notice_receipt_sha256=proof["sha256"],
                       source_kind="ISSUER_CMF_CONTRACT_AMENDMENT_HTML")
        if urlsplit(receipt["url"]).hostname == "static.cninfo.com.cn":
            # 巨潮新原件的真实下载收据单独按原字节复制，不能只登记verified标志。
            # 此处仅归档；主体、日期、HTTP结果与PDF的绑定在V2披露读取器重核。
            proof = item.get("cninfo_download_receipt")
            if not isinstance(proof, dict) or set(proof) != {"file", "sha256"}:
                raise ValueError("CNINFO_DOWNLOAD_RECEIPT_REFERENCE_REQUIRED")
            proof_path = (source_root/proof["file"]).resolve()
            if (not proof_path.is_relative_to(source_root/"raw") or not proof_path.is_file()
                    or proof_path.stat().st_size > 100_000):
                raise ValueError("CNINFO_DOWNLOAD_RECEIPT_SOURCE_PATH_SIZE")
            proof_raw = proof_path.read_bytes()
            if hashlib.sha256(proof_raw).hexdigest() != proof["sha256"]:
                raise ValueError("CNINFO_DOWNLOAD_RECEIPT_SOURCE_HASH")
            proof_target = rating_root/"sources"/(proof["sha256"]+".json")
            if proof_target.exists():
                if proof_target.read_bytes() != proof_raw:
                    raise ValueError("CNINFO_DOWNLOAD_RECEIPT_DESTINATION_CONFLICT")
            else:
                _publish(proof_target, proof_raw)
            result[proof["sha256"]] = {"label": "cninfo-download-receipt",
                "sha256": proof["sha256"], "file": proof_target.relative_to(rating_root).as_posix(),
                "url": receipt["url"], "acquired_at": receipt["received_at"], "published_on": published,
                "source_kind": "ORIGINAL_VERSIONED_DOWNLOAD_RECEIPT_NOT_PDF_CONTENT_OR_VALIDATION_FLAG"}
            ref["cninfo_receipt_sha256"] = proof["sha256"]
        if "publication_evidence" in item:
            # 证明文件是原始目录/报告的独立收据；保留真实取得或本地复用观测日。
            # 这里只做文件登记，具体目录条目和公告索引仍在转换时重读核验。
            evidence = item["publication_evidence"]
            if not isinstance(evidence, dict) or set(evidence) != {"catalog", "notice_index", "article_id"}:
                raise ValueError("PUBLICATION_EVIDENCE_SCHEMA")
            proof_refs = {}
            for proof_name in ("catalog", "notice_index"):
                proof = evidence[proof_name]
                proof_path = (source_root / proof["file"]).resolve()
                if not proof_path.is_relative_to(source_root / "raw") or not proof_path.is_file():
                    raise ValueError("PUBLICATION_EVIDENCE_SOURCE_PATH")
                proof_raw = proof_path.read_bytes()
                if len(proof_raw) > 20_000_000 or hashlib.sha256(proof_raw).hexdigest() != proof["sha256"]:
                    raise ValueError("PUBLICATION_EVIDENCE_SOURCE_HASH_SIZE")
                proof_suffix = ".pdf" if proof_raw.startswith(b"%PDF") else ".json"
                proof_target = rating_root / "sources" / (proof["sha256"] + proof_suffix)
                if proof_target.exists():
                    if proof_target.read_bytes() != proof_raw:
                        raise ValueError("PUBLICATION_EVIDENCE_DESTINATION_CONFLICT")
                else:
                    _publish(proof_target, proof_raw)
                proof_refs[proof_name] = {**proof, "file": proof_target.relative_to(rating_root).as_posix()}
            ref["publication_evidence"] = {**proof_refs, "article_id": evidence["article_id"]}
        if article:
            ref["article_id"] = article
        if key in result and result[key].get("report_end") and not ref.get("report_end"):
            ref["report_end"] = result[key]["report_end"]
        result[key] = ref
    refs = (
        list(result.values())
        + reuse_api_originals(code, source_root, rating_root, frozen)
        + reuse_regulatory_originals(code, source_root, rating_root, frozen, as_of)
        + reuse_database_nav(code, source_root, rating_root, frozen, as_of)
        + reuse_cash_originals(code, source_root, rating_root, frozen, as_of)
    )
    from app.services.fund_quality_v2_benchmark_sources import reuse_benchmark_originals

    refs += reuse_benchmark_originals(code, source_root, rating_root, frozen)
    from app.services.fund_quality_v2_component_prices import reuse_component_originals

    refs += reuse_component_originals(code, source_root, rating_root, frozen)
    halves = [r for r in refs if r["label"] == "historical-half" and r.get("report_end")]
    if halves:
        max(halves, key=lambda r: (r["report_end"], r["published_on"]))["label"] = "half"
    # 年报保留historical-annual附件角色；source_documents根据实际考察日
    # 重读正文并选为report_kind=annual主报告。不能在单个目录内提升角色后，
    # 导致它在跨目录合并或旧完整包的历史附件筛选中消失。
    for label in ("summary", "prospectus"):
        values = sorted([r for r in refs if r["label"] == label], key=lambda r: r["published_on"], reverse=True)
        for old in values[1:]:
            old["label"] = "historical-" + label
    return refs
