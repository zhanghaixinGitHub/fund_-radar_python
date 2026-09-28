"""按原公开日期协议准备补参考资料的独立 L20 数据包；本模块不导入模型。

原协议采用来源声明的历史公告日期并拦截已知后修订；不把“每条必须另有
独立历史存档”增加为门槛。来源声明不等于绝对证明没有隐性修订，证据层级
必须保留。当前副本相同不能单独放行，须同时核原来源、日期、原件关联、
来源使用范围、有效期和逐行完整输入。拟合预算与资料合格分开。
"""

import math
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from pypdf import PdfReader

from app.services.fund_002112_peer_admission import BUSINESS_FIELDS, exact_label, raw_path, report_inventory
from app.services.fund_002112_peer_material_repair import FrozenSources
from app.services.fund_002112_round3_data import COHORT, exposure, gate, nav_window, public_day, select_report
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once

OUTPUT = ROOT / "peer-training-ready/20260928-v1"
PACKAGE = ROOT / "peer-training-ready/20260928-v2"
ADMISSION = ROOT / "peer-admission/20260927-v1"
GAP = ROOT / "peer-gap-evidence/20260927-v1"
RULES = ROOT / "peer-admission-rules/20260928-v1"
PEERS = {"017493", "160323"}
ZONE = timezone(timedelta(hours=8))


def public_date(value):
    """未知公开日拒绝，不用下载时间、URL 日期或日历常识填补。"""
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("PUBLIC_DATE_UNKNOWN")
    return value


def nav_evidence(frozen, database, public_value):
    """核净值原值、原公告日和来源摘要；返回来源声明口径的最晚已知版本日。"""
    announced = public_date(frozen["ann_date"])
    day = public_date(frozen["date"])
    if not "2021-01-01" <= day <= "2023-12-31" or announced < day or announced > "2024-12-31":
        raise ValueError("NAV_DATE_SCOPE")
    values = [Decimal(str(v)) for v in (frozen["nav"], database["unit_nav"], public_value)]
    if any(not v.is_finite() or v <= 0 for v in values) or len(set(values)) != 1:
        raise ValueError("NAV_VALUE_CONFLICT")
    if database["nav_date"] != day or database["ann_date"] != announced:
        raise ValueError("NAV_DATE_CONFLICT")
    if database["content_hash"] != frozen["source_hash"] or not re.fullmatch(r"[0-9a-f]{64}", frozen["source_hash"]):
        raise ValueError("NAV_SOURCE_HASH_CONFLICT")
    version_days = [announced]
    for field in ("version_publication_date", "revised_at"):
        if field in frozen:
            version_days.append(public_date(frozen[field][:10] if frozen[field] else None))
    if frozen.get("revision_known") and len(version_days) == 1:
        raise ValueError("KNOWN_REVISION_DATE_UNKNOWN")
    return {
        "declared_public_date": announced,
        "known_version_public_date": max(version_days),
        "value": str(values[0]),
        "source_hash": frozen["source_hash"],
        "evidence_level": "DATED_PROVIDER_RECORD_WITH_PUBLIC_VALUE_CROSSCHECK",
        "independent_historical_capture_verified": False,
    }


def report_date_check(original_public, catalog_public, pdf_modified, native_modified=None):
    """原件、原目录公开日必须一致；已知修改日只能推迟，不能绕过或提前。

    PDF 元数据只作异常提示。它晚于原目录且没有能解释该修改的源端记录时，
    停止准入。正常较早生成时间不当作公开日。官网 publishDate 等字段不能
    仅凭英文名称推定含义，东方红按其展示代码确认的 activationDate 解释。
    """
    original_public, catalog_public = public_date(original_public), public_date(catalog_public)
    if original_public != catalog_public:
        raise ValueError("REPORT_PUBLICATION_CONFLICT")
    version = max(original_public, public_date(native_modified)) if native_modified else original_public
    if pdf_modified and public_date(pdf_modified) > version:
        raise ValueError("PDF_LATE_MODIFICATION_UNRESOLVED")
    return version


def correction_conflicts(items):
    """只放过明确属于合同/费率的目录修订；涉及净值、定期报告或不明对象即阻断。"""
    conflicts = []
    for item in items:
        title = re.sub(r"<[^>]+>", "", item.get("announcementTitle", item.get("title", "")))
        if not any(word in title for word in ("更正", "修订", "勘误", "修正")):
            continue
        if "基金合同" in title and not any(word in title for word in ("净值", "季度报告", "年度报告", "中期报告")):
            continue
        conflicts.append(title)
    return conflicts


def training_time_ok(row, start, base_publication):
    """保留原 mature_at，并补查基准与目标两个端点均已成熟，不以目标公告代替基准。"""
    pair_date = max(row["target"], public_date(base_publication), public_date(row["label_publication"]))
    pair_mature = (date.fromisoformat(pair_date) + timedelta(days=1)).isoformat() + "T08:00:00+08:00"
    limit = start + "T00:00:00+08:00"
    return row["target"] < start and row["mature_at"] < limit and pair_mature < limit


def assert_fit_budget(budget):
    """独立准备不可借用旧剩余拟合数；供未来执行入口在导入模型前调用。"""
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("NEW_FIT_BUDGET_NOT_AUTHORIZED")


class EvidenceFiles:
    """逐个记录真实读取的原文和声明，输出来源摘要；封存答案只保护字节不解析。"""

    def __init__(self):
        self.files = {}

    def load(self, path, expected=None):
        path = Path(path)
        self.check(path, expected)
        return read_json(path)

    def check(self, path, expected=None):
        path = Path(path)
        actual = file_hash(path)
        if expected and actual != expected:
            raise ValueError("EVIDENCE_CHANGED:" + path.name)
        self.files[str(path)] = actual


def public_nav_rows(cross, files):
    """从本次全部分页原始响应重建交叉核对表，检查页码、总数、日期边界与去重。"""
    result, total = {}, None
    for page, path in enumerate(cross["receipts"], 1):
        receipt = files.load(path)
        params = receipt["params"]
        if params["fundCode"] != cross["fund_code"] or params["endDate"] > "2023-12-31":
            raise ValueError("PUBLIC_NAV_REQUEST_SCOPE")
        raw = files.load(receipt["path"], receipt["sha256"])
        if raw["PageIndex"] != page or not raw["Data"] or raw["ErrCode"] != 0:
            raise ValueError("PUBLIC_NAV_PAGE_INVALID")
        total = raw["TotalCount"] if total is None else total
        if raw["TotalCount"] != total:
            raise ValueError("PUBLIC_NAV_TOTAL_CHANGED")
        for row in raw["Data"]["LSJZList"]:
            day = row["FSRQ"]
            if not params["startDate"] <= day <= params["endDate"] or day in result:
                raise ValueError("PUBLIC_NAV_DATE_OR_DUPLICATE")
            result[day] = row["DWJZ"]
    if len(result) != total or result != {r["date"]: r["unit_nav"] for r in cross["rows"]}:
        raise ValueError("PUBLIC_NAV_PAGINATION_INCOMPLETE")
    return result


def reports_review(files, original_reports):
    """用原生公告记录绑定全部 26 份原件；正文仍与原解析业务字段逐项相同。"""
    hx = files.load(OUTPUT / "hx-complete-catalog.json")
    df = files.load(OUTPUT / "df-periodic-catalog-review.json")
    temporary = files.load(OUTPUT / "df-temporary-catalog-review.json")
    native_hx = []
    for page in range(1, 9):
        name = "catalog-160323-standard-1" if page == 1 else f"catalog-160323-{page}"
        receipt = files.load(OUTPUT / "public-sources" / (name + "-receipt.json"))
        native_hx.extend(files.load(receipt["path"], receipt["sha256"])["announcements"])
    if native_hx != hx["all"] or len({r["announcementId"] for r in native_hx}) != 221:
        raise ValueError("NATIVE_CATALOG_RECONSTRUCTION_DIFFERS")
    periodic_raw = files.load(OUTPUT / "public-sources/df-periodic-30-1-raw.json")
    if any(r not in periodic_raw["contents"] for r in df["relevant"]):
        raise ValueError("ISSUER_CATALOG_BINDING_DIFFERS")
    native_temp = []
    for page in range(1, 9):
        native_temp.extend(files.load(OUTPUT / "public-sources" / f"df-temporary-30-{page}-raw.json")["contents"])
    if len(native_temp) != temporary["count"] or any(r not in native_temp for r in temporary["relevant"]):
        raise ValueError("ISSUER_CORRECTION_CATALOG_INCOMPLETE")
    conflicts = correction_conflicts(hx["all"] + temporary["relevant"])
    if conflicts:
        raise ValueError("DISCLOSURE_CORRECTION_NEEDS_REVIEW:" + str(conflicts))
    hx_index = {"https://static.cninfo.com.cn/" + r["adjunctUrl"]: r for r in hx["all"]}
    df_index = {"https://www.dfham.com" + r["url"]: r for r in df["relevant"]}
    native_q3 = files.load(OUTPUT / "hx-2023q3-parsed.json")
    supplements = files.load(GAP / "report-supplements-v2.json")
    supplements = {(r["fund_code"], r["report_end"], r["report_type"]): r for r in supplements}
    original_by_key = {(r["fund_code"], r["report_end"], r["report_type"]): r for r in original_reports.values()}
    review, report_pool = [], []
    for entry in files.load(GAP / "report-version-gap-worklist.json"):
        key = tuple(entry[k] for k in ("fund_code", "report_end", "report_type"))
        supplement = supplements.get(key)
        canonical = original_by_key.get(key)
        if supplement:
            parsed = files.load(supplement["parsed_file"], supplement["sha256"])
            if canonical and any(parsed[k] != canonical[k] for k in BUSINESS_FIELDS):
                raise ValueError("PRIMARY_TABLE_CONFLICT")
            primary_path, primary_sha = parsed["raw"]["path"], parsed["raw"]["sha256"]
            url = parsed["raw"]["url"]
            canonical = canonical or parsed
        else:
            primary_path, primary_sha, url = (
                entry["primary_pdf_path"],
                entry["primary_pdf_sha256"],
                entry["primary_pdf_url"],
            )
        if key == ("160323", "2023-09-30", "QUARTER"):
            if any(native_q3[k] != canonical[k] for k in BUSINESS_FIELDS):
                raise ValueError("ISSUER_AND_CNINFO_Q3_CONFLICT")
            primary_path, primary_sha, url = (native_q3["raw"][k] for k in ("path", "sha256", "url"))
        files.check(primary_path, primary_sha)
        reader = PdfReader(primary_path)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= 180:
            raise ValueError("PDF_IDENTITY_SCOPE")
        mod = (reader.metadata or {}).get("/ModDate", "")
        match = re.match(r"D:(\d{4})(\d{2})(\d{2})", str(mod))
        modified = "-".join(match.groups()) if match else None
        if key[0] == "017493":
            native = df_index[url]
            pub = datetime.fromtimestamp(int(native["activationDate"]) / 1000, ZONE).date().isoformat()
            changed = datetime.fromtimestamp(int(native["modificationDate"]) / 1000, ZONE).date().isoformat()
            identifier = native["contentId"]
        else:
            native = hx_index[url]
            if native["secCode"] != key[0]:
                raise ValueError("ANNOUNCEMENT_FUND_CONFLICT")
            pub = datetime.fromtimestamp(native["announcementTime"] / 1000, ZONE).date().isoformat()
            changed, identifier = None, native["announcementId"]
        version = report_date_check(public_day(canonical), pub, modified, changed)
        # 原始输入继续绑定原报告摘要；两份副本按业务字段一致建立别名，不制造版本冲突。
        canonical = {**canonical, "version_publication_date": version}
        report_pool.append(canonical)
        review.append(
            {
                "fund_code": key[0],
                "report_end": key[1],
                "report_type": key[2],
                "original_raw_sha256": canonical["raw"]["sha256"],
                "primary_sha256": primary_sha,
                "primary_path": str(primary_path),
                "source_url": url,
                "native_record_id": identifier,
                "public_date": pub,
                "version_publication_date": version,
                "pdf_modified": modified,
                "business_fields_equal": True,
                "declared_disclosure_admitted": True,
                "independent_historical_capture_verified": False,
            }
        )
    return report_pool, review


def prepare():
    """核原文/净值/行情后组装 5,137 行候选包和固定折；所有原门槛通过才写 ready。"""
    files = EvidenceFiles()
    contract = files.load(OUTPUT / "admission-contract.json")
    if contract["current_fit_budget"] != 0:
        raise ValueError("PREPARATION_FIT_BUDGET_CHANGED")
    previous = files.load(ADMISSION / "protocol.json")
    for path, sha in previous["files"].items():
        files.check(path, sha)
    # 旧交付和新下载分别验原始文件；忽略本轮生成物，防止证据自循环。
    for old in (GAP, RULES):
        manifest = files.load(old / "delivery-manifest.json")
        for path, sha in manifest["files"].items():
            files.check(path, sha)
    for receipt in sorted((OUTPUT / "public-sources").glob("*-receipt.json")):
        item = files.load(receipt)
        if item.get("path"):
            files.check(item["path"], item["sha256"])
    files.check(OUTPUT / "df-cmsPage.js")
    files.load(OUTPUT / "df-script-receipt.json")
    snapshot = files.load(previous["named"]["snapshot"])
    old = files.load(previous["named"]["third_inputs"])
    folds = files.load(previous["named"]["third_folds"])
    db = files.load(previous["named"]["database"])
    policy = db["source"]
    if not policy["enabled"] or "fund_nav" not in policy["authorized_api_names"] or policy["retention_days"] != 365:
        raise ValueError("SOURCE_SCOPE_NOT_VERIFIED")
    if datetime.fromisoformat(db["checked_at"]) + timedelta(days=policy["retention_days"]) <= datetime.now(ZONE):
        raise ValueError("NAV_VERIFIED_SNAPSHOT_EXPIRED")
    native_db = {(r["fund_code"], r["nav_date"]): r for r in db["nav_rows"]}
    days = sorted(snapshot["indices"]["000300.SH"]["rows"])
    mapping = {c: {r["date"]: r for r in snapshot["funds"][c]["nav"]["rows"]} for c in PEERS}
    public_values = {}
    for code in PEERS:
        cross = files.load(OUTPUT / f"nav-crosscheck-{code}.json")
        if cross["differences"] or cross["fund_code"] != code or len(cross["rows"]) != cross["raw_count"]:
            raise ValueError("PUBLIC_NAV_CROSSCHECK_FAILED")
        public_values.update({(code, day): value for day, value in public_nav_rows(cross, files).items()})
    nav_checks = {}
    requirements = files.load(RULES / "requirements.json")
    for req in requirements.values():
        if req["type"] != "NAV":
            continue
        code, day = req["fund_code"], req["business_date"]
        nav_checks[code, day] = nav_evidence(mapping[code][day], native_db[code, day], public_values[code, day])
    original_reports = report_inventory(files.load(previous["named"]["worklist"]), snapshot)
    for report in original_reports.values():
        files.check(raw_path(report), report["raw"]["sha256"])
    reports, report_checks = reports_review(files, original_reports)
    source = FrozenSources(files.load(previous["named"]["repair_protocol"]))
    source.indices = snapshot["indices"]
    old_rows = files.load(ADMISSION / "row-admission.json")
    added_inputs = files.load(GAP / "additional-inputs-not-admitted.json")
    label_rows = old_rows + files.load(GAP / "additional-labels-not-admitted.json")
    labels = {(r["fund_code"], r["target"]): r for r in label_rows}
    vectors = files.load(ADMISSION / "candidate-inputs-not-admitted.json") + added_inputs
    originals = {(r["fund_code"], r["target"]): r for r in old["train"]}
    admitted, row_checks = [], []
    for vector in sorted(vectors, key=lambda r: (r["fund_code"], r["target"])):
        code, target = vector["fund_code"], vector["target"]
        pos = days.index(target)
        window = nav_window(mapping[code], days, target)
        report = select_report(
            [r for r in reports if r["fund_code"] == code], snapshot["funds"][code]["catalog"], target
        )
        needed = days[pos - 21 : pos]
        x = window["x"] + exposure(report, source.quotes, source.indices, needed)
        if x != vector["x"] or not all(math.isfinite(v) for v in x) or len(x) != 20:
            raise ValueError("FROZEN_20_FEATURES_CHANGED")
        for day in window["nav_dates"]:
            if nav_checks[code, day]["known_version_public_date"] >= target:
                raise ValueError("LATE_NAV_VERSION_IN_INPUT")
        label = exact_label(mapping[code][days[pos - 1]], mapping[code][target], days)
        if any(label[k] != labels[code, target][k] for k in ("base", "target", "actual_direction", "mature_at")):
            raise ValueError("FROZEN_LABEL_CHANGED")
        row = {
            "fund_code": code,
            "family": snapshot["funds"][code]["family"],
            "target": target,
            "x": x,
            "nav": window,
            "quote_dates": needed,
            "report_end": report["report_end"],
            "report_publication": public_day(report),
            "report_sha256": report["raw"]["sha256"],
            **label,
        }
        if (code, target) in originals:
            original = originals[code, target]
            if original["x"] != x or original["actual_direction"] != label["actual_direction"]:
                raise ValueError("ORIGINAL_112_ROWS_CHANGED")
            row = original  # 原 112 条整体保留，准入核查附在旁表，不往原行塞新状态。
        admitted.append(row)
        row_checks.append(
            {
                "fund_code": code,
                "target": target,
                "input_sha256": digest(x),
                "base_publication": label["base_publication"],
                "declared_source_admitted": True,
                "report_version_publication": public_day(report),
                "original_row_preserved": (code, target) in originals,
            }
        )
    files.files.update(source.verified)
    pool = sorted(
        [r for r in old["train"] if r["fund_code"] not in PEERS] + admitted, key=lambda r: (r["target"], r["fund_code"])
    )
    family = {code: snapshot["funds"][code]["family"] for code in COHORT}
    index = {(r["fund_code"], r["target"]): r for r in pool}
    if len(index) != len(pool) or len(admitted) != 830 or sum(r["original_row_preserved"] for r in row_checks) != 112:
        raise ValueError("CANDIDATE_IDENTITIES_CHANGED")
    prepared_folds = []
    # 两个答案端点的已知版本日均受训练截止约束；原 mature_at 与公告日不改写。
    base_public = {
        (r["fund_code"], r["target"]): max(
            nav_checks[r["fund_code"], r["base"]]["known_version_public_date"],
            nav_checks[r["fund_code"], r["target"]]["known_version_public_date"],
        )
        for r in admitted
    }
    for fold in folds:
        train = [originals[tuple(k)] for k in fold["train_ids"] if k[0] not in PEERS]
        train += [r for r in admitted if training_time_ok(r, fold["start"], base_public[r["fund_code"], r["target"]])]
        train.sort(key=lambda r: (r["target"], r["fund_code"]))
        data_gate = gate(train, family)
        exams = (
            old["development"]
            if fold["name"] == "FULL"
            else [originals["002112", day] for day in fold["expected_dates"]]
        )
        if not data_gate["passed"] or [r["target"] for r in exams] != fold["expected_dates"]:
            raise ValueError("ORIGINAL_DATA_GATE_OR_EXAM_COVERAGE_FAILED")
        count = Counter(r["family"] for r in train)
        # 仅产生新包训练必须使用的权重，不重复旧折影响分析，不变更原公式。
        weights = [len(train) / (11 * count[r["family"]]) for r in train]
        prepared_folds.append(
            {
                "name": fold["name"],
                "start": fold["start"],
                "train_ids": [[r["fund_code"], r["target"]] for r in train],
                "weights": weights,
                "gate": data_gate,
                "exam": exams,
                "original_exam_identity_sha256": digest(fold["expected_dates"]),
                "train_sha256": digest(train),
                "exam_sha256": digest(exams),
            }
        )
    if any(m.startswith("sklearn") or m.endswith("fund_002112_round3_model") for m in sys.modules):
        raise ValueError("MODEL_IMPORTED_DURING_PREPARATION")
    plan = files.load(GAP / "independent-preparation.json")
    decision = {
        "material_ready": True,
        "fit_execution_allowed": False,
        "current_fit_budget": 0,
        "actual_new_fits": 0,
        "cumulative_actual_fits": 52,
        "train_pool_rows": len(pool),
        "admitted_peer_rows": len(admitted),
        "net_added_peer_rows": len(admitted) - 112,
        "nav_dependencies_checked": len(nav_checks),
        "reports_checked": len(report_checks),
        "original_peer_rows_preserved": 112,
        "original_exam_days": [50, 51, 60, 230],
        "folds": {f["name"]: {"rows": len(f["train_ids"]), "gate": f["gate"]} for f in prepared_folds},
        "basis": "ORIGINAL_DECLARED_PUBLICATION_PROTOCOL_WITH_NEW_SOURCE_CHECKS",
        "independent_historical_capture_verified": False,
        "remaining_material_blockers_under_original_protocol": [],
        "database_refresh": "Timed out; immutable previously verified snapshot used offline",
        "status": "MATERIAL_READY_AWAIT_NEW_FIT_BUDGET",
    }
    outputs = {
        "inputs.json": {"train": pool, "development": old["development"]},
        "folds.json": prepared_folds,
        "report-admission.json": report_checks,
        "nav-admission.json": [{"fund_code": c, "date": d, **v} for (c, d), v in sorted(nav_checks.items())],
        "row-admission.json": row_checks,
        "decision.json": decision,
        "independent-preparation.json": plan,
        "sources.json": {"files": files.files, "proof_level": contract},
    }
    return outputs


def save_prepared(outputs):
    """所有准入及原门槛完成后才发出不可变文件；资料合格不自动授权拟合。"""
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        save_once(PACKAGE / name, value)
    anchors = {str(PACKAGE / name): file_hash(PACKAGE / name) for name in outputs}
    save_once(
        PACKAGE / "ready.json",
        {
            "material_ready": outputs["decision.json"]["material_ready"],
            "fit_execution_allowed": False,
            "current_fit_budget": 0,
            "artifact_hashes": anchors,
            "actual_new_fits": 0,
        },
    )
