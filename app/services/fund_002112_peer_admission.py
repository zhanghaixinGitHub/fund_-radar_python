"""002112 补参考资料的独立准入检查；不联网、不写库、不载入或拟合模型。

这里区分数值重放成功、历史版本证据及训练准入。报告印刷日期、PDF 元数据和
当前数据库相同值都可以支持核查，但不能单独证明该内容版本当时已经公开。
输出中的 actual_direction 是 2021—2023 冻结净值的核查答案，不是历史预测。
"""

import ast
import math
import re
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.integrations.fund_report_sections_v2 import parse_text
from app.services.fund_002112_peer_coverage_audit import calendar_days
from app.services.fund_002112_peer_fold_impact import FOLDS, PEERS, maturity, time_status
from app.services.fund_002112_peer_material_repair import FrozenSources
from app.services.fund_002112_round3_data import (
    CLASSES,
    COHORT,
    FEATURES,
    TIE_ORDER,
    exposure,
    nav_window,
    select_report,
)
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once

OUTPUT = ROOT / "peer-admission/20260927-v1"
PREVIOUS = ROOT / "peer-fold-impact/20260927-v1"
REPAIR = ROOT / "peer-material-repair/20260927-v1"
THIRD = ROOT / "round3-repair-runs/002112-r3r-46b69f78f27a82dc30acbe66"
BUSINESS_FIELDS = (
    "fund_code",
    "fund_master_code",
    "report_end",
    "report_type",
    "published_date",
    "holdings",
    "holding_count",
    "disclosed_nav_pct",
    "stock_nav_pct",
    "stock_value_cny",
    "full_stock_disclosure",
    "assets",
    "reported_industries",
)


def checked(path: Path | str, expected: str):
    """先验字节摘要再解析，拒绝来源被替换；从不追随 latest/ready 指针。"""
    if file_hash(path) != expected:
        raise ValueError("FROZEN_SOURCE_CHANGED:" + Path(path).name)
    return read_json(path)


def exact_label(base: dict, target: dict, days: list[str]) -> dict:
    """只在原参考范围内核答案：相邻交易日原单位净值，完全相等才为持平。

    标签和输入分别保存。两端公告日期均保留；基准净值晚于目标公告时必须另行
    阻断，不能沿原目标公告日把尚未公开的基准值当作已成熟标签。
    """
    t, u = base["date"], target["date"]
    if not "2021-01-04" <= u <= "2023-12-29" or t != days[days.index(u) - 1]:
        raise ValueError("LABEL_OUTSIDE_SCOPE_OR_NOT_ADJACENT")
    values = [Decimal(str(r["nav"])) for r in (base, target)]
    if any(not v.is_finite() or v <= 0 for v in values):
        raise ValueError("NON_POSITIVE_OR_NONFINITE_NAV")
    for row in (base, target):
        day = row.get("ann_date")
        if not day or date.fromisoformat(day).isoformat() != day or day > "2024-12-31":
            raise ValueError("LABEL_PUBLICATION_UNKNOWN_OR_OUTSIDE_SCOPE")
        if day < row["date"] or not re.fullmatch(r"[0-9a-f]{64}", row.get("source_hash", "")):
            raise ValueError("LABEL_PUBLICATION_OR_SOURCE_HASH_INVALID")
    a, b = values
    return {
        "actual_direction": "UP" if b > a else "DOWN" if b < a else "FLAT",
        "base": t,
        "target": u,
        "base_unit_nav": str(a),
        "target_unit_nav": str(b),
        "base_publication": base["ann_date"],
        "label_publication": target["ann_date"],
        "base_source_hash": base["source_hash"],
        "target_source_hash": target["source_hash"],
        "mature_at": maturity(u, target["ann_date"]),
        "both_navs_public_by_label_date": base["ann_date"] <= target["ann_date"],
    }


def raw_path(report: dict) -> Path:
    raw = report["raw"]
    return Path(raw["path"]) if raw.get("path") else ROOT / raw["file"]


def report_inventory(worklist: dict, snapshot: dict) -> dict:
    """严格使用保存的 24 份清单；新增记录依赖的 18 份优先，不能偷偷补换资料。"""
    if worklist["report_count"] != 24 or len(worklist["reports"]) != 24:
        raise ValueError("REPORT_WORKLIST_CHANGED")
    result = {}
    for item in worklist["reports"]:
        report = (
            read_json(item["parsed_file"])
            if item["parsed_file"]
            else next(
                r for r in snapshot["funds"][item["fund_code"]]["reports"] if r["raw"]["sha256"] == item["raw_sha256"]
            )
        )
        if any(report[k] != item[k] for k in ("fund_code", "report_end", "report_type")):
            raise ValueError("REPORT_IDENTITY_CHANGED")
        if report["raw"]["sha256"] != item["raw_sha256"] or file_hash(raw_path(report)) != item["raw_sha256"]:
            raise ValueError("REPORT_RAW_CHANGED")
        if item["raw_sha256"] in result:
            raise ValueError("DUPLICATE_REPORT")
        result[item["raw_sha256"]] = report
    if sum(any(r["additional_dates_vs_original_per_fold"].values()) for r in worklist["reports"]) != 18:
        raise ValueError("PRIORITY_REPORT_COUNT_CHANGED")
    return result


def report_review(item: dict, report: dict, attachment_page: dict) -> dict:
    """重放优先报告全部表格，记录能支持版本判断的证据与尚缺的证据。

    PDF 内部时间可修改，转载站“自下载后未变”只说明 2026 年后的稳定性。
    这些信息不等同于来源提供了历史内容版本或更正记录，因而不自动打开准入。
    """
    path, raw = raw_path(report), report["raw"]
    priority = any(item["additional_dates_vs_original_per_fold"].values())
    meta, pages, page_date, binding = {}, None, None, False
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= 180:
            raise ValueError("PDF_ENCRYPTED_OR_PAGE_LIMIT")
        pages = [p.extract_text() or "" for p in reader.pages]
        meta = {k: str(v) for k, v in (reader.metadata or {}).items() if k in ("/CreationDate", "/ModDate")}
    elif "read.php" in raw["url"]:
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        pages = [soup.select_one("#sohu_content").get_text("\n", strip=True)]
        title = soup.select_one("h1").get_text(" ", strip=True)
        stamp = soup.select_one(".article_info > .txt .c")
        match = re.search(r"20\d{2}-\d{2}-\d{2}", stamp.text if stamp else "")
        if not match or digest({"title": title, "text": pages[0], "published": match[0]}) != raw["content_hash"]:
            raise ValueError("REPRINT_CONTENT_HASH_CHANGED")
        page_date = match[0]
    if pages:
        replay = parse_text(
            pages,
            report["title"],
            fund_code=report["fund_code"],
            fund_name="东方红新动力" if report["fund_code"] == "017493" else "华夏磐泰",
            master_code=report["fund_master_code"],
        )
        if any(replay[k] != report[k] for k in BUSINESS_FIELDS):
            raise ValueError("REPORT_BUSINESS_REPLAY_MISMATCH:" + report["report_end"])
        product = re.sub(r"\s", "", "\n".join(pages[:12]))
        master = re.search(r"基金主代码[:：]?" + report["fund_master_code"], product)
        if (
            not master
            or report["fund_code"] not in product[master.start() : master.end() + 3500]
            or "混合" not in report["title"]
        ):
            raise ValueError("REPORT_SHARE_IDENTITY_MISSING")
    elif priority:
        raise ValueError("PRIORITY_REPORT_REPLAY_MISSING")
    source = report.get("source", {})
    catalogue = source.get("published_date")
    dates = [d for d in (catalogue, page_date, report["published_date"], report.get("source_publication_date")) if d]
    public = max(dates)
    if public != report.get("source_publication_date", report["published_date"]):
        raise ValueError("CONSERVATIVE_REPORT_PUBLICATION_CHANGED")
    primary_path = re.search(r"/finalpage/(\d{4}-\d{2}-\d{2})/\d+\.PDF$", raw["url"], re.I)
    dated_primary = bool(
        urlparse(raw["url"]).hostname == "static.cninfo.com.cn" and primary_path and primary_path[1] == public
    )
    if raw["url"] == "https://accountquery.chinaamc.com/front/ui/contentcore/resource/download?ID=79258":
        html = BeautifulSoup(Path(attachment_page["path"]).read_bytes(), "html.parser")
        binding = (
            any(urljoin(attachment_page["url"], a["href"]) == raw["url"] for a in html.find_all("a", href=True))
            and public in html.get_text()
        )
    pdf_dates = [v[2:10] for v in meta.values() if re.match(r"D:\d{8}", v)]
    metadata_consistent = len(pdf_dates) == 2 and all(d <= public.replace("-", "") for d in pdf_dates)
    gaps = ["HISTORICAL_CONTENT_VERSION_BINDING_NOT_ESTABLISHED"]
    if meta and not metadata_consistent:
        gaps.append("PDF_TIMESTAMP_MISSING_OR_AFTER_DECLARED_PUBLICATION")
    if not path.suffix.lower() == ".pdf":
        gaps.append("REPRINT_HAS_NO_ORIGINAL_VERSION_OR_CORRECTION_HISTORY")
    elif not dated_primary and not binding:
        gaps.append("DATED_PRIMARY_NOTICE_TO_EXACT_ATTACHMENT_NOT_CAPTURED")
    if raw.get("revised_after_receipt") or any(t in report["title"] for t in ("更正", "修订")):
        gaps.append("REVISION_REQUIRES_ITS_OWN_PUBLICATION_DATE")
    return {
        "fund_code": item["fund_code"],
        "report_end": item["report_end"],
        "report_type": item["report_type"],
        "priority_new_row_dependency": priority,
        "raw_sha256": item["raw_sha256"],
        "raw_path": str(path),
        "url": raw["url"],
        "parsed_file": item["parsed_file"],
        "source_notice_id": source.get("ID", source.get("article_id")),
        "table_replay_equal": True if pages else None,
        "holding_count": report["holding_count"],
        "declared_publication": report["published_date"],
        "catalogue_publication": catalogue,
        "reprint_publication": page_date,
        "conservative_publication_unchanged": public,
        "pdf_metadata": meta,
        "pdf_metadata_not_after_publication": metadata_consistent if meta else None,
        "dated_primary_pdf_url": dated_primary,
        "dated_issuer_attachment_binding": binding,
        "documentary_support": dated_primary or binding,
        "historical_version_verified": False,
        "system_download_in_history_required": False,
        "training_eligible": False,
        "gaps": gaps,
        "additional_dates_vs_original_per_fold": item["additional_dates_vs_original_per_fold"],
        "next_evidence": "来源披露的内容版本/更正记录或可核对的历史公开副本，绑定本份正文；不要求本系统当年下载。",
    }


def check_current_nav(row: dict, current: dict | None) -> bool:
    return current is not None and (
        row["date"] == current["nav_date"]
        and Decimal(row["nav"]) == Decimal(current["unit_nav"])
        and row["ann_date"] == current["ann_date"]
        and row["source_hash"] == current["content_hash"]
    )


def recipe_from_source(path: Path) -> dict:
    """静态读取固定配方，不导入 sklearn/模型模块，也不调用旧训练入口。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "LOGISTIC"
    }
    return values["LOGISTIC"]


def ensure_zero_budget(protocol: dict):
    if protocol["current_fit_budget"] != 0 or protocol["proposed_fit_cap"] != 12:
        raise ValueError("PREPARATION_ONLY_BUDGET_CANNOT_BE_OPENED")
    if protocol["cohort"] != list(COHORT) or protocol["folds"] != FOLDS:
        raise ValueError("FIXED_RESEARCH_SCOPE_CHANGED")


def verify_indices(sources, indices):
    """把冻结的两指数行逐项追溯至既有回执原文，仍只使用截至 2024 的行情。"""
    for code, index in indices.items():
        replay = {}
        for receipt in index["receipts"]:
            if receipt["params"].get("start_date", "") > "20241231":
                continue
            if receipt.get("api") != "index_daily":
                raise ValueError("INDEX_SOURCE_API_CHANGED")
            if datetime.fromisoformat(receipt["expires_at"]) <= datetime.now().astimezone():
                raise ValueError("INDEX_SOURCE_EXPIRED")
            path = Path(receipt["raw_path"]) if receipt.get("raw_path") else ROOT / receipt["file"]
            if file_hash(path) != receipt["sha256"]:
                raise ValueError("INDEX_RAW_CHANGED")
            raw = sources.market_read(path)["data"]
            for item in raw["items"]:
                row = dict(zip(raw["fields"], item, strict=True))
                if row.pop("ts_code") != code:
                    raise ValueError("INDEX_CODE_CHANGED")
                day = datetime.strptime(row.pop("trade_date"), "%Y%m%d").date().isoformat()
                if day <= "2024-12-31":
                    if day in replay and replay[day] != row:
                        raise ValueError("INDEX_CONFLICTING_VERSION")
                    replay[day] = row
        if replay != index["rows"]:
            raise ValueError("INDEX_SOURCE_REPLAY_MISMATCH")


def run(output: Path = OUTPUT, progress=print) -> dict:
    """执行资料准入、逐条重放和独立方案准备；不生成可训练包，不预留拟合槽位。"""
    protocol = read_json(output / "protocol.json")
    ensure_zero_budget(protocol)
    for path, sha in protocol["files"].items():
        if file_hash(path) != sha:
            raise ValueError("FROZEN_SOURCE_CHANGED:" + Path(path).name)
    load = lambda name: checked(protocol["named"][name], protocol["files"][protocol["named"][name]])  # noqa: E731
    snapshot, worklist, db = load("snapshot"), load("worklist"), load("database")
    reports = report_inventory(worklist, snapshot)
    report_rows = [
        report_review(item, reports[item["raw_sha256"]], load("issuer_page"))
        for item in sorted(
            worklist["reports"], key=lambda r: not any(r["additional_dates_vs_original_per_fold"].values())
        )
    ]
    save_once(output / "report-admission.json", report_rows)
    progress("24 份报告身份核对完成，优先 18 份表格已从原文重放。")
    sources = FrozenSources(load("repair_protocol"))
    sources.indices = snapshot["indices"]
    verify_indices(sources, sources.indices)
    days = calendar_days([load("calendar_older"), load("calendar_recent")])
    old = load("third_inputs")
    old_index = {(r["fund_code"], r["target"]): r for r in old["train"]}
    if len(old_index) != 4419 or len(old["development"]) != 230 or set(snapshot["funds"]) != set(COHORT):
        raise ValueError("ORIGINAL_DATA_SCOPE_CHANGED")
    current = {(r["fund_code"], r["nav_date"]): r for r in db.get("nav_rows", [])}
    daily, evidence, inputs = load("current_daily"), [], []
    if len({(r["fund_code"], r["target"]) for r in daily}) != 1454:
        raise ValueError("ORIGINAL_DAILY_UNIVERSE_CHANGED")
    exclusions = [r for r in daily if r["after"]["status"] != "INPUT_COMPLETE"]
    unchanged = 0
    for code in PEERS:
        fund = snapshot["funds"][code]
        mapping = {r["date"]: r for r in fund["nav"]["rows"]}
        if max(mapping) > "2024-12-31":
            raise ValueError("SEALED_NAV_SCOPE")
        pool = [r for r in reports.values() if r["fund_code"] == code]
        for n, entry in enumerate(
            r for r in daily if r["fund_code"] == code and r["after"]["status"] == "INPUT_COMPLETE"
        ):
            u = entry["target"]
            i = days.index(u)
            t = days[i - 1]
            window = nav_window(mapping, days, u)
            selected = select_report(pool, fund["catalog"], u)
            label = exact_label(mapping[t], mapping[u], days)
            quote_dates = days[i - 21 : i]
            x = window["x"] + exposure(selected, sources.quotes, sources.indices, quote_dates)
            if (
                len(x) != 20
                or not all(math.isfinite(v) for v in x)
                or digest(x) != entry["after"]["input_vector_sha256"]
            ):
                raise ValueError("RECOVERED_INPUT_REPLAY_MISMATCH")
            if selected["raw"]["sha256"] != entry["after"]["report_sha256"]:
                raise ValueError("SELECTED_REPORT_CHANGED")
            key = (code, u)
            old_row = old_index.get(key)
            if old_row:
                if old_row["x"] != x or old_row["actual_direction"] != label["actual_direction"]:
                    raise ValueError("ORIGINAL_PEER_ROW_CHANGED")
                unchanged += 1
            dependencies = sorted(set(window["nav_dates"] + [t, u]))
            equal = all(check_current_nav(mapping[d], current.get((code, d))) for d in dependencies)
            gaps = ["REPORT_HISTORICAL_VERSION_PENDING", "NAV_INPUT_AND_LABEL_HISTORICAL_VERSION_PENDING"]
            if not equal:
                gaps.append("CURRENT_DATABASE_VALUE_OR_HASH_DIFFERS_OR_UNAVAILABLE")
            if not label["both_navs_public_by_label_date"]:
                gaps.append("BASE_NAV_PUBLISHED_AFTER_TARGET_NAV")
            evidence.append(
                {
                    "fund_code": code,
                    **label,
                    "old_row_preserved": bool(old_row),
                    "input_vector_sha256": digest(x),
                    "nav_window": {k: v for k, v in window.items() if k != "x"},
                    "nav_dependency_dates": dependencies,
                    "current_database_equal": equal,
                    "report_sha256": selected["raw"]["sha256"],
                    "quote_dates": quote_dates,
                    "fold_time": {
                        f: time_status(u, label["label_publication"], start)[0] for f, start in FOLDS.items()
                    },
                    "training_eligible": False,
                    "label_numeric_check_passed": True,
                    "gaps": gaps,
                }
            )
            # 输入和答案分文件；无任何预测字段，更不会写入历史 input-only 记录。
            inputs.append(
                {
                    "fund_code": code,
                    "family": fund["family"],
                    "target": u,
                    "x": x,
                    "input_sha256": digest(x),
                    "training_eligible": False,
                }
            )
            if n % 150 == 0:
                progress(f"{code} 资料逐行核验 {n}；新增拟合 0。")
    if len(evidence) != 750 or unchanged != 112:
        raise ValueError("EXPECTED_CANDIDATE_OR_ORIGINAL_PEER_COUNT_CHANGED")
    source_review = {
        "database_status": db["status"],
        "source": db.get("source"),
        "current_nav_rows": len(current),
        "peer_historical_version_inventory": db.get("version_inventory"),
        "numeric_consistency_is_not_historical_version_evidence": True,
        "input_and_label_version_status": "UNRESOLVED",
        "system_historical_download_required": False,
        "market_raw_files_verified": len(sources.verified),
        "quote_source_manifest": load("repair_protocol")["sources"]["third_source_manifest"],
        "new_provider_requests": 0,
        "database_writes": 0,
    }
    # 不改变旧折权重，也不再统计权重：这里只冻结候选身份和考试身份，复用旧折审计。
    folds = load("third_folds")
    preparation = prepare_contract(protocol, old, folds, evidence, load)
    summary = {
        "report_count": len(report_rows),
        "priority_reports": 18,
        "priority_table_replays": sum(
            r["priority_new_row_dependency"] and r["table_replay_equal"] is True for r in report_rows
        ),
        "dated_primary_document_support": sum(r["documentary_support"] for r in report_rows),
        "priority_dated_primary_document_support": sum(
            r["documentary_support"] and r["priority_new_row_dependency"] for r in report_rows
        ),
        "candidate_records": len(evidence),
        "additional_unique_records_vs_old": sum(not r["old_row_preserved"] for r in evidence),
        "original_peer_rows_unchanged": unchanged,
        "numeric_labels_checked": len(evidence),
        "current_database_equal_records": sum(r["current_database_equal"] for r in evidence),
        "base_publication_later_than_target": sum(not r["both_navs_public_by_label_date"] for r in evidence),
        "labels_by_fund": {
            c: dict(Counter(r["actual_direction"] for r in evidence if r["fund_code"] == c)) for c in PEERS
        },
        "remaining_exclusions": {
            c: dict(Counter(r["after"]["status"] for r in exclusions if r["fund_code"] == c)) for c in PEERS
        },
        "newly_admitted_records": 0,
        "training_ready": False,
        "current_fit_budget": 0,
        "new_fits": 0,
        "cumulative_fits": 52,
    }
    for name, value in {
        "row-admission.json": evidence,
        "candidate-inputs-not-admitted.json": inputs,
        "source-review.json": source_review,
        "market-sources-verified.json": sources.verified,
        "excluded-dates.json": exclusions,
        "independent-preparation.json": preparation,
        "summary.json": summary,
        "decision.json": {
            "status": "ADMISSION_REVIEW_COMPLETE_STOP_BEFORE_FIT",
            **summary,
            "stop_reasons": [
                "REPORT_CONTENT_VERSION_NOT_BOUND_TO_HISTORICAL_PUBLICATION",
                "NAV_CONTENT_VERSION_NOT_BOUND_TO_HISTORICAL_INPUT_AND_LABEL_AVAILABILITY",
                "CURRENT_FIT_BUDGET_ZERO",
            ],
            "next_step": "补齐逐报告及逐净值版本证据后另存准入版本；通过前不形成训练包、不占用任何拟合槽位。",
            "after_pass": "合格更优模型按已有授权完成必要入库及实际使用；失败候选不得采用。",
        },
    }.items():
        save_once(output / name, value)
    return summary


def prepare_contract(protocol, old, folds, evidence, load):
    """可重放的独立准备：锁定原对照、参数、日期、九基金身份及 12 个建议槽位。

    不复制旧拟合账本，不载入模型。2024 原资料 N7/L20 的六次主跑/复现必须是
    新实验的条件阶段，不能把旧第三轮没有运行的 FULL 结果冒充为现成对照。
    """
    checks = {}
    original_index = {(r["fund_code"], r["target"]): r for r in old["train"]}
    for fold in folds:
        name = fold["name"]
        training = [original_index[tuple(k)] for k in fold["train_ids"]]
        if digest(training) != fold["train_hash"]:
            raise ValueError("ORIGINAL_TRAINING_IDENTITY_CHANGED")
        exam = [
            r
            for r in (old["development"] if name == "FULL" else old["train"])
            if r["fund_code"] == "002112" and r["target"] in fold["expected_dates"]
        ]
        if digest(exam) != fold["exam_hash"] or len(exam) != fold["exam_count"]:
            raise ValueError("FROZEN_EXAM_CHANGED")
        row = {
            "exam_dates": fold["expected_dates"],
            "exam_hash": fold["exam_hash"],
            "other_nine_train_ids": [k for k in fold["train_ids"] if k[0] not in PEERS],
        }
        if name in FOLDS:
            row["candidate_peer_ids_pending_admission"] = [
                [r["fund_code"], r["target"]] for r in evidence if r["fold_time"][name] == "TIME_ELIGIBLE"
            ]
            row["controls"] = {}
            for model in ("N7", "L20"):
                main, replay = [load(f"{name}-{model}-{mode}") for mode in ("main", "replay")]
                for result in (main, replay):
                    if [(r["target"], r["actual_direction"], r["input_hash"]) for r in result["exam"]] != [
                        (r["target"], r["actual_direction"], digest(r)) for r in exam
                    ]:
                        raise ValueError("CONTROL_EXAM_IDENTITY_CHANGED")
                    if [r["input_hash"] for r in result["train"]] != [digest(r) for r in training]:
                        raise ValueError("CONTROL_TRAINING_IDENTITY_CHANGED")
                if main != replay:
                    raise ValueError("CONTROL_REPLAY_NOT_EXACT")
                row["controls"][model] = {
                    "source": protocol["named"][f"{name}-{model}-main"],
                    "prediction_hash": digest(main),
                    "replay_equal": True,
                }
        else:
            row["stage"] = "ONLY_AFTER_161_DAY_GATE_PASS_AND_NEW_BUDGET"
            row["candidate_peer_ids_pending_admission"] = [
                [r["fund_code"], r["target"]]
                for r in evidence
                if r["target"] < "2024-01-01"
                and r["label_publication"] < "2024-01-01"
                and r["mature_at"] < "2024-01-01T00:00:00+08:00"
            ]
        checks[name] = row
    slots = [f"{fold}-L20_RECOVERED-{mode}" for fold in FOLDS for mode in ("main", "replay")]
    slots += [
        f"FULL-{model}-{mode}"
        for model in ("L20_RECOVERED", "L20_ORIGINAL", "N7_ORIGINAL")
        for mode in ("main", "replay")
    ]
    return {
        "status": "PREPARED_BUT_DATA_NOT_ADMITTED",
        "current_execute_budget": 0,
        "proposed_max_fits": 12,
        "proposed_slots_not_reserved": slots,
        "actual_new_fit_attempts": [],
        "features": list(FEATURES),
        "classes": list(CLASSES),
        "tie_order": list(TIE_ORDER),
        "recipe": recipe_from_source(Path(protocol["named"]["model_code"])),
        "scaler": "weighted StandardScaler fitted on train only; same original implementation",
        "weight_rule": "N/(11*n_family), unchanged; previous fold statistics reused without recalculation",
        "folds": checks,
        "frozen_plan": load("next_plan"),
        "source_protocol_content_sha256": digest(protocol),
        "inherited_environment": load("third_protocol")["environment"],
        "current_preparation_does_not_import_or_execute_models": True,
    }
