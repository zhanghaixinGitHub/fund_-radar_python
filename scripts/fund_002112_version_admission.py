"""零拟合版本准入命令。freeze/review 可复放；没有训练、联网或修改旧结果入口。"""

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.fund_002112_round3_data import nav_window
from app.services.fund_002112_version_admission import load_evidence, row_at
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json, save_once

OUTPUT = ROOT / "peer-admission-rules/20260928-v1"
PREVIOUS = ROOT / "peer-admission/20260927-v1"
GAP = ROOT / "peer-gap-evidence/20260927-v1"


def freeze():
    """绑定旧准入、补证、协议与本次代码；真实已审阅版本证据目前为空。

    证据可以来自今天取得的第三方历史版本。先核原文和版本时间语义，再在新
    目录冻结清单；不得编辑本目录协议或用 --eligible 等开关直接授予资格。
    """
    target = OUTPUT / "protocol.json"
    if target.exists():
        return read_json(target)
    prior = read_json(PREVIOUS / "protocol.json")
    files = dict(prior["files"])
    for folder, names in (
        (PREVIOUS, ["protocol.json", "row-admission.json", "candidate-inputs-not-admitted.json"]),
        (
            GAP,
            [
                "report-version-gap-worklist.json",
                "nav-version-gap-worklist.json",
                "additional-inputs-not-admitted.json",
                "additional-labels-not-admitted.json",
                "report-supplements-v2.json",
                "independent-preparation.json",
                "independent-audit.json",
                "df-issuer-browser-catalog.json",
                "hx-cninfo-report-links.json",
                "hx-cninfo-extra-report-links.json",
                "delivery-manifest.json",
            ],
        ),
        (OUTPUT, ["source-feasibility.json", "rule-trace.json"]),
        (
            ROOT.parents[1],
            [
                "app/services/fund_002112_version_admission.py",
                "scripts/fund_002112_version_admission.py",
                "tests/test_fund_002112_version_admission.py",
            ],
        ),
    ):
        for name in names:
            p = folder / name
            files[str(p)] = file_hash(p)
    # 证据内容与审阅清单分别锁定；本次不能把当前副本登记成历史版本。
    protocol = {
        "version": "EVIDENCE_DRIVEN_PEER_ADMISSION_V1",
        "files": files,
        "current_fit_budget": 0,
        "proposed_cap_not_authorized": 12,
        "historical_local_download_required": False,
        "evidence_entries": [],
        "reviewed_evidence_digests": [],
        "acceptance_rule": "Reviewed source content plus dated version binding; date-only catalog is incomplete",
        "source_review_boundary": "Normalized evidence must be traced to raw source before pinning its digest",
        "scope": "Two fixed peers; input and label NAV dates 2021-2023; original 26 reports",
    }
    save_once(target, protocol)
    return protocol


def dependencies():
    """只读取此前已核的允许年份来源，列清每行真正的 61 日输入和两个答案端点。"""
    old_protocol = read_json(PREVIOUS / "protocol.json")
    snapshot = read_json(old_protocol["named"]["snapshot"])
    old_rows = read_json(PREVIOUS / "row-admission.json")
    old_inputs = read_json(PREVIOUS / "candidate-inputs-not-admitted.json")
    added = read_json(GAP / "additional-inputs-not-admitted.json")
    labels = read_json(GAP / "additional-labels-not-admitted.json")
    label_map = {(r["fund_code"], r["target"]): r for r in labels}
    inputs = {(r["fund_code"], r["target"]): r for r in old_inputs + added}
    if len(inputs) != 830 or len(old_rows) != 750 or len(added) != 80:
        raise ValueError("FROZEN_CANDIDATE_IDENTITIES_CHANGED")
    requirements, aliases, report_rows = {}, {}, []
    for report in read_json(GAP / "report-version-gap-worklist.json"):
        new = report.get("newly_recovered_report", False)
        if new:
            parsed = read_json(report["parsed_file"])
            path, sha = parsed["raw"]["path"], report["raw_sha256"]
            pub, original = report["source_publication_date"], sha
        else:
            path, sha = report["primary_pdf_path"], report["primary_pdf_sha256"]
            pub, original = report["public_date_unchanged"], report["original_raw_sha256_preserved"]
            if not report["primary_table_matches_previous"]:
                raise ValueError("REPORT_SEMANTIC_ALIAS_NOT_VERIFIED")
        if file_hash(path) != sha:
            raise ValueError("REPORT_RAW_CHANGED")
        key = f"REPORT:{report['fund_code']}:{report['report_end']}:{report['report_type']}"
        aliases[original] = key
        aliases[sha] = key
        requirements[key] = {
            "key": key,
            "type": "REPORT",
            "fund_code": report["fund_code"],
            "business_date": report["report_end"],
            "publication_date": pub,
            "value": sha,
        }
        report_rows.append(
            {
                **requirements[key],
                "original_priority": report["original_priority"],
                "primary_pdf": path,
                "old_content_alias": original,
            }
        )
    for nav in read_json(GAP / "nav-version-gap-worklist.json"):
        key = f"NAV:{nav['fund_code']}:{nav['nav_date']}"
        requirements[key] = {
            "key": key,
            "type": "NAV",
            "fund_code": nav["fund_code"],
            "business_date": nav["nav_date"],
            "publication_date": nav["ann_date_unchanged"],
            "value": nav["unit_nav"],
            "original_source_hash": nav["frozen_source_hash"],
        }
    rows = []
    days = sorted(snapshot["indices"]["000300.SH"]["rows"])
    nav = {c: {r["date"]: r for r in snapshot["funds"][c]["nav"]["rows"]} for c in ("017493", "160323")}
    for row in old_rows + added:
        code, u = row["fund_code"], row["target"]
        label = row if "mature_at" in row else label_map[code, u]
        window = row.get("nav_window") or nav_window(nav[code], days, u)
        vector = inputs[code, u]
        expected = vector.get("input_sha256", vector.get("input_vector_sha256"))
        if len(vector["x"]) != 20 or digest(vector["x"]) != expected:
            raise ValueError("FROZEN_INPUT_VECTOR_CHANGED")
        input_keys = [f"NAV:{code}:{d}" for d in window["nav_dates"]] + [aliases[row["report_sha256"]]]
        label_keys = [f"NAV:{code}:{label[d]}" for d in ("base", "target")]
        if len(window["nav_dates"]) != 61 or any(k not in requirements for k in input_keys + label_keys):
            raise ValueError("INCOMPLETE_DEPENDENCY_MAPPING")
        rows.append(
            {
                "fund_code": code,
                "target": u,
                "mature_at": label["mature_at"],
                "input_keys": input_keys,
                "label_keys": label_keys,
                "input_sha256": expected,
                "actual_direction": label["actual_direction"],
            }
        )
    return requirements, rows, report_rows


def catalog_publications():
    """重核上轮保存的公告目录与附件对应关系，明确它只解决哪一层证据。

    巨潮日期取公告记录时间，不能从附件 URL 猜日期。东方红沿用已保存的官网
    浏览器目录记录。目录关联能支持公开日期与附件身份，但不当作旧字节版本
    存档，也不会因此授予净值或报告训练资格。
    """
    manifest = read_json(GAP / "delivery-manifest.json")
    # 历史交付的清单结构在运行时显式核实；只读取已列入清单的公告响应。
    files = manifest["files"]
    native = {}
    for receipt_path in sorted(GAP.glob("*-receipt.json")):
        if str(receipt_path) not in files or file_hash(receipt_path) != files[str(receipt_path)]:
            raise ValueError("CATALOG_RECEIPT_CHANGED")
        receipt = read_json(receipt_path)
        if not receipt.get("form") or receipt.get("status") != "RECEIVED":
            continue
        if file_hash(receipt["path"]) != receipt["sha256"]:
            raise ValueError("CATALOG_RAW_CHANGED")
        raw = read_json(receipt["path"])
        for item in raw["announcements"]:
            if item.get("secCode") == "160323":
                native[item["announcementId"]] = (item, receipt)
    bindings = {}
    for name in ("hx-cninfo-report-links.json", "hx-cninfo-extra-report-links.json"):
        for selected in read_json(GAP / name):
            original, receipt = native[selected["announcementId"]]
            if selected != original:
                raise ValueError("SELECTED_CATALOG_RECORD_CHANGED")
            published = (
                datetime.fromtimestamp(original["announcementTime"] / 1000, timezone(timedelta(hours=8)))
                .date()
                .isoformat()
            )
            url = "https://static.cninfo.com.cn/" + original["adjunctUrl"]
            bindings[url] = {
                "publication_date": published,
                "announcement_id": original["announcementId"],
                "raw_catalog_sha256": receipt["sha256"],
                "basis": "CNINFO_NATIVE_ANNOUNCEMENT_RECORD",
            }
    for item in read_json(GAP / "df-issuer-browser-catalog.json")["reports"]:
        bindings[item["url"]] = {
            "publication_date": item["published_date"],
            "basis": "SAVED_ISSUER_BROWSER_CATALOG_OBSERVATION",
            "raw_catalog_sha256": file_hash(GAP / "df-issuer-browser-catalog.json"),
        }
    results = []
    for supplement in read_json(GAP / "report-supplements-v2.json"):
        binding = bindings[supplement["source_url"]]
        if binding["publication_date"] != supplement["source_publication_date"]:
            raise ValueError("REPORT_CATALOG_PUBLICATION_DIFFERS")
        results.append(
            {
                "fund_code": supplement["fund_code"],
                "report_end": supplement["report_end"],
                "report_type": supplement["report_type"],
                "pdf_sha256": supplement["raw_sha256"],
                "url": supplement["source_url"],
                **binding,
                "establishes": "Dated publication-to-attachment association, not archived content version",
            }
        )
    return results


def review():
    """逐依赖/阶段执行新判定，保存具体缺口及停止决定；不计算旧折权重。"""
    protocol = freeze()
    if protocol["current_fit_budget"] != 0:
        raise ValueError("PREPARATION_ONLY_ZERO_FIT_BUDGET")
    for path, sha in protocol["files"].items():
        if file_hash(path) != sha:
            raise ValueError("FROZEN_SOURCE_CHANGED:" + Path(path).name)
    index = load_evidence(protocol["evidence_entries"], protocol["reviewed_evidence_digests"])
    requirements, rows, reports = dependencies()
    publications = catalog_publications()
    prep = read_json(GAP / "independent-preparation.json")
    limits = {"2023Q2": "2023-04-01", "2023Q3": "2023-07-01", "2023Q4": "2023-10-01", "FULL": "2024-01-01"}
    requests = defaultdict(lambda: {"input_before": set(), "label_before": set(), "dependent_rows": set()})
    decisions, summary = [], {}
    for name, cutoff in limits.items():
        counts, admitted = Counter(), Counter()
        for row in rows:
            result = row_at(row, cutoff, requirements, index)
            if not result["time_eligible"]:
                continue
            counts[row["fund_code"]] += 1
            admitted[row["fund_code"]] += int(result["material_eligible"])
            decisions.append({"fold": name, **result})
            for failure in result["failed_dependencies"]:
                req = requests[failure["key"]]
                req["input_before" if failure["purpose"] == "INPUT" else "label_before"].add(failure["before"])
                req["dependent_rows"].add(row["fund_code"] + ":" + row["target"])
        expected_ids = {tuple(x) for x in prep["folds"][name]["candidate_peer_ids_pending_admission"]}
        actual_ids = {(r["fund_code"], r["target"]) for r in decisions if r["fold"] == name}
        if actual_ids != expected_ids:
            raise ValueError("ORIGINAL_FOLD_CANDIDATE_IDENTITIES_CHANGED")
        summary[name] = {"input_and_time_complete": dict(counts), "version_admitted": dict(admitted)}
    worklist = [
        {
            **requirements[key],
            **{k: sorted(v) for k, v in use.items()},
            "needed": "原文内容与历史版本日期绑定；不要求本系统当年下载",
        }
        for key, use in sorted(requests.items())
    ]
    unused = sorted(set(requirements) - set(requests))
    save_once(OUTPUT / "requirements.json", requirements)
    save_once(OUTPUT / "row-dependencies.json", rows)
    save_once(OUTPUT / "report-review.json", reports)
    save_once(OUTPUT / "publication-bindings.json", publications)
    save_once(OUTPUT / "row-version-admission.json", decisions)
    save_once(OUTPUT / "version-evidence-requests.json", worklist)
    # “部分来源通过”也不能绕过全包门槛或获得拟合许可。这里不发出训练包。
    decision = {
        "can_train": False,
        "status": "STOP_VERSION_EVIDENCE_MISSING_AND_ZERO_FIT_BUDGET"
        if worklist
        else "STOP_FULL_PACKAGE_CHECK_REQUIRED_AND_ZERO_FIT_BUDGET",
        "current_fit_budget": 0,
        "actual_new_fits": 0,
        "cumulative_actual_fits": 52,
        "candidate_rows": len(rows),
        "reports": len(reports),
        "priority_reports": sum(r["original_priority"] for r in reports),
        "supplement_publication_bindings_checked": len(publications),
        "distinct_nav_dependencies": sum(r["type"] == "NAV" for r in requirements.values()),
        "used_report_version_gaps": sum(r["type"] == "REPORT" for r in worklist),
        "unused_report_keys": unused,
        "folds": summary,
        "reviewed_version_evidence_count": len(protocol["evidence_entries"]),
        "training_package_emitted": False,
        "next": "取得带历史版本时间的原文后审阅、锁定新证据协议，再运行此判定和原全包门槛；预算另行授权",
    }
    save_once(OUTPUT / "decision.json", decision)
    if any(m.startswith("sklearn") or m.endswith("fund_002112_round3_model") for m in sys.modules):
        raise ValueError("MODEL_MODULE_IMPORTED")
    save_once(
        OUTPUT / "runtime-boundary.json",
        {"models_imported": False, "actual_new_fits": 0, "database_writes": 0, "network_calls_in_review": 0},
    )
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "review"))
    args = parser.parse_args()
    print(
        review() if args.command == "review" else {"protocol": str(OUTPUT / "protocol.json"), "frozen": bool(freeze())}
    )
