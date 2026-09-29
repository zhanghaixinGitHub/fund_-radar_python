"""H2 首次拟合前的资料版本：使用完整覆盖的经理策略文字，保留整组事件的失败证据。"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from app.services.fund_002112_holdings_compatibility import json_subtree
from app.services.fund_002112_information_admission import OUT as EARLY
from app.services.fund_002112_information_admission import PREVIOUS, RUN, report_pages, save_once
from app.services.fund_002112_style_coverage import strategy_section
from app.services.fund_002112_zero_fit_review import ROOT, digest, file_hash, read_json

OUT = RUN / "strategy-admission-v1"
STYLE = ROOT / "style-training-coverage/20260928-v1"
TAGS = ("MEDICINE_FOCUS", "AI_COMPUTE_FOCUS", "OTHER_EXPLICIT_FOCUS", "DIVERSIFIED_OR_MULTI", "UNSPECIFIED")
# 2024 年只作为已观察的开发资料；先核原文再冻结，不查看各日预测对错来选标签。
DEVELOPMENT_REVIEW = {
    "a34025093dcea139ca3e1116c4349c282705a7956ed7ff11f5ca76d411797399": ("MEDICINE_FOCUS", "我们医药配置方向较为均衡"),
    "00210b443023f058318871a8aec4d40bdf89182aeb88998a6282663a1b0a31f4": (
        "AI_COMPUTE_FOCUS",
        "接下来本产品将主要聚焦AI算力赛道",
    ),
    "e3c7090238111436412b609481a3ab9f7dff31629a2c34499f1a79a23aaeafe6": (
        "AI_COMPUTE_FOCUS",
        "我们继续看好人工智能算力板块的整体表现",
    ),
    "365e8bb50caab686ca98c244e50666bdd029607e9c7bd26b6a546d519aa2ecf6": (
        "AI_COMPUTE_FOCUS",
        "本产品将持续深耕人工智能算力细分赛道",
    ),
    "88c9b5e3ed4887081058dcc8b024dbb3aa9dcb5cb537fc0a00442630150a079f": (
        "AI_COMPUTE_FOCUS",
        "本产品将持续深耕人工智能算力细分赛道",
    ),
    "1de8ac0a73e14f37e7b68952baa0668d23370e18ce45a89cd9ce7088fcf6c7d7": (
        "AI_COMPUTE_FOCUS",
        "本产品将持续深耕人工智能算力细分赛道",
    ),
}


def append_strategy(row, record):
    """只增五列报告文字类别；原 20 列、原答案和日期字段保留，未知文字有独立类别。"""
    if record["tag"] not in TAGS or record["published_date"] >= row["target"]:
        raise ValueError("STRATEGY_TAG_OR_PUBLICATION_INVALID")
    if record["report_sha256"] != row["report_sha256"] or len(row["x"]) != 20:
        raise ValueError("STRATEGY_REPORT_OR_INPUT_IDENTITY")
    return {
        **row,
        "x": row["x"] + [int(record["tag"] == t) for t in TAGS],
        "strategy_tag": record["tag"],
        "base_input_sha256": digest(row),
    }


def prepare():
    """基于覆盖差异收敛 H2；没有第三个假设、不变更一日目标或原验收条件。"""
    from app.services.fund_002112_round3_data import FEATURES

    OUT.mkdir(parents=True, exist_ok=True)
    # 先登记资料原因与最终特征，再生成新输入，任何 H2 拟合必须发生在此后。
    amendment = {
        "hypothesis": "H2_EVENT_INFORMATION",
        "version": "REPORT_STRATEGY_ONLY_V1",
        "reason": (
            "Only 705/5576 company windows complete and policy/news catalogs incomplete; "
            "report strategy covers all 5576"
        ),
        "question": "Does already-public fund-manager strategy text improve H1 and original L20?",
        "features": list(FEATURES) + ["report_strategy_" + t.lower() for t in TAGS],
        "encoding": "Five explicit one-hot narrative categories; unknown is UNSPECIFIED, not zero news",
        "scope_limit": "Tests report narrative only; does not test company events, policy or news utility",
        "semantic_limit": (
            "Narrative expressed investment focus/plans, not verified daily holdings or completed trading"
        ),
        "training_tags": "Reuse all 140 previously reviewed and frozen tags unchanged",
        "development_tags": (
            "Six previously public report sections reviewed before H2 fits; report intent is retained as narrative"
        ),
        "maximum_fits": 12,
        "global_maximum_fits": 24,
        "algorithm_parameters_weights_gates_changed": False,
        "earlier_event_draft_preserved": str(RUN / "event-admission/decision.json"),
    }
    save_once(OUT / "feature-contract.json", amendment)
    source_files = dict(read_json(EARLY / "sources.json")["files"])
    paths = [
        EARLY / "inputs.json",
        EARLY / "folds.json",
        STYLE / "sections-reviewed.json",
        STYLE / "reviewed-tags.json",
        STYLE / "protocol.json",
        RUN / "strategy-development-sections.json",
        RUN / "event-admission/coverage-summary.json",
        RUN / "event-admission/decision.json",
    ]
    for p in paths:
        source_files[str(p)] = file_hash(p)
    sections = {r["source"]["report_sha256"]: r for r in read_json(STYLE / "sections-reviewed.json")}
    reviewed = {}
    for tag in read_json(STYLE / "reviewed-tags.json"):
        sha = tag["report_sha256"]
        source, section = sections[sha]["source"], sections[sha]["section"]
        if (
            section["status"] != "EXTRACTED"
            or section["text"][tag["section_quote_offset"] :][: len(tag["evidence"])] != tag["evidence"]
        ):
            raise ValueError("PRIOR_STRATEGY_ANCHOR_CHANGED")
        if file_hash(source["raw_path"]) != sha:
            raise ValueError("STRATEGY_RAW_CHANGED")
        source_files[source["raw_path"]] = sha
        reviewed[sha] = {
            **tag,
            "published_date": max(source["effective_publication_dates"]),
            "section_sha256": digest(section["text"]),
            "review_origin": "PREVIOUS_FROZEN_140_UNCHANGED",
        }
    snapshot_path = Path(read_json(PREVIOUS / "protocol.json")["sources"]["snapshot"]["path"])
    reports = {
        r["raw"]["sha256"]: r
        for r in json_subtree(snapshot_path.read_text(encoding="utf-8"), ["payload", "funds", "002112", "reports"])
    }
    extracted = {r["report_sha256"]: r for r in read_json(RUN / "strategy-development-sections.json")}
    for sha, (tag, quote) in DEVELOPMENT_REVIEW.items():
        report = reports[sha]
        pages, public, modified = report_pages(report)
        section = strategy_section([{"page": i + 1, "text": t} for i, t in enumerate(pages)])
        if section != extracted[sha]["section"] or quote not in section["text"]:
            raise ValueError("DEVELOPMENT_STRATEGY_REPLAY_OR_VERSION_FAILED")
        if modified and modified > public:
            # 不忽略较晚 PDF 修改日；为新增文字单独核同期目录绑定的公开全文。
            # 只证明来源声明日的对应业务内容一致，不宣称 PDF 字节在历史上从未变化。
            from app.integrations.fund_report_sections_v2 import parse_text
            from app.services.fund_002112_peer_admission import BUSINESS_FIELDS

            folder = RUN / "strategy-source-version-check"
            entry = read_json(folder / "selected-catalog.json")[0]
            body = read_json(folder / "annual-body.json")["data"]
            if (
                sha != "00210b443023f058318871a8aec4d40bdf89182aeb88998a6282663a1b0a31f4"
                or body["art_code"] != entry["ID"]
                or "002112" not in {s["stock"] for s in body["security"]}
                or body["notice_date"][:10] != public
                or entry["PUBLISHDATEDesc"] != public
            ):
                raise ValueError("LATE_PDF_ALTERNATE_SOURCE_IDENTITY_OR_DATE")
            reprint = parse_text([body["notice_content"]], body["notice_title"], derive_missing_weights=True)
            if any(reprint[k] != report[k] for k in BUSINESS_FIELDS):
                raise ValueError("LATE_PDF_ALTERNATE_SOURCE_TABLES_DIFFER")
            alternative = strategy_section([{"page": None, "text": body["notice_content"]}])
            if alternative["text"] != section["text"]:
                raise ValueError("LATE_PDF_ALTERNATE_STRATEGY_DIFFERS")
            for p in folder.iterdir():
                if p.is_file():
                    source_files[str(p)] = file_hash(p)
            save_once(
                OUT / "source-version-resolution.json",
                {
                    "primary_pdf_sha256": sha,
                    "primary_pdf_modified": modified,
                    "primary_pdf_historical_version_proven": False,
                    "alternate_body": str(folder / "annual-body.json"),
                    "alternate_body_sha256": file_hash(folder / "annual-body.json"),
                    "alternate_notice_id": entry["ID"],
                    "alternate_publication": public,
                    "tables_equal": True,
                    "strategy_text_equal": True,
                    "basis": (
                        "Dated public reprint bound to fund and report identity; original declared-publication rule"
                    ),
                    "independent_historical_archive_required": False,
                },
            )
        raw_path = ROOT / report["raw"]["file"]
        source_files[str(raw_path)] = sha
        reviewed[sha] = {
            "report_sha256": sha,
            "fund_code": "002112",
            "report_end": report["report_end"],
            "tag": tag,
            "evidence": quote,
            "published_date": public,
            "source_pages": section["pages"],
            "section_quote_offset": section["text"].index(quote),
            "section_sha256": digest(section["text"]),
            "review_origin": "PRE_H2_FIT_SOURCE_ONLY_REVIEW",
            "meaning": "Expressed strategy including intent, not proof of execution",
        }
    base = read_json(EARLY / "inputs.json")
    inputs = {part: [append_strategy(r, reviewed[r["report_sha256"]]) for r in rows] for part, rows in base.items()}
    index = {(r["fund_code"], r["target"]): r for r in inputs["train"] + inputs["development"]}
    folds = []
    for f in read_json(EARLY / "folds.json"):
        train = [index[tuple(k)] for k in f["train_ids"]]
        exam = [index[r["fund_code"], r["target"]] for r in f["exam"]]
        folds.append({**f, "train_sha256": digest(train), "exam": exam, "exam_sha256": digest(exam)})
    save_once(OUT / "reviewed-strategy.json", list(reviewed.values()))
    save_once(OUT / "inputs.json", inputs)
    save_once(OUT / "folds.json", folds)
    save_once(OUT / "sources.json", {"files": source_files})
    summary = {
        "material_ready": True,
        "scope": "REPORT_STRATEGY_ONLY_V1",
        "pool_rows": len(inputs["train"]),
        "development_rows": len(inputs["development"]),
        "reports": len(reviewed),
        "features": 25,
        "new_fits": 0,
        "tags": dict(Counter(r["strategy_tag"] for r in inputs["train"])),
        "original_twenty_inputs_unchanged": True,
    }
    save_once(OUT / "decision.json", summary)
    audit()
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def audit():
    """独立反向解码新增五列，剥离新增字段后与 H1 每一行及每折权重逐字比较。"""
    original = read_json(EARLY / "inputs.json")
    inputs = read_json(OUT / "inputs.json")
    tags = {r["report_sha256"]: r for r in read_json(OUT / "reviewed-strategy.json")}
    columns = {
        "MEDICINE_FOCUS": 20,
        "AI_COMPUTE_FOCUS": 21,
        "OTHER_EXPLICIT_FOCUS": 22,
        "DIVERSIFIED_OR_MULTI": 23,
        "UNSPECIFIED": 24,
    }
    count = 0
    for part in original:
        for old, new in zip(original[part], inputs[part], strict=True):
            restored = {k: v for k, v in new.items() if k not in ("strategy_tag", "base_input_sha256")}
            restored["x"] = new["x"][:20]
            positions = [i for i, value in enumerate(new["x"]) if i >= 20 and value == 1]
            tag = tags[new["report_sha256"]]
            if restored != old or new["base_input_sha256"] != digest(old) or positions != [columns[tag["tag"]]]:
                raise ValueError("INDEPENDENT_STRATEGY_OR_BASE_ROW_CHANGED")
            if sum(new["x"][20:]) != 1 or len(new["x"]) != 25 or tag["published_date"] >= new["target"]:
                raise ValueError("STRATEGY_INPUT_OR_TIME_INVALID")
            count += 1
    for a, b in zip(read_json(EARLY / "folds.json"), read_json(OUT / "folds.json"), strict=True):
        for name in ("name", "start", "train_ids", "weights", "gate", "original_exam_identity_sha256"):
            if a[name] != b[name]:
                raise ValueError("FOLD_MEMBERSHIP_OR_WEIGHTS_CHANGED")
        if [(r["target"], r["actual_direction"]) for r in a["exam"]] != [
            (r["target"], r["actual_direction"]) for r in b["exam"]
        ]:
            raise ValueError("EXAM_LABEL_OR_DATE_CHANGED")
    for path, sha in read_json(OUT / "sources.json")["files"].items():
        if file_hash(path) != sha:
            raise ValueError("STRATEGY_SOURCE_CHANGED")
    result = {
        "passed": True,
        "rows_independently_checked": count,
        "old_twenty_values_and_labels_preserved": True,
        "training_membership_weights_and_exam_dates_preserved": True,
        "new_fits": 0,
        "code_sha256": file_hash(__file__),
    }
    save_once(OUT / "independent-audit-v2.json", result)
    return result


def freeze():
    """H2 使用同一个已测试执行器；只新增数据版本冻结清单，不修改 H1 代码或清单。

    初版整组事件未进入拟合；其资料失败决定保留。新的 H2 清单明确指向策略资料包，
    标准 train/verify 入口随后完全沿用同一全局账本和一次性许可机制。
    """
    from app.services import fund_002112_information_experiment as engine

    with engine.lock():
        path = engine.candidate_path(engine.HYPOTHESES[1])
        if (path / "freeze.json").exists():
            engine.load(engine.HYPOTHESES[1])
            return
        if any(e["hypothesis"] == engine.HYPOTHESES[1] for e in engine.ledger()):
            raise ValueError("CANNOT_REDEFINE_CONSUMED_H2")
        audit()
        spec, _ = engine.load(engine.HYPOTHESES[0])
        spec = copy.deepcopy(spec)
        spec.update(
            hypothesis=engine.HYPOTHESES[1],
            admission=str(OUT),
            features=read_json(OUT / "feature-contract.json")["features"],
            material_version="REPORT_STRATEGY_ONLY_V1",
        )
        spec["sources"].update(read_json(OUT / "sources.json")["files"])
        spec["sources"].update({str(p): file_hash(p) for p in OUT.iterdir() if p.is_file()})
        for p in (Path(__file__), Path(__file__).parents[1] / "tests/test_fund_002112_strategy_candidate.py"):
            spec["code"][str(p.resolve())] = file_hash(p)
        engine.write(path / "freeze.json", spec)
        engine.write(path / "freeze-anchor.json", {"sha256": file_hash(path / "freeze.json")})
        engine.load(engine.HYPOTHESES[1])
        print(json.dumps({"frozen_h2": True, "features": len(spec["features"]), **engine.status()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "freeze"))
    args = parser.parse_args()
    {"prepare": prepare, "audit": audit, "freeze": freeze}[args.command]()
