"""手动未来观察：从实际输入算概率，真实时钟留存，答案成熟后只追加。

默认不启动观察。历史未晋级时拒绝签发增强候选；迟报另记，不补造事前预测。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts import fund_002112_price_sources_v1 as price
from scripts import fund_002112_signal_common_v1 as c
from scripts import fund_002112_signal_features_v1 as features
from scripts import fund_002112_signal_sources_v1 as sources
from scripts import fund_002112_signal_train_v1 as train


def protocol(root):
    selected = c.io.read(root / "selection.json")
    path = root / "forward-protocol.json"
    if path.exists():
        return c.io.read(path)
    value = {
        "at": c.io.now(),
        "selected": selected["selected"],
        "observation_start": next(d for d in c.bundle()["sessions"] if d > c.io.now()[:10]),
        "eligible": selected["future_observation_eligible"],
        "start": "首次候选冻结之后的新交易日，手动签发；本次没有生成真实未来预测",
        "cutoff": "D 08:00 Asia/Shanghai",
        "target": "next session U; NAV(U) versus NAV(D)",
        "cadence_sessions": 20,
        "engineering_checks": [20, 60],
        "formal_evaluation_mature_answers": 120,
        "bootstrap": {"consecutive_session_block": 20, "replicates": 2000, "seed": 0, "confidence": 0.95},
        "late_missing_policy": "全体预定日期保留；迟报/缺报不算事前预测，不进入共同日期准确率，计入覆盖率分母",
        "answers": "答案版本追加保存，绝不覆盖原预测；更正追加并单列，禁止择优挑版本",
        "fits": "后续手动更新仍占本轮累计80硬预算，额度不足停止；不自行另开额度",
        "automatic_task": False,
        "adoption": False,
    }
    c.io.save(path, value)
    return value


def load_package(path):
    data = c.io.read(path)
    receipt = c.io.read(Path(path).with_suffix(".receipt.json"))
    if receipt["sha256"] != c.io.sha(path):
        raise ValueError("FORWARD_PACKAGE_CHANGED")
    c.verify_files(data["sources"])
    if c.moment(data["created_at"]) > c.moment(c.io.now()):
        raise ValueError("PACKAGE_FROM_FUTURE")
    return data


def snapshot(root):
    """只读日常手动同步的本机数据库和原件；新资料未通过原件校验时不会入场。

    快照记录实际读取时间。旧净值用已冻结版本；新值按本地实际见到的版本时刻
    及公告日期较晚者处理。本命令没有公开读取、大模型请求或业务写入。
    """
    from sqlalchemy import text

    from scripts import fund_002112_recent_completion_v1 as recent

    stamp = c.io.now()
    folder = root / "forward/snapshots" / stamp.replace(":", "-")
    folder.mkdir(parents=True, exist_ok=False)
    reader = c.bodies.Snapshot(folder)
    old_bundle = reader.read(c.bodies.BUNDLE)
    nav = old_bundle["nav"]
    engine = recent.engine()
    try:
        with engine.connect() as connection:
            records = [
                dict(r)
                for r in connection.execute(
                    text(
                        "select nav_date::text,unit_nav::text,ann_date::text,content_hash,"
                        "created_at::text,updated_at::text from nav_daily "
                        "where fund_code='002112' and nav_date >= '2023-09-01' order by nav_date"
                    )
                ).mappings()
            ]
    finally:
        engine.dispose()
    c.io.save(folder / "database-nav.json", {"read_at": stamp, "records": records})
    from decimal import Decimal

    grouped = {}
    for r in records:
        day = r["nav_date"]
        if day in grouped and Decimal(grouped[day]["unit_nav"]) != Decimal(r["unit_nav"]):
            raise ValueError("NAV_SOURCES_DISAGREE")
        grouped[day] = r
    for day, r in grouped.items():
        if day in nav and Decimal(nav[day]["unit_nav"]) == Decimal(r["unit_nav"]):
            continue
        if not r["ann_date"]:
            continue
        available = max(
            c.moment(c.nav_inputs.available_at(r["ann_date"])), c.moment(r["created_at"]), c.moment(r["updated_at"])
        )
        nav[day] = {**r, "available_at": available.isoformat(), "first_seen_basis": "LOCAL_OBSERVED_VERSION"}
    reports = reader.read(c.OLD / "reports.json")
    hashes = {r["raw"]["sha256"] for r in reports}
    for directory in (c.io.RESEARCH / "reports", c.io.RESEARCH / "materials-live/reports"):
        for p in directory.glob("*.json"):
            r = c.io.payload(c.io.read(p))
            if (
                r.get("fund_code") != "002112"
                or r.get("fund_master_code") != "001412"
                or r.get("quality") != "VERIFIED_TABLE_TOTALS"
                or r.get("raw", {}).get("sha256") in hashes
            ):
                continue
            raw = r["raw"]
            raw_path = Path(raw["raw_path"]) if raw.get("raw_path") else c.bodies.local_source(raw["file"])
            if not raw_path or c.io.sha(raw_path) != raw["sha256"]:
                continue
            r = reader.read(p)
            r["available_at"] = c.nav_inputs.available_at(r["published_date"], r["available_at"], r.get("revised_at"))
            reports.append(r)
            hashes.add(raw["sha256"])
    docs = sources.existing_docs()
    docs.update({d["id"]: d for d in c.lines(root / "supplement-documents.jsonl")})
    index = reader.read(c.bodies.INDEX)
    for entry in index["documents"]:
        if entry["kind"] != "company" or entry.get("publishedDate", "") < "2026-09-29":
            continue
        d = c.bodies.resolve_document(entry, reader)
        if d.get("body_available_at") and not d.get("body_exclusions"):
            if d["id"] in docs and d.get("body_sha256") == docs[d["id"]].get("body_sha256"):
                continue
            docs[d["id"]] = d
    cached = {e["id"]: e for e in c.lines(c.OLD / "feature-revision-r3/events.jsonl")}
    events = features.build_events(list(docs.values()), cached, reports)
    from app.services.direction_1d_protocol import calendar

    sessions, _ = calendar()
    files = {v["snapshot"]: v["sha256"] for v in reader.files.values()}
    files[str(folder / "database-nav.json")] = c.io.sha(folder / "database-nav.json")
    package = {
        "created_at": c.io.now(),
        "sessions": [str(d) for d in sessions],
        "nav": nav,
        "reports": reports,
        "events": events,
        "sources": {"files": files},
        "external_requests": 0,
    }
    path = folder / "package.json"
    c.io.save(path, package)
    c.io.save(path.with_suffix(".receipt.json"), {"at": c.io.now(), "sha256": c.io.sha(path)})
    return {"package": str(path), "nav_rows": len(nav), "reports": len(reports), "events": len(events)}


def make_row(package, base):
    sessions, nav = package["sessions"], package["nav"]
    index = sessions.index(base)
    target = sessions[index + 1]
    end, _, n = c.nav_inputs.choose_nav_window(sessions, nav, base)
    if end is None:
        raise ValueError("NO_COMPLETE_AVAILABLE_NAV_WINDOW")
    idx = sessions.index(end)
    nav_days = sessions[idx - 60 : idx + 1]
    row = {
        "base": base,
        "target": target,
        "as_of": c.nav_inputs.at0800(base).isoformat(),
        "session_index": index,
        "groups": {"N": n, "NE": price.nav_extra([float(nav[d]["unit_nav"]) for d in nav_days])},
        "label_mature_at": None,
    }
    row["E"], plain = features.vector(row, package["events"], package["reports"], sessions)
    row["ED"], decay = features.vector(row, package["events"], package["reports"], sessions, True)
    row["trigger"], row["trigger_decay"] = plain["trigger"], decay["trigger"]
    return row, {"B2": plain, "B3": decay, "nav_days": nav_days}


def timing_status(base, sessions, generated_at):
    now = c.moment(generated_at)
    if base not in sessions:
        return "NOT_A_SESSION"
    if now.date().isoformat() != base:
        return "HISTORICAL_BACKFILL" if now.date().isoformat() > base else "TOO_EARLY"
    return "TIMELY" if now <= c.nav_inputs.at0800(base) else "LATE"


def append_unique(path, row, keys):
    """同一目标日的同类记录独占写入，失败和迟报另有类型，不覆盖已签发概率。"""
    if any(all(old.get(k) == row.get(k) for k in keys) for old in c.lines(path)):
        raise ValueError("APPEND_ONLY_DUPLICATE")
    c.io.append(path, row)


def issue(root, package_path, base, allow_fit=False):
    root = Path(root).resolve()
    train.verify_freeze(root)
    policy = protocol(root)
    if not policy["eligible"]:
        raise ValueError("HISTORICAL_CANDIDATES_NOT_PROMOTED; 工具就绪，但本轮不自动开始增强候选观察")
    package = load_package(package_path)
    generated = c.io.now()
    status = timing_status(base, package["sessions"], generated)
    if status != "TIMELY":
        append_unique(
            root / "forward/issues.jsonl", {"at": generated, "base": base, "status": status}, ("base", "status")
        )
        return {"status": status, "saved_as_prospective": False}
    row, lineage = make_row(package, base)
    # 提前签发时，不得把稍后才可用的资料当作已见。未来答案也不能已存在于包中。
    assert row["target"] not in package["nav"], "ANSWER_ALREADY_IN_INPUT_PACKAGE"
    assert all(c.moment(package["nav"][d]["available_at"]) <= c.moment(generated) for d in lineage["nav_days"])
    for proof in (lineage["B2"], lineage["B3"]):
        assert all(c.moment(e["version_available_at"]) <= c.moment(generated) for e in proof["events"])
    chosen = policy["selected"]
    groups = ["B0", "B3_MODEL" if chosen == "B3" else "B1"]
    ledger = c.lines(root / "fit-ledger.jsonl")
    model_records = {
        g: max((r for r in ledger if r["group"] == g and not r["replay"]), key=lambda r: r["update"]["cutoff"])
        for g in groups
    }
    last_base = min(r["update"]["cutoff"][:10] for r in model_records.values())
    if package["sessions"].index(base) - package["sessions"].index(last_base) >= 20:
        if not allow_fit:
            raise ValueError("FIXED_20_SESSION_MODEL_REFRESH_REQUIRED; 请在同一命令显式加 --allow-fit，仍占累计80预算")
        history = c.lines(train.feature_root(root) / "inputs.jsonl")
        for record in c.lines(root / "forward/predictions.jsonl"):
            x = record["input"]
            if all(d in package["nav"] for d in (x["base"], x["target"])):
                x["label_mature_at"] = max(package["nav"][d]["available_at"] for d in (x["base"], x["target"]))
                history.append(x)
        history.append(row)
        indices = [
            i
            for i, r in enumerate(history[:-1])
            if r["label_mature_at"]
            and c.moment(r["label_mature_at"]) < c.moment(row["as_of"])
            and r["target"] < row["target"]
        ]
        update = {"id": "FORWARD_" + base, "training": indices, "evaluate": [len(history) - 1], "cutoff": row["as_of"]}
        c.io.save(root / "forward/updates" / (base + "-rows.json"), history)
        for g in groups:
            train.fit_one(root, g, update, history, package["nav"])
        ledger = c.lines(root / "fit-ledger.jsonl")
        model_records = {g: next(r for r in ledger if r["id"] == g + "__" + update["id"]) for g in groups}
    probabilities, model_hashes = {}, {}
    for group, record in model_records.items():
        path = root / "runs" / record["id"] / "model.joblib"
        done = c.io.read(path.parent / "complete.json")
        assert c.io.sha(path) == done["model_sha256"]
        probabilities[group] = train.aligned_probabilities(joblib.load(path), [row])[0].tolist()
        model_hashes[group] = done["model_sha256"]
    candidate = (
        probabilities["B1"]
        if chosen == "B1"
        else train.blend(
            probabilities["B0"], probabilities[groups[1]], row["trigger_decay" if chosen == "B3" else "trigger"]
        )
    )
    finished = c.io.now()
    if timing_status(base, package["sessions"], finished) != "TIMELY":
        append_unique(
            root / "forward/issues.jsonl",
            {"at": finished, "base": base, "status": "LATE_DURING_COMPUTE"},
            ("base", "status"),
        )
        return {"status": "LATE_DURING_COMPUTE", "saved_as_prospective": False}
    record = {
        "generated_at": finished,
        "base": base,
        "target": row["target"],
        "status": "TIMELY",
        "selected": chosen,
        "baseline": probabilities["B0"],
        "candidate": candidate,
        "input": row,
        "input_sha256": c.io.digest(row),
        "package_sha256": c.io.sha(package_path),
        "model_hashes": model_hashes,
        "lineage": lineage,
        "protocol_sha256": c.io.sha(root / "forward-protocol.json"),
    }
    append_unique(root / "forward/predictions.jsonl", record, ("target",))
    return {"status": "PREDICTION_SAVED_BEFORE_ANSWER", "target": row["target"]}


def mature_answer(prediction, nav, now):
    days = (prediction["base"], prediction["target"])
    if not all(d in nav for d in days):
        return None
    mature = max(c.moment(nav[d]["available_at"]) for d in days)
    if mature >= c.moment(now):
        return None
    if c.moment(prediction["generated_at"]) >= mature:
        raise ValueError("PREDICTION_WAS_NOT_BEFORE_ANSWER")
    label = c.nav_inputs.classify_label(nav[days[0]]["unit_nav"], nav[days[1]]["unit_nav"])
    return {
        "target": days[1],
        "base": days[0],
        "label": label,
        "mature_at": mature.isoformat(),
        "source_values": {d: nav[d] for d in days},
        "version": c.io.digest([nav[d] for d in days]),
    }


def settle(root, package_path):
    package = load_package(package_path)
    prior = c.lines(root / "forward/answers.jsonl")
    written = 0
    for p in c.lines(root / "forward/predictions.jsonl"):
        answer = mature_answer(p, package["nav"], c.io.now())
        if answer and not any(a["target"] == answer["target"] and a["version"] == answer["version"] for a in prior):
            answer.update(
                recorded_at=c.io.now(),
                revision=any(a["target"] == answer["target"] for a in prior),
                prediction_sha256=c.io.digest(p),
            )
            append_unique(root / "forward/answers.jsonl", answer, ("target", "version"))
            written += 1
    return {"answers_appended": written}


def bootstrap_difference(differences):
    """固定连续20日分块抽样；只度量共同日期，相关性和缺报限制单独报告。"""
    values = np.asarray(differences, dtype=float)
    if len(values) < 120:
        raise ValueError("WAIT_FOR_120_COMMON_MATURE_DATES")
    rng = np.random.default_rng(0)
    estimates = []
    for _ in range(2000):
        starts = rng.integers(0, len(values) - 20 + 1, size=int(np.ceil(len(values) / 20)))
        sample = np.concatenate([values[s : s + 20] for s in starts])[: len(values)]
        estimates.append(float(sample.mean()))
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def evaluate(root, sessions, nav=None):
    predictions = c.lines(root / "forward/predictions.jsonl")
    policy = protocol(root)
    first, today = policy["observation_start"], c.io.now()[:10]
    expected = [d for d in sessions if first <= d <= today and c.nav_inputs.at0800(d) <= c.moment(c.io.now())]
    if not predictions:
        return {
            "state": "WAITING_FOR_REAL_FUTURE_SAMPLES",
            "prospective_predictions": 0,
            "expected_dates": expected,
            "missing_dates": expected,
            "historical_promotion": policy["eligible"],
        }
    by_base = {p["base"]: p for p in predictions}
    answers = {}
    for a in c.lines(root / "forward/answers.jsonl"):
        # 第一成熟版本是主评价；后续修订另行报告，不从多个答案里挑成绩。
        answers.setdefault(a["target"], a)
    ready = [p for p in predictions if p["target"] in answers]
    summary = {
        "at": c.io.now(),
        "expected_dates": expected,
        "missing_dates": [d for d in expected if d not in by_base],
        "coverage": len(predictions) / len(expected) if expected else 0,
        "mature_common_dates": len(ready),
        "issues": c.lines(root / "forward/issues.jsonl"),
        "answer_revisions": sum(a.get("revision", False) for a in c.lines(root / "forward/answers.jsonl")),
        "state": "WAITING_FOR_120_REAL_MATURE_SAMPLES",
        "formal_effect_evaluation": None,
    }
    scheduled_mature = []
    if nav is not None:
        for base in expected:
            target = sessions[sessions.index(base) + 1]
            if all(day in nav and c.moment(nav[day]["available_at"]) < c.moment(c.io.now()) for day in (base, target)):
                scheduled_mature.append(base)
    summary["scheduled_mature_dates"] = scheduled_mature
    # 120个预定日期答案成熟就评价；不能因漏报而向后挑到120个漂亮的共同日期。
    if len(scheduled_mature) >= 120 and not (root / "forward/formal-evaluation.json").exists():
        scheduled_mature = scheduled_mature[:120]
        ready = sorted([p for p in ready if p["base"] in scheduled_mature], key=lambda p: p["target"])
        truth = {p["target"]: answers[p["target"]]["label"] for p in ready}
        result = {}
        for name in ("baseline", "candidate"):
            ps = [
                {"target": p["target"], "probabilities": p[name], "predicted": c.CLASSES[int(np.argmax(p[name]))]}
                for p in ready
            ]
            result[name] = train.metrics(ps, truth) if ps else None
        differences = [
            int(c.CLASSES[int(np.argmax(p["candidate"]))] == truth[p["target"]])
            - int(c.CLASSES[int(np.argmax(p["baseline"]))] == truth[p["target"]])
            for p in ready
        ]
        # 缺报会破坏连续交易日区块，保留成绩但不伪造所要求的区块区间。
        indices = [sessions.index(p["base"]) for p in ready]
        contiguous = len(indices) == 120 and indices == list(range(indices[0], indices[0] + len(indices)))
        result["interval"] = bootstrap_difference(differences) if contiguous else None
        result["interval_limitation"] = (
            "共同日期不连续，不能把缺报挤掉冒充连续20日区块" if not contiguous else "有限样本及时序相关性仍限制外推"
        )
        result["dates"] = [p["target"] for p in ready]
        result["scheduled_dates"] = scheduled_mature
        result["coverage"] = len(ready) / 120
        c.io.save(root / "forward/formal-evaluation.json", result)
        summary.update(state="FORMAL_EVALUATION_RECORDED", formal_effect_evaluation=result)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("protocol", "snapshot", "issue", "settle", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--allow-fit", action="store_true")
    args = parser.parse_args()
    with c.io.writer_lock(args.root):
        if args.command == "protocol":
            result = protocol(args.root)
        elif args.command == "snapshot":
            result = snapshot(args.root)
        elif args.command == "issue":
            result = issue(args.root, args.package, args.base, args.allow_fit)
        elif args.command == "settle":
            result = settle(args.root, args.package)
        else:
            package = load_package(args.package) if args.package else None
            result = evaluate(
                args.root,
                package["sessions"] if package else c.bundle()["sessions"],
                package["nav"] if package else None,
            )
        print(c.io.canonical(result))


if __name__ == "__main__":
    main()
