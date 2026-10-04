"""持仓事件研究的时点关联与固定对照实验；无外部网络、数据库或业务发布动作。"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import fund_002112_existing_events_features_v1 as legacy_features
from scripts import fund_002112_existing_events_fit_v1 as stats
from scripts import fund_002112_holding_impact_v1 as e

io, ROOT, OLD = e.io, e.ROOT, e.OLD
CONTEXT = ["report_missing", "report_age_days", "report_public_age_days", "disclosed_nav_weight", "full_disclosure"]
# 分类输入是经营语义及阶段；模型仍需从历史结果学习，不能预设利好必涨。
SEMANTICS = [f"channel_{v}" for v in e.CHANNELS] + [f"direction_{v}" for v in e.DIRECTIONS]
SEMANTICS += [f"stage_{v}" for v in e.STAGES] + [f"kind_{v}" for v in e.KINDS]
SEMANTICS += ["profit_yoy_known", "profit_yoy_increase", "profit_yoy_decrease"]
WINDOWS = (1, 5, 20)
DIAGNOSTICS = ["body_missing", "extraction_quarantined", "unmatched", "weight_rounded_zero", "grounded"]
GLOBAL_COLUMNS = [f"global_{t}_{w}_{c}" for t in e.TYPES for w in WINDOWS for c in SEMANTICS]
WEIGHT_COLUMNS = [f"holding_{t}_{w}_{c}" for t in e.TYPES for w in WINDOWS for c in SEMANTICS]
QUALITY_COLUMNS = [f"quality_{t}_{q}" for t in e.TYPES for q in DIAGNOSTICS]


def known_at(doc: dict) -> str:
    """精确时间按实；日期级资料从下一自然日08:00起算，修订取更晚约束。"""
    points = []
    if doc.get("published_at"):
        points.append(datetime.fromisoformat(doc["published_at"]).astimezone(io.ZONE))
    else:
        next_day = date.fromisoformat(doc["published_date"]) + timedelta(days=1)
        points.append(datetime.fromisoformat(str(next_day) + "T08:00:00+08:00"))
    if doc.get("revision_at"):
        point = datetime.fromisoformat(doc["revision_at"])
        points.append(point.replace(tzinfo=io.ZONE) if point.tzinfo is None else point.astimezone(io.ZONE))
    return max(points).isoformat()


def links(doc: dict, result: dict, report: dict | None, profiles: dict, cutoff: str, timing: str) -> list[dict]:
    """公司直接关联用已验收证券代码；政策间接关联必须有双方逐字业务证据。

    政策与产品词重合仅建立“相关业务”关系，不足以证明同一公司受益或承压。
    所以间接关联不继承政策的笼统方向，后续权重向量把方向置为UNKNOWN。
    """
    if report is None:
        return []
    output = []
    for holding in report["holdings"]:
        code = holding["stock_code"][:6]
        proof, relation = None, None
        if code in doc["issuer_codes"]:
            relation = "ISSUER_CODE"
            proof = {"issuer_codes": doc["issuer_codes"], "source_refs": doc["text_evidence"]["source_refs"]}
        elif result["status"] == "SOURCE_GROUNDED":
            for target in result["targets"]:
                if target["term"] in {"公司", "企业", "行业", "产品", "服务", "业务", "工业", "经济", "金融"}:
                    continue
                for profile in profiles.get((code, target["term"]), []):
                    point = profile.get("effective_" + timing)
                    if point and point <= cutoff[:10] and profile.get("known_at", "") <= cutoff:
                        relation = "EXACT_PRODUCT_WITH_TWO_SOURCE_QUOTES"
                        proof = {"target": target, "company_product": profile}
                        break
                if relation:
                    break
        if relation:
            output.append(
                {
                    "stock_code": holding["stock_code"],
                    "stock_name": holding["stock_name"],
                    "nav_weight_pct": float(holding["nav_weight_pct"]),
                    "relation": relation,
                    "proof": proof,
                    "report_end": report["report_end"],
                    "report_available_at": report["available_at"],
                    "report_sha256": report["raw"]["sha256"],
                    "weight_is_rounded": float(holding["nav_weight_pct"]) == 0,
                    "company_direction": result.get("direction", "UNKNOWN") if relation == "ISSUER_CODE" else "UNKNOWN",
                    "applicability_confirmed": relation == "ISSUER_CODE",
                }
            )
    return output


def semantic_vector(result: dict, indirect: bool = False) -> np.ndarray:
    values = {
        "channel": result["channel"],
        "direction": result["direction"],
        "stage": result["stage"],
        "kind": result["kind"],
    }
    if indirect:
        values["direction"] = "UNKNOWN"
        values["channel"] = "NONE"
    vector = np.asarray([float(c in {f"{k}_{v}" for k, v in values.items()}) for c in SEMANTICS], dtype=float)
    # 只认同一自然语言引句中明确的净利润同比变化，不能从无列头的表格猜单位/期间。
    # 该量是经营事实输入，乘持仓权重后仍不是基金收益预测。
    numeric = profit_yoy(result)
    if numeric is not None and not indirect:
        vector[SEMANTICS.index("profit_yoy_known")] = 1
        name = "profit_yoy_increase" if numeric >= 0 else "profit_yoy_decrease"
        vector[SEMANTICS.index(name)] = math.log1p(abs(numeric) / 100)
    return vector


def profit_yoy(result: dict) -> float | None:
    """存在明确期间且引句只有一个不矛盾的净利润同比数值时返回百分数，否则未知。"""
    if result.get("kind") != "EARNINGS" or not re.search(r"20\d{2}", result.get("period_quote", "")):
        return None
    found = set()
    for quote in result.get("facts", []):
        for clause in re.split(r"[。；]", e.quote_text(quote)):
            if "扣除非经常性" in clause or "扣非" in clause or "亏损" in clause or "净利润" not in clause:
                continue
            if re.search(r"左右|约|[%％]\s*[-~～—–至到]", clause):
                continue
            matches = re.findall(
                r"净利润(?:(?!营业收入|收入|成本|现金流|扣非|每股).){0,60}?"
                r"同比(?:增长|增加|下降|减少)([-+]?\d+(?:\.\d+)?)\s*[%％]",
                clause,
            )
            if len(matches) == 1 and len(re.findall(r"同比", clause)) == 1:
                value = float(matches[0])
                if re.search(r"同比(?:下降|减少)", clause):
                    value = -abs(value)
                if math.isfinite(value) and abs(value) <= 10000:
                    found.add(value)
    return found.pop() if len(found) == 1 else None


def event_window(membership: list, docs: list, results: list, timing: str) -> list:
    """同一发行人同一业绩期在窗口内保留最新披露；修正标题亦能使旧预测失效。

    无法识别相同期间的文档不擅自合并。原有r7文档去重继续保留。
    """
    chosen, other = {}, []
    for index, age in membership:
        doc = docs[index]
        title = doc["title"]
        period = re.search(r"(20\d{2})年?(年度|半年度|第一季度|第三季度|一季度|三季度)", title)
        if (
            len(doc["issuer_codes"]) != 1
            or not period
            or not re.search(r"业绩|报告", title)
            or "问询" in title
            or "说明" in title
        ):
            other.append([index, age])
            continue
        key = (doc["issuer_codes"][0], period.group(1), period.group(2).replace("第", ""))
        # 同日多份原件不能靠目录顺序判定修订顺序；同日优先明确修正，否则优先已实现报告。
        rank = (
            doc["effective_" + timing] or "",
            bool(re.search(r"修正|更正", title)),
            results[index].get("stage") == "REALIZED",
            doc["event_id"],
        )
        if key not in chosen or rank > chosen[key][0]:
            chosen[key] = (rank, [index, age])
    return sorted(other + [item[1] for item in chosen.values()])


def build(root: Path = ROOT) -> dict:
    """生成每个历史时点的输入及关联证据，构建过程中不读取净值答案。"""
    if not (root / "extraction-complete.json").exists():
        raise ValueError("ALL_DOCUMENT_DISPOSITIONS_REQUIRED")
    docs = io.lines(root / "documents.jsonl")
    original_events = io.lines(OLD / "events.jsonl")
    reports = io.read(root / "reports.json")
    results = [io.read(root / "validated-v4" / (d["event_id"] + ".json")) for d in docs]
    profiles = defaultdict(list)
    for doc, result in zip(docs, results, strict=True):
        if result["status"] == "SOURCE_GROUNDED" and len(doc["issuer_codes"]) == 1:
            for product in result["products"]:
                profiles[(doc["issuer_codes"][0], product["term"])].append(
                    {
                        **product,
                        "event_id": doc["event_id"],
                        "effective_main": doc["effective_main"],
                        "effective_aux": doc["effective_aux"],
                        "known_at": known_at(doc),
                        "text_sha256": doc["text_sha256"],
                    }
                )
    profile_rows = [{"code": code, "term": term, "evidence": items} for (code, term), items in sorted(profiles.items())]
    io.save(root / "business-profiles.json", profile_rows)
    rows, members = io.lines(OLD / "daily-inputs.jsonl"), io.lines(OLD / "event-membership.jsonl")
    output, evidence, coverage = [], [], Counter()
    linked_events, direct_events, indirect_events = set(), set(), set()
    for row, member in zip(rows, members, strict=True):
        # 组合背景统一取早晨已知版本，便于08:00全信息分支直接复用，禁止借下午报告回填。
        cutoff = row["target"] + "T08:00:00+08:00"
        report = e.select_report(reports, cutoff)
        context = (
            [1, 0, 0, 0, 0]
            if report is None
            else [
                0,
                (date.fromisoformat(row["target"]) - date.fromisoformat(report["report_end"])).days,
                (date.fromisoformat(row["target"]) - date.fromisoformat(report["available_at"][:10])).days,
                float(report["disclosed_nav_pct"]) / 100,
                float(report["full_stock_disclosure"]),
            ]
        )
        out = {
            "target": row["target"],
            "base": row["base"],
            "session_index": row["session_index"],
            "n8": row["n8"],
            "context": context,
        }
        for timing in ("aux", "morning"):
            cutoff = row["target"] + ("T08:00:00+08:00" if timing == "morning" else "T15:00:00+08:00")
            report = e.select_report(reports, cutoff)
            eligible_members = [[index, age] for index, age in member["aux"] if known_at(docs[index]) <= cutoff]
            # 准入时间筛选必须先于修订去重，否则未来的修正会挤掉当前仍有效的旧版本。
            window = event_window(eligible_members, docs, results, "aux")
            global_x = np.zeros((3, 3, len(SEMANTICS)))
            holding_x = np.zeros_like(global_x)
            quality = np.zeros((3, len(DIAGNOSTICS)))
            valid_today, weighted_today, indirect_today = 0, 0, 0
            for index, age in window:
                doc, result = docs[index], results[index]
                if not doc["effective_aux"] or doc["effective_aux"] > row["target"]:
                    raise ValueError("FUTURE_DOCUMENT_IN_WINDOW")
                t = e.TYPES.index(doc["type"])
                relations = links(doc, result, report, profiles, cutoff, "aux")
                if not relations:
                    quality[t, DIAGNOSTICS.index("unmatched")] += 1
                if any(link["weight_is_rounded"] for link in relations):
                    quality[t, DIAGNOSTICS.index("weight_rounded_zero")] += 1
                if result["status"] != "SOURCE_GROUNDED":
                    reason = (
                        "body_missing" if result["status"] == "INSUFFICIENT_TITLE_ONLY" else "extraction_quarantined"
                    )
                    quality[t, DIAGNOSTICS.index(reason)] += 1
                    continue
                quality[t, DIAGNOSTICS.index("grounded")] += 1
                vector = semantic_vector(result)
                valid_today += 1
                direct_weight = sum(
                    link["nav_weight_pct"] / 100 for link in relations if link["relation"] == "ISSUER_CODE"
                )
                indirect_weight = sum(
                    link["nav_weight_pct"] / 100 for link in relations if link["relation"] != "ISSUER_CODE"
                )
                weight_vector = direct_weight * vector + indirect_weight * semantic_vector(result, indirect=True)
                if direct_weight + indirect_weight > 0:
                    weighted_today += 1
                    linked_events.add(index)
                    if direct_weight:
                        direct_events.add(index)
                    if indirect_weight:
                        indirect_today += 1
                        indirect_events.add(index)
                for w, days in enumerate(WINDOWS):
                    if age < days:
                        global_x[t, w] += (0.9**age) * vector
                        holding_x[t, w] += (0.9**age) * weight_vector
                if timing == "aux" and age == 0 and relations:
                    evidence.append(
                        {
                            "target": row["target"],
                            "event_id": doc["event_id"],
                            "links": relations,
                            "impact_quote": result["impact_quote"],
                            "stage": result["stage"],
                            "direction": result["direction"],
                            "facts": result["facts"],
                            "price_effect": None,
                        }
                    )
            out[timing] = {
                "global": np.log1p(global_x).ravel().tolist(),
                "weighted": np.log1p(holding_x).ravel().tolist(),
                "quality": np.log1p(quality).ravel().tolist(),
                "valid_events": valid_today,
                "weighted_events": weighted_today,
                "indirect_events": indirect_today,
                "legacy_counts": legacy_features.counts_and_members(
                    original_events,
                    window,
                    {h["stock_code"][:6] for h in report["holdings"]} if report else set(),
                ),
            }
            if timing == "aux":
                for name, value in (
                    ("body_available_dates", valid_today),
                    ("weighted_dates", weighted_today),
                    ("indirect_dates", indirect_today),
                ):
                    coverage[row["target"][:4] + "_" + name] += bool(value)
        output.append(out)
    io.save_lines(root / "feature-rows.jsonl", output)
    io.save_lines(root / "holding-evidence.jsonl", evidence)
    report = {
        "dates": len(rows),
        "documents": len(docs),
        "statuses": dict(Counter(r["status"] for r in results)),
        "coverage_by_year": dict(coverage),
        "unique_linked_events": len(linked_events),
        "direct_events": len(direct_events),
        "indirect_events": len(indirect_events),
        "profile_pairs": len(profiles),
        "columns": {
            "context": CONTEXT,
            "global": GLOBAL_COLUMNS,
            "weighted": WEIGHT_COLUMNS,
            "quality": QUALITY_COLUMNS,
        },
        "body_documents_by_year_type": dict(
            Counter(d["published_date"][:4] + "_" + d["type"] for d in docs if d["text_level"] != "TITLE_ONLY")
        ),
        "unknown_policy_sign_never_inherited": True,
    }
    io.save(root / "feature-audit.json", report)
    io.save(
        root / "feature-freeze.json",
        {
            "files": {
                str(p): io.sha(p)
                for p in [
                    root / "feature-rows.jsonl",
                    root / "documents.jsonl",
                    root / "reports.json",
                    Path(e.__file__),
                    Path(__file__),
                ]
            },
            "at": io.now(),
        },
    )
    return report


def numeric(rows: list[dict], candidate: str, timing: str) -> np.ndarray:
    result = []
    for row in rows:
        values = row["n8"] + row["context"]
        if candidate in ("M2", "M3"):
            # 两种事件模型共享缺失状态，防止把没有正文误当没有事件。
            values += row[timing]["quality"]
            values += row[timing]["global" if candidate == "M2" else "weighted"]
        result.append(values)
    return np.asarray(result, dtype=np.float64)


def labels(rows: list[dict], nav: dict) -> list[str]:
    result = []
    for row in rows:
        a, b = Decimal(nav[row["base"]]["unit_nav"]), Decimal(nav[row["target"]]["unit_nav"])
        if not (a.is_finite() and b.is_finite() and a > 0 and b > 0):
            raise ValueError("INVALID_NAV_LABEL")
        result.append("UP" if b > a else "DOWN" if b < a else "FLAT")
    return result


def fit_one(root: Path, stage: str, candidate: str, timing: str = "aux", replay: bool = False) -> dict:
    """只有训练日期拟合尺度与分类器；一次登记对应一次真实监督fit。"""
    name = f"{stage}_{timing}_{candidate}" + ("_replay" if replay else "")
    destination = root / "models" / name
    if (destination / "complete.json").exists():
        return io.read(destination / "complete.json")
    freeze = io.read(root / "feature-freeze.json")
    for path, expected in freeze["files"].items():
        if io.sha(path) != expected:
            raise ValueError("FROZEN_FEATURE_CODE_CHANGED")
    plan = io.read(root / "plan.json")
    ledger = io.lines(root / "fit-ledger.jsonl")
    if len(ledger) >= plan["max_supervised_fits"]:
        raise ValueError("FIT_BUDGET_EXHAUSTED")
    rows = io.lines(root / "feature-rows.jsonl")
    split = io.read(OLD / "split-manifest.json")[stage]
    tr = [rows[i] for i in split["train_indices"]]
    er = [rows[i] for i in split["evaluation_indices"]]
    if [r["target"] for r in tr] != split["train_dates"] or [r["target"] for r in er] != split["evaluation_dates"]:
        raise ValueError("COMPARISON_DATE_MISMATCH")
    if replay and stage == "FINAL":
        er = tr[-180:]
    x = numeric(tr, candidate, timing)
    scaler = StandardScaler().fit(x)
    nav = io.read(OLD / "snapshot/nav-facts.json")
    y = labels(tr, nav)
    model = LogisticRegression(**plan["recipe"])
    io.append(
        root / "fit-ledger.jsonl",
        {
            "name": name,
            "at": io.now(),
            "training_dates": len(tr),
            "input_sha256": io.digest(x.tolist()),
            "train_dates_sha256": io.digest(split["train_dates"]),
            "parameters": plan["recipe"],
        },
    )
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ValueError("MODEL_NOT_CONVERGED")
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "candidate": candidate,
            "timing": timing,
            "feature_freeze_sha256": io.sha(root / "feature-freeze.json"),
        },
        destination / "model.joblib",
    )
    predictions = []
    if er:
        z = scaler.transform(numeric(er, candidate, timing))
        predicted, probs = model.predict(z), model.predict_proba(z)
        predictions = [
            {
                "target": row["target"],
                "predicted": str(predicted[i]),
                "probabilities": dict(zip(model.classes_.tolist(), probs[i].tolist(), strict=True)),
            }
            for i, row in enumerate(er)
        ]
    io.save_lines(destination / "predictions.jsonl", predictions)
    completion = {
        "name": name,
        "training_dates": len(tr),
        "evaluation_dates": len(er),
        "converged": True,
        "iterations": model.n_iter_.tolist(),
        "features": x.shape[1],
        "model_sha256": io.sha(destination / "model.joblib"),
        "predictions_sha256": io.sha(destination / "predictions.jsonl"),
    }
    io.save(destination / "complete.json", completion)
    return completion


def train(root: Path = ROOT) -> dict:
    for stage in ("V2023", "C2024", "C2025", "C2026", "FINAL"):
        for candidate in ("M1", "M2", "M3"):
            print(json.dumps(fit_one(root, stage, candidate), ensure_ascii=False), flush=True)
    return {"fits": len(io.lines(root / "fit-ledger.jsonl"))}


def compare(root: Path = ROOT) -> dict:
    """所有指定预测保存后再统一算分；已见过的年份只算回溯比较，不能叫新盲测。"""
    nav, rows = io.read(OLD / "snapshot/nav-facts.json"), io.lines(root / "feature-rows.jsonl")
    by_date = {r["target"]: r for r in rows}
    result, pooled = {}, defaultdict(list)
    for stage in ("V2023", "C2024", "C2025", "C2026"):
        dates = io.read(OLD / "split-manifest.json")[stage]["evaluation_dates"]
        selected = [by_date[d] for d in dates]
        actual = labels(selected, nav)
        definitions = {"B0": OLD / "models" / f"{stage}_main_B0", "B2": OLD / "models" / f"{stage}_aux_B2"}
        definitions.update({m: root / "models" / f"{stage}_aux_{m}" for m in ("M1", "M2", "M3")})
        predictions = {}
        for key, path in definitions.items():
            saved = io.lines(path / "predictions.jsonl")
            if [p["target"] for p in saved] != dates:
                raise ValueError("PAIRED_PREDICTION_DATES_DIFFER")
            predictions[key] = [p["predicted"] for p in saved]
            pooled[key].extend(zip(actual, predictions[key], strict=True))
        metrics = {key: stats.score(actual, pred) for key, pred in predictions.items()}
        comparison = {}
        for baseline in ("B0", "B2", "M1", "M2"):
            delta = [
                int(a == c) - int(a == b)
                for a, c, b in zip(actual, predictions["M3"], predictions[baseline], strict=True)
            ]
            comparison[baseline] = {
                "extra_correct": sum(delta),
                "accuracy_delta": sum(delta) / len(delta),
                "interval": stats.block_interval(delta, [r["session_index"] for r in selected]),
            }
        result[stage] = {
            "scores": metrics,
            "M3_minus": comparison,
            "dates_with_body_semantics": sum(r["aux"]["valid_events"] > 0 for r in selected),
            "dates_with_weighted_semantics": sum(r["aux"]["weighted_events"] > 0 for r in selected),
        }
    summary = {
        "folds": result,
        "pooled": {k: stats.score([a for a, _ in v], [b for _, b in v]) for k, v in pooled.items()},
        "adopted": False,
        "prospective_verified": False,
        "reason": "RESEARCH_ONLY_INCOMPLETE_RECENT_BODY_COVERAGE",
    }
    io.save(root / "comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "train", "compare", "replay"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "replay":
        result = [fit_one(args.root, stage, "M3", replay=True) for stage in ("C2026", "FINAL")]
    else:
        result = {"build": build, "train": train, "compare": compare}[args.command](args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
