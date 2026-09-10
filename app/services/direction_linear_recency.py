"""近期成熟样本对照及时间迁移诊断；同快照独立复算不等于外部数值核实。"""

from collections import Counter
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from math import sqrt

from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services import direction_nav_data as data
from app.services.cash_reinvestment_samples import _event_available_at, _events_for_period
from app.services.direction_followup_data import restore_fund
from app.services.direction_linear_models import predict_model
from app.services.direction_linear_protocol import RECENCY_BRANCHES, RECENCY_VERSION
from app.services.direction_training_artifacts import digest, read_json, read_jsonl
from app.services.direction_training_dataset import END, FUNDS, START
from app.services.direction_training_evaluation import grouped_metrics
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.trading_calendar import load_calendar


def independent_series(dates, nav, events, limit):
    """用末值/初值乘分红份额增长复算，独立于训练用的相邻日现金收益连乘。"""
    cash, issues = _events_for_period(events, dates, limit)
    if issues:
        raise ValueError("RECENCY_AUDIT_DIVIDEND_POLICY")
    values = [nav[d].unit_nav for d in dates]
    if any(v is None or not v.is_finite() or v <= 0 for v in values):
        raise ValueError("RECENCY_AUDIT_NAV_VALUE")
    with localcontext() as context:
        context.prec = 50
        shares, result = Decimal(1), []
        for day, value in zip(dates, values, strict=True):
            if day in cash:
                shares *= 1 + cash[day].cash_dividend / value
            result.append((100 * value / values[0] * shares).quantize(Decimal("1e-12"), rounding=ROUND_HALF_UP))
    available = max([data.assumed_available(dates[-1]), *(_event_available_at(e) for e in cash.values())])
    return result, available


def independent_features(series):
    """按固定七项定义另算一遍，保留原收益61点、回撤/位置60点及8位舍入。"""
    with localcontext() as context:
        context.prec = 50
        returns = [series[-1] / series[-n - 1] - 1 for n in (5, 20, 60)]
        daily = [series[i] / series[i - 1] - 1 for i in range(len(series) - 20, len(series))]
        mean = sum(daily) / 20
        volatility = (sum((v - mean) ** 2 for v in daily) / 20).sqrt()
        recent = series[-60:]
        drawdown = min(v / max(recent[: i + 1]) - 1 for i, v in enumerate(recent))
        position = (recent[-1] - min(recent)) / (max(recent) - min(recent))
        decline = 0
        for i in range(len(series) - 1, 0, -1):
            if series[i] >= series[i - 1]:
                break
            decline += 1
        return [
            float(v.quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP))
            for v in (*returns, volatility, drawdown, position)
        ] + [decline]


def audit_input(item, record, nav, events):
    item = DirectionInput.model_validate(item)
    dates = data.history_dates(item.cutoff)
    audit = record["audit"]["CLEAN"]
    if (
        len(dates) != 61
        or item.anchor != dates[-1]
        or audit["dates"] != [str(d) for d in dates]
        or audit["missing_dates"]
        or digest(audit) != item.input_hash
    ):
        raise ValueError("RECENCY_AUDIT_INPUT_HISTORY")
    series, available = independent_series(dates, nav, data.known_events(events, item.cutoff), item.cutoff)
    if available != item.available_at or available > item.cutoff:
        raise ValueError("RECENCY_AUDIT_INPUT_AVAILABILITY")
    if tuple(independent_features(series)) != item.x:
        raise ValueError("RECENCY_AUDIT_FEATURE_MISMATCH")


def audit_answer(answer, nav, events):
    answer = DirectionAnswer.model_validate(answer)
    calendar = load_calendar()
    future = calendar.future_sessions(answer.cutoff)
    base = calendar.sessions[calendar.at_or_before_index(answer.cutoff)]
    if answer.end != future[-1] or len(future) != 20:
        raise ValueError("RECENCY_AUDIT_LABEL_DATES")
    series, available = independent_series((base, *future), nav, events, END)
    with localcontext() as context:
        context.prec = 50
        expected = (series[-1] / 100 - 1).quantize(Decimal("1e-12"), rounding=ROUND_HALF_UP)
    if expected != Decimal(answer.future_return) or int(expected > 0) != answer.y or available != answer.available_at:
        raise ValueError("RECENCY_AUDIT_LABEL_MISMATCH")


def latest_rows(reference, mature):
    """每只基金保留与原训练集完全相同的行数，按截止日选最近样本，不读取答案挑行。"""
    counts = Counter(r["input"]["fund"] for r in reference)
    if set(counts) != set(FUNDS) or len({(r["input"]["fund"], r["input"]["cutoff"]) for r in mature}) != len(mature):
        raise ValueError("RECENCY_COUNT_OR_DUPLICATE")
    selected = []
    for fund in FUNDS:
        rows = sorted((r for r in mature if r["input"]["fund"] == fund), key=lambda r: r["input"]["cutoff"])
        if len(rows) < counts[fund]:
            raise ValueError("RECENCY_COUNT_SHORTAGE")
        selected.extend(rows[-counts[fund] :])
    return selected


def load_sources(folder):
    result = {}
    for fund in FUNDS:
        _, points, events = restore_fund(read_json(folder / f"source-{fund}.json"))
        rows = read_jsonl(folder / f"nav-inputs-{fund}.jsonl")
        records = {r["cutoff"]: r for r in rows}
        if len(records) != len(rows):
            raise ValueError("RECENCY_SOURCE_DUPLICATE")
        result[fund] = ({p.nav_date: p for p in points}, events, records)
    return result


def extend_fit(bundle, sources):
    window, reference = bundle["window"], bundle["complete"]["fit"]
    limit = date.fromisoformat(window["cal_end"])
    mature, audits = [], {}
    for fund in FUNDS:
        nav, events, records = sources[fund]
        selected, excluded = [], Counter()
        for cutoff, record in sorted(records.items()):
            if cutoff > str(limit):
                continue
            if record["inputs"]["CLEAN"] is None:
                excluded.update(record["input_issues"]["CLEAN"])
                continue
            metadata = record["label"]
            if metadata is None or metadata["available_at"] > str(limit):
                excluded["ANSWER_NOT_AVAILABLE_BY_TRAINING_LIMIT"] += 1
                continue  # 先查成熟元数据，再导出答案值；不按涨跌筛选。
            item = record["inputs"]["CLEAN"]
            answer, issues = data.build_answer(fund, date.fromisoformat(cutoff), nav, events, include_value=True)
            if issues or answer is None or answer.available_at > limit:
                raise ValueError("RECENCY_TRAINING_ANSWER_BOUNDARY")
            audit_input(item, record, nav, events)
            audit_answer(answer, nav, events)
            selected.append({"input": item, "answer": answer.model_dump(mode="json")})
        mature.extend(selected)
        old = [r for r in reference if r["input"]["fund"] == fund]
        reconstructed = [r for r in selected if r["answer"]["available_at"] <= window["fit_end"]]
        if reconstructed != old:
            raise ValueError("RECENCY_REFERENCE_RECONSTRUCTION_CHANGED")
        audits[fund] = {
            "reference_count": len(old),
            "mature_count": len(selected),
            "added_count": len(selected) - len(old),
            "reference_last_cutoff": old[-1]["input"]["cutoff"],
            "mature_last_cutoff": selected[-1]["input"]["cutoff"],
            "last_label_available_at": max(r["answer"]["available_at"] for r in selected),
            "training_limit": str(limit),
            "excluded_before_limit": dict(excluded),
            "independent_feature_and_label_arithmetic_verified": len(selected),
            "reference_rows_reconstructed_identically": True,
        }
    same = latest_rows(reference, mature)
    for fund in FUNDS:
        rows = [r for r in same if r["input"]["fund"] == fund]
        audits[fund].update(same_count=len(rows), same_count_first_cutoff=rows[0]["input"]["cutoff"])
    return {"MATURE_EXPANDING": mature, "MATURE_SAME_COUNT": same}, audits


def association(rows):
    """各时段特征与答案的描述性关联；不据此调整特征或解释因果。"""
    if not rows:
        return None
    y = [r["answer"]["y"] for r in rows]
    ym = sum(y) / len(y)
    result = {}
    for j, name in enumerate(FEATURE_NAMES):
        x = [r["input"]["x"][j] for r in rows]
        xm = sum(x) / len(x)
        xv = sum((v - xm) ** 2 for v in x)
        yv = sum((v - ym) ** 2 for v in y)
        result[name] = {
            "mean": xm,
            "std": sqrt(xv / len(x)),
            "correlation_with_up": sum((a - xm) * (b - ym) for a, b in zip(x, y, strict=True)) / sqrt(xv * yv)
            if xv > 0 and yv > 0
            else None,
        }
    return {"sample_count": len(rows), "actual_up_rate": ym, "features": result}


def diagnostic_report(folder, original, protocol, answers):
    """预测已封存后比较同一模型在原训练、原先未拟合时段及考试期的变化。"""
    sources, windows = load_sources(original), {}
    answer_map = {(r["window"], r["sample_key"]): r["answer"] for r in answers}
    audited_exam, unique_inputs, unique_labels = 0, set(), set()
    for window in protocol["windows"]:
        name = window["name"]
        bundle = read_json(folder / f"linear-prepared-{name}.json")
        if bundle["complete"] is None:
            windows[name] = {"status": "INSUFFICIENT_DATA"}
            continue
        old = bundle["complete"]["fit"]
        full = bundle["recent_fit"]["MATURE_EXPANDING"]
        old_keys = {(r["input"]["fund"], r["input"]["cutoff"]) for r in old}
        gap = [r for r in full if (r["input"]["fund"], r["input"]["cutoff"]) not in old_keys]
        exam = []
        for item in bundle["complete"]["exam"]["CLEAN"]:
            value = answer_map[(name, f"{item['fund']}:{item['cutoff']}")]
            if value is None or value["available_at"] > window["exam_end"]:
                continue
            nav, events, records = sources[item["fund"]]
            audit_input(item, records[item["cutoff"]], nav, events)
            audit_answer(value, nav, events)
            audited_exam += 1
            exam.append({"input": item, "answer": value})
        stages = {"ORIGINAL_FIT": old, "PREVIOUSLY_UNUSED_MATURE": gap, "EXAM": exam}
        outputs = read_json(folder / f"linear-models-{name}.json")
        metrics = {}
        for branch in RECENCY_BRANCHES:
            output = outputs[branch]
            if output["status"] != "PREDICTED":
                metrics[branch] = {"status": output["status"]}
                continue
            own = old if branch == "REFERENCE" else bundle["recent_fit"][branch]
            fitted = {(r["input"]["fund"], r["input"]["cutoff"]) for r in own}
            metrics[branch] = {}
            for stage, rows in {**stages, "OWN_FIT": own}.items():
                values = predict_model(
                    output["models"]["POOLED"], [DirectionInput.model_validate(r["input"]) for r in rows]
                )
                result = grouped_metrics(
                    [
                        {"fund": r["input"]["fund"], "y": r["answer"]["y"], "score": v}
                        for r, v in zip(rows, values, strict=True)
                    ],
                    FUNDS,
                )
                metrics[branch][stage] = {
                    "metrics": result,
                    "overlap_with_own_fit": sum((r["input"]["fund"], r["input"]["cutoff"]) in fitted for r in rows),
                }
        for row in (*full, *exam):
            unique_inputs.add((row["input"]["fund"], row["input"]["cutoff"]))
            unique_labels.add((row["answer"]["fund"], row["answer"]["cutoff"]))
        windows[name] = {
            "status": "DIAGNOSED",
            "window": window,
            "metrics": metrics,
            "associations": {
                stage: {f: association([r for r in rows if r["input"]["fund"] == f]) for f in FUNDS}
                for stage, rows in stages.items()
            },
        }
    return {
        "version": RECENCY_VERSION,
        "source_value_status": "SAME_SNAPSHOT_ARITHMETIC_VERIFIED_NOT_INDEPENDENT_PROVIDER_VERIFICATION",
        "dividend_audit_scope": "REUSE_EVENT_ELIGIBILITY_RULES_INDEPENDENT_RETURN_ARITHMETIC",
        "first_publication_versions_verified": False,
        "independent_test": False,
        "interpretation": "DESCRIPTIVE_ONLY_OVERLAPPING_LABELS_NO_CAUSAL_PROOF_OR_NEW_PARAMETER_SEARCH",
        "audited_exam_rows": audited_exam,
        "unique_audited_input_count": len(unique_inputs),
        "unique_audited_label_count": len(unique_labels),
        "source_health": {
            f: {
                "nav_count": len(nav),
                "invalid_values": sum(
                    p.unit_nav is None or not p.unit_nav.is_finite() or p.unit_nav <= 0 for p in nav.values()
                ),
                "missing_trading_dates": [
                    str(d) for d in load_calendar().sessions if START <= d <= END and d not in nav
                ],
                "event_count": len(events),
            }
            for f, (nav, events, _) in sources.items()
        },
        "windows": windows,
    }
