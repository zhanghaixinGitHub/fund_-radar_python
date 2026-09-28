"""002112 独立离线候选的数值契约和同日比较，不登记模型、不进入正式推理路由。"""

import math
import warnings
from collections import Counter
from datetime import datetime
from decimal import Decimal

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import fund_training_package as package
from app.services.direction_1d_protocol import RECIPE, digest
from app.services.direction_1d_three_state import CLASSES, TIE_ORDER
from app.services.direction_1d_training import weights

PROTOCOL = "FUND_002112_OFFLINE_THREE_STATE_V1"
VARIANTS = dict(package.PLAN["variants"])
SELECTION = {
    "correct_strictly_above_A_and_best_constant": True,
    "every_class_correct_at_least_A": True,
    "ranking": "CORRECT_DESC_THEN_FEWER_FEATURES",
    "automatic_adoption": False,
}


def validate_data(data):
    """防止从外部传入不合格数组；完整来源重算由冻结层在此之前执行。

    目标日期、原始十进制净值、成熟时间、家族日期和权重均单独检查；
    不使用四舍五入涨跌幅重建标签，也不将持平合并或缺失数字补零。
    """
    days = [str(d) for d in package.own.sessions()[0]]
    positions = {d: i for i, d in enumerate(days)}
    for split in ("train", "development"):
        seen = set()
        for row in data[split]:
            target, base = row["target"], row["base"]
            if target not in positions or positions[target] == 0 or days[positions[target] - 1] != base:
                raise ValueError("OFFLINE_TRADING_DATE_INVALID")
            if row["fund_code"] not in {package.FUND, *package.PEERS}:
                raise ValueError("OFFLINE_COHORT_INVALID")
            if split == "train":
                if target > package.PLAN["fit_end"]:
                    raise ValueError("OFFLINE_FUTURE_LABEL")
                if datetime.fromisoformat(row["mature_at"]) > datetime.fromisoformat(package.PLAN["fit_as_of"]):
                    raise ValueError("OFFLINE_IMMATURE_LABEL")
            elif row["fund_code"] != package.FUND or not "2024-01-01" <= target <= "2024-12-31":
                raise ValueError("OFFLINE_DEVELOPMENT_SCOPE")
            key = (row["family"], target)
            if key in seen:
                raise ValueError("OFFLINE_DUPLICATE_FAMILY_DATE")
            seen.add(key)
            before, after = Decimal(row["base_unit_nav"]), Decimal(row["target_unit_nav"])
            expected = "UP" if after > before else "DOWN" if after < before else "FLAT"
            if row["actual_direction"] != expected:
                raise ValueError("OFFLINE_EXACT_LABEL_MISMATCH")
            exposure = row["exposure"]
            if datetime.fromisoformat(exposure["report_available_at"]) > datetime.fromisoformat(row["as_of"]):
                raise ValueError("OFFLINE_FUTURE_REPORT")
            vector = row["x"]
            if len(vector) != 20 or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                raise ValueError("OFFLINE_INPUT_VALUE_OR_SHAPE")
            if vector != row["nav_features"] + exposure["holdings_features"] + exposure["market_features"]:
                raise ValueError("OFFLINE_INPUT_ORDER")
    if not package.eligibility(data["train"], data["development"])["training_ready"]:
        raise ValueError("OFFLINE_SAMPLE_GATE")
    if weights(data["train"]).tolist() != data["weights"]:
        raise ValueError("OFFLINE_WEIGHTS_CHANGED")
    if any(not math.isfinite(w) or w <= 0 for w in data["weights"]):
        raise ValueError("OFFLINE_WEIGHTS_INVALID")


def predict(model, values):
    """还原 JSON 参数，严格绑定基金、周期、维度、顺序和类别；分数不称为校准概率。"""
    size = VARIANTS.get(model.get("variant"))
    if (
        size is None
        or model.get("protocol") != PROTOCOL
        or model.get("fund_code") != package.FUND
        or model.get("horizon") != 1
        or model.get("features") != package.PLAN["features"][:size]
        or model.get("class_order") != list(CLASSES)
        or model.get("tie_order") != list(TIE_ORDER)
        or model.get("recipe") != RECIPE
    ):
        raise ValueError("OFFLINE_MODEL_PROTOCOL")
    vectors = [model.get("mean", []), model.get("scale", []), *model.get("coef", [])]
    intercept = model.get("intercept", [])
    if (
        len(vectors) != 5
        or any(len(v) != size for v in vectors)
        or len(values) != size
        or len(intercept) != 3
        or any(
            type(v) not in (float, int) or not math.isfinite(v) for row in [*vectors, values, intercept] for v in row
        )
        or any(v <= 0 for v in model["scale"])
    ):
        raise ValueError("OFFLINE_MODEL_VALUE_OR_SHAPE")
    normalized = [(x - m) / s for x, m, s in zip(values, model["mean"], model["scale"], strict=True)]
    logits = [
        b + sum(w * x for w, x in zip(row, normalized, strict=True))
        for row, b in zip(model["coef"], intercept, strict=True)
    ]
    if not all(math.isfinite(v) for v in logits):
        raise ValueError("OFFLINE_NONFINITE_LOGIT")
    exp = [math.exp(v - max(logits)) for v in logits]
    scores = {k: v / sum(exp) for k, v in zip(CLASSES, exp, strict=True)}
    return {"direction": max(TIE_ORDER, key=scores.__getitem__), "class_scores": scores}


def fit(data, variant):
    """仅一次拟合：训练权重标准化后使用固定配方，独立复现由运行账本另行调用。

    开发记录只参与参数还原核对及输出，绝不传入 scaler.fit 或 classifier.fit。
    收敛警告视为技术失败；运行层已在调用前永久占用一次预算。
    """
    if variant not in VARIANTS:
        raise ValueError("OFFLINE_VARIANT_INVALID")
    size = VARIANTS[variant]
    x = np.asarray([r["x"][:size] for r in data["train"]], dtype=float)
    y = np.asarray([r["actual_direction"] for r in data["train"]])
    sample_weight = np.asarray(data["weights"], dtype=float)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = StandardScaler().fit(x, sample_weight=sample_weight)
        classifier = LogisticRegression(**RECIPE).fit(scaler.transform(x), y, sample_weight=sample_weight)
        if classifier.classes_.tolist() != list(CLASSES):
            raise ValueError("OFFLINE_THREE_CLASSES_REQUIRED")
        model = {
            "protocol": PROTOCOL,
            "fund_code": package.FUND,
            "horizon": 1,
            "variant": variant,
            "features": package.PLAN["features"][:size],
            "recipe": RECIPE,
            "class_order": list(CLASSES),
            "tie_order": list(TIE_ORDER),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "dataset_hash": digest(data),
            "weight_hash": digest(data["weights"]),
            "fit_count": len(x),
            "train_end": max(r["target"] for r in data["train"]),
        }
        evidence = {}
        for split in ("train", "development"):
            rows = data[split]
            restored = [predict(model, r["x"][:size]) for r in rows]
            library = classifier.predict_proba(scaler.transform(np.asarray([r["x"][:size] for r in rows])))
            score_array = np.asarray([[r["class_scores"][k] for k in CLASSES] for r in restored])
            difference = float(np.max(np.abs(library - score_array)))
            library_directions = [max(TIE_ORDER, key=lambda k, p=p: p[list(CLASSES).index(k)]) for p in library]
            if difference > 1e-12 or library_directions != [r["direction"] for r in restored]:
                raise ValueError("OFFLINE_RESTORE_MISMATCH")
            evidence[split] = {"max_score_difference": difference, "directions_equal": True, "count": len(rows)}
        evidence["iterations"] = classifier.n_iter_.tolist()
    return {"model": model, "restore": evidence}


def predictions(model, rows):
    size = VARIANTS[model["variant"]]
    return [
        {
            "fund_code": r["fund_code"],
            "target": r["target"],
            "base": r["base"],
            "actual_direction": r["actual_direction"],
            "input_hash": digest(r),
            "model_hash": digest(model),
            **predict(model, r["x"][:size]),
        }
        for r in rows
    ]


def metrics(rows):
    actual = Counter(r["actual_direction"] for r in rows)
    correct = {k: sum(r["actual_direction"] == r["direction"] == k for r in rows) for k in CLASSES}
    recalls = {k: correct[k] / actual[k] if actual[k] else None for k in CLASSES}
    measured = [v for v in recalls.values() if v is not None]
    return {
        "count": len(rows),
        "correct": sum(correct.values()),
        "actual": {k: actual[k] for k in CLASSES},
        "class_correct": correct,
        "recall": recalls,
        "balanced_recall": sum(measured) / len(measured) if measured else None,
        "balanced_recall_class_count": len(measured),
        "confusion": {
            k: {p: sum(r["actual_direction"] == k and r["direction"] == p for r in rows) for p in CLASSES}
            for k in CLASSES
        },
    }


def compare(data, by_variant):
    """从逐日输出重算指标，日期严格一致；固定筛选规则不随成绩更换。"""
    dates = [r["target"] for r in data["development"]]
    expected = [(r["target"], r["actual_direction"], digest(r)) for r in data["development"]]
    for rows in by_variant.values():
        if [(r["target"], r["actual_direction"], r["input_hash"]) for r in rows] != expected:
            raise ValueError("OFFLINE_COMPARISON_ROWS_CHANGED")
    scores = {v: metrics(rows) for v, rows in by_variant.items()}
    weighted = {
        k: sum(w for r, w in zip(data["train"], data["weights"], strict=True) if r["actual_direction"] == k)
        for k in CLASSES
    }
    majority = max(TIE_ORDER, key=weighted.__getitem__)
    constants = {k: metrics([{**r, "direction": k} for r in data["development"]]) for k in CLASSES}
    quarters = {
        str(q): {
            v: metrics([r for r in rows if (int(r["target"][5:7]) - 1) // 3 + 1 == q]) for v, rows in by_variant.items()
        }
        for q in range(1, 5)
    }
    differences = {}
    base = by_variant.get("NAV7")
    for variant, rows in by_variant.items():
        if base is None or variant == "NAV7":
            continue
        daily = []
        for a, b in zip(base, rows, strict=True):
            ac, bc = a["direction"] == a["actual_direction"], b["direction"] == b["actual_direction"]
            outcome = "both_correct" if ac and bc else "A_only" if ac else "new_only" if bc else "both_wrong"
            daily.append({"target": a["target"], "outcome": outcome})
        differences[variant] = {
            "counts": {
                k: sum(r["outcome"] == k for r in daily) for k in ("both_correct", "A_only", "new_only", "both_wrong")
            },
            "daily": daily,
        }
    qualified = []
    if "NAV7" in scores:
        a = scores["NAV7"]
        for v, score in scores.items():
            if (
                v != "NAV7"
                and score["correct"] > max(a["correct"], *(c["correct"] for c in constants.values()))
                and all(score["class_correct"][k] >= a["class_correct"][k] for k in CLASSES)
            ):
                qualified.append(v)
    qualified.sort(key=lambda v: (-scores[v]["correct"], VARIANTS[v]))
    complete = set(by_variant) == set(VARIANTS)
    excluded = [
        r for r in data["excluded"] if r["fund_code"] == package.FUND and "2024-01-01" <= r["target"] <= "2024-12-31"
    ]
    return {
        "fund_code": package.FUND,
        "kind": "OBSERVED_2024_DEVELOPMENT_NOT_BLIND_TEST",
        "dates": dates,
        "models": scores,
        "constants": constants,
        "weighted_majority": {
            "direction": majority,
            "training_weighted_counts": weighted,
            "metrics": constants[majority],
        },
        "quarters": quarters,
        "relative_to_A": differences,
        "coverage": {
            "eligible": len(dates),
            "excluded_dates": len({r["target"] for r in excluded}),
            "excluded_reasons": dict(Counter(r["reason"] for r in excluded)),
            "excluded": excluded,
        },
        "training_summary": package.eligibility(data["train"], data["development"]),
        "complete": complete,
        "selected_for_further_experiment": qualified[0] if complete and qualified else None,
        "conclusion": "INCOMPLETE"
        if not complete
        else "WORTH_FURTHER_EXPERIMENT"
        if qualified
        else "NO_DEMONSTRATED_HELP",
        "adopted": False,
    }


def render_report(result):
    """报告保留三类分母、空季度和简单参照，不把方向命中转写为投资收益。"""
    names = {
        "NAV7": "A：净值 7 项",
        "NAV7_HOLDINGS": "B：净值＋持仓 16 项",
        "NAV7_HOLDINGS_MARKET": "C：净值＋持仓＋大盘 20 项",
        "UP": "始终上涨",
        "FLAT": "始终持平",
        "DOWN": "始终下跌",
    }
    lines = [
        "# 002112 一日离线模型开发检查",
        "",
        "口径：下一交易日单位净值方向，原始数值完全相等才算持平。",
        "2024 已被观察过，仅作开发检查；以下不是独立盲测、实际提前预测或收益证明。",
        "",
        "| 做法 | 正确 / 总天数 | 上涨正确 / 实际 | 持平正确 / 实际 | 下跌正确 / 实际 | 平均类别召回率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, score in {**result["models"], **result["constants"]}.items():
        cells = [f"{score['class_correct'][k]} / {score['actual'][k]}" for k in ("UP", "FLAT", "DOWN")]
        lines.append(
            f"| {names[name]} | {score['correct']} / {score['count']} | "
            + " | ".join(cells)
            + f" | {score['balanced_recall']:.4%} |"
        )
    majority = result["weighted_majority"]
    lines += [
        "",
        f"训练集加权多数类为 {majority['direction']}，固定猜它对 {majority['metrics']['correct']} 天。",
        "",
        "## 相对 A 的逐日差异",
        "",
        "| 做法 | 新做法对、A 错 | A 对、新做法错 | 都对 | 都错 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for v, diff in result["relative_to_A"].items():
        c = diff["counts"]
        lines.append(f"| {names[v]} | {c['new_only']} | {c['A_only']} | {c['both_correct']} | {c['both_wrong']} |")
    lines += [
        "",
        "## 分季度",
        "",
        "| 季度 | 做法 | 正确 / 总数 | 上涨召回 | 持平召回 | 下跌召回 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for q, variants in result["quarters"].items():
        for v, score in variants.items():
            recall = [
                "该季度没有这类样本" if score["recall"][k] is None else f"{score['recall'][k]:.2%}"
                for k in ("UP", "FLAT", "DOWN")
            ]
            lines.append(f"| Q{q} | {names[v]} | {score['correct']} / {score['count']} | " + " | ".join(recall) + " |")
    lines += ["", "## 三类混淆表", "", "行是真实类别，列是判断类别；完整逐日数据见 predictions 和 comparison.json。"]
    for v, score in result["models"].items():
        lines += ["", names[v], "", "| 实际 / 判断 | DOWN | FLAT | UP |", "| --- | ---: | ---: | ---: |"]
        for k, counts in score["confusion"].items():
            lines.append(f"| {k} | " + " | ".join(str(counts[p]) for p in CLASSES) + " |")
    conclusion = {
        "INCOMPLETE": "有候选未完成，不能宣称整轮训练通过。",
        "WORTH_FURTHER_EXPERIMENT": "增强版本满足固定开发筛选规则，值得继续实验。",
        "NO_DEMONSTRATED_HELP": "本轮未证明新增资料改善目标开发表现。",
    }[result["conclusion"]]
    coverage = result["coverage"]
    lines += [
        "",
        "## 结论与限制",
        "",
        conclusion,
        f"只评价 {coverage['eligible']} 个合格日期，另有 {coverage['excluded_dates']} 个目标日期被排除，"
        "原因见 comparison.json。",
        f"002112 自身训练持平 {result['training_summary']['target_train']['classes']['FLAT']} 天，"
        "采用自身历史加固定十基金方案；"
        f"开发集持平 {result['training_summary']['target_development']['classes']['FLAT']} 天，不能证明稳定识别能力。",
        "历史可用时间按披露日期重建，不证明当年已实时收到。参照基金风格不同，成绩可能不能迁移到未来。",
        "分类分数未校准。没有登记、实验采用、页面接入或正式切换；当前页面未使用本轮候选。",
        "",
    ]
    return "\n".join(lines)
