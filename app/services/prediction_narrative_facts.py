"""将已核对的原预测整理成文案事实；不读取最新行情，不接纳客户端提供的理由。"""

import math
from datetime import date

from app.services.direction_1d_protocol import FEATURES

STYLE_VERSION = "PREDICTION_NARRATIVE_ZH_V4"
LABELS = {"UP": "上涨", "DOWN": "下跌", "FLAT": "持平", "NON_UP": "下跌或持平"}
PERIODS = {"T5_V1": "未来 5 个交易日", "T20_V1": "未来 20 个交易日", "M6_V1": "未来 6 个月"}
NAMES = {
    "return_5d": "近 5 个交易日累计涨跌",
    "return_20d": "近 20 个交易日累计涨跌",
    "return_60d": "近 60 个交易日累计涨跌",
    "volatility_20d": "近 20 个交易日的日波动",
    "max_drawdown_60d": "近 60 个交易日最大回撤",
    "relative_position_60d": "近 60 个交易日净值相对位置",
    "consecutive_decline_days": "连续下跌天数",
}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("INVALID_EVIDENCE_NUMBER")
    return value


def fact_text(key, value):
    """由程序固定指标与数值的对应关系，模型只能引用整项事实，不能改数值或正负号。"""
    number(value)
    if key.startswith("return_"):
        days = key.split("_")[1][:-1]
        movement = "上涨" if value > 0 else "下跌" if value < 0 else "涨跌"
        return f"近 {days} 个交易日累计{movement} {abs(value * 100):.2f}%"
    if key == "consecutive_decline_days":
        if value < 0 or value != int(value):
            raise ValueError("INVALID_DECLINE_DAYS")
        return f"连续下跌 {int(value)} 个交易日"
    if key == "volatility_20d":
        if value < 0:
            raise ValueError("INVALID_VOLATILITY")
        return f"近 20 个交易日的日波动为 {value * 100:.2f}%（每日收益率的标准差，未年化，并非平均每天涨跌幅）"
    return f"{NAMES[key]}为 {value * 100:.2f}%"


def check_reasoning(item, direction):
    """继续核对原参数关系，但仅留作后台校验，不把系数或标准化均值送给文字编辑。"""
    reason = item["reasoning"]
    if not isinstance(reason, dict):
        raise ValueError("MISSING_ORIGINAL_REASONING")
    value, reference = number(item["value"]), number(reason["referenceValue"])
    weight = number(reason["relativeWeight"])
    comparison, changed = reason["comparisonDirection"], reason["directionAtReference"]
    if comparison not in LABELS or changed not in LABELS or comparison == direction:
        raise ValueError("INVALID_REASONING_DIRECTION")
    product = number(weight * (value - reference))
    if (item["impact"] == 0) != (product == 0) or item["impact"] * product < 0:
        raise ValueError("REASONING_SIGN_MISMATCH")


def pack(direction, period, data_date, factors, *, daily=False, baseline=False, limited=False, legacy=False):
    """组织可核验的观察与解释边界；指标可合并行文，不把计算作用当作市场原因。"""
    if direction not in LABELS or date.fromisoformat(data_date).isoformat() != data_date:
        raise ValueError("INVALID_NARRATIVE_IDENTITY")
    label = "基本持平" if direction == "FLAT" and not daily else LABELS[direction]
    by_key = {item["key"]: item for item in factors}
    if not baseline:
        for item in factors:
            check_reasoning(item, direction)
    # 合并相关区间，不再强迫展示每个计算贡献项。正文必含走势与波动口径；
    # 回撤等补充观察可择一使用，已知区间矛盾和解释不足仍必须保留。
    returns = [by_key[key] for key in ("return_5d", "return_20d") if key in by_key]
    signs = {"UP" if f["value"] > 0 else "DOWN" if f["value"] < 0 else "FLAT" for f in returns}
    facts, required, optional = {}, [], []
    if baseline:
        facts["F1"] = by_key["momentum"]["text"]
        required.append("F1")
    elif len(returns) == 2:
        if len(signs) == 1 and signs != {"FLAT"}:
            move = LABELS[next(iter(signs))]
            facts["F1"] = (
                f"近 5 和 20 个交易日累计分别{move} "
                f"{abs(returns[0]['value'] * 100):.2f}% 和 {abs(returns[1]['value'] * 100):.2f}%"
            )
        else:
            facts["F1"] = "，而".join(item["text"] for item in returns)
        facts["F2"] = by_key["volatility_20d"]["text"]
        required.extend(("F1", "F2"))
        for key in ("max_drawdown_60d", "consecutive_decline_days", "relative_position_60d"):
            if key in by_key:
                ref = f"F{len(facts) + 1}"
                facts[ref] = by_key[key]["text"]
                optional.append(ref)
    else:
        raise ValueError("INSUFFICIENT_EXPLANATION_FACTS")
    # 只比较已发生的区间方向与原预测；不把两者相同或相反升级成支持证据。
    if len(signs) > 1:
        facts["Q1"] = "近期不同区间的涨跌方向并不一致，不能将其中一段走势直接外推"
    elif signs and direction not in signs and direction != "NON_UP":
        past = LABELS[next(iter(signs))]
        facts["Q1"] = f"近期区间表现为{past}，与本次偏向{label}的判断并不一致"
    if baseline:
        facts["B1"] = (
            "这次基本持平的判断仅依据这段累计涨跌处于原判断的持平范围，不能据此判断后续净值不变"
            if direction == "FLAT"
            else "这次判断仅沿用这段累计涨跌的方向，尚无其他依据确认后续会延续"
        )
    else:
        facts["B1"] = "这些历史观察不足以解释后续为何会朝预测方向变化，本次判断的解释依据仍有限"
    required.extend(key for key in ("Q1", "B1") if key in facts)
    limitations = ["仅使用原预测当时的净值信息，未包含新闻、公告或持仓变化。"]
    if "volatility_20d" in by_key:
        limitations.append("原记录未提供可核验的本基金历史波动分布，暂不判断波动处于高位或低位。")
    if limited:
        limitations.append("部分分析当时不可用，本次保留的是已有判断。")
    if daily and not legacy:
        limitations.append("下一交易日单位净值高于基准日算上涨，相等算持平，低于基准日算下跌。")
    if direction == "NON_UP" or legacy:
        limitations.append("这条历史预测将下跌与持平合并，无法再分别判断。")
    return {
        "styleVersion": STYLE_VERSION,
        "direction": direction,
        "directionLabel": label,
        "period": period,
        "dataAsOf": data_date,
        "summary": f"依据截至 {data_date} 的净值信息，本次对{period}的判断偏向“{label}”。",
        "facts": facts,
        "requiredRefs": required,
        "optionalRefs": optional,
        "limitations": limitations,
    }


def safe_narrative(facts):
    """生成不可用时仍展示已核验的事实与局限；不保存为已完成的第三方说明。"""

    def ref(key):
        return facts["facts"][key]

    first = ref("F1") + "。" + (ref("Q1") + "。" if "Q1" in facts["facts"] else "")
    second = (ref("F2") + "。" if "F2" in facts["facts"] else "") + ref("B1") + "。"
    return {
        "styleVersion": STYLE_VERSION,
        "summary": facts["summary"],
        "context": first + ("\n" if "F2" in facts["facts"] else "") + second,
        "supporting": "",
        "opposing": "",
        "limitations": facts["limitations"],
        "evidenceRefs": {"context": facts["requiredRefs"], "supporting": [], "opposing": []},
    }


def daily_facts(restored, *, ternary=True):
    """相同登记包去重；存在方向分歧时拒绝统一解释，不让文字编辑替用户选方向。"""
    branches = list({b["modelHash"]: b for b in restored["branches"]}.values())
    directions = {b["direction"] for b in branches}
    if len(directions) != 1 or not directions <= LABELS.keys():
        raise ValueError("DIRECTION_DISAGREEMENT")
    direction = next(iter(directions))
    factors = []
    for branch in branches:
        raw = branch["factors"]
        if len(raw) != 7 or {f["feature"] for f in raw} != set(FEATURES):
            raise ValueError("INCOMPLETE_FACTORS")
        toward = 1 if ternary or direction == "UP" else -1
        for f in raw:
            factors.append(
                {
                    "key": f["feature"],
                    "text": fact_text(f["feature"], f["value"]),
                    "impact": number(f["contribution"]) * toward,
                    "value": f["value"],
                    "reasoning": f.get("reasoning"),
                }
            )
    # 多份判断中同一指标出现相反作用时不能合并为一个无条件理由。
    for key in FEATURES:
        impacts = [f["impact"] for f in factors if f["key"] == key]
        if any(x > 0 for x in impacts) and any(x < 0 for x in impacts):
            raise ValueError("FACTOR_DISAGREEMENT")
        reasons = [f["reasoning"] for f in factors if f["key"] == key]
        if any(r != reasons[0] for r in reasons):
            raise ValueError("REASONING_DISAGREEMENT")
    factors = list({f["key"]: f for f in factors}.values())
    return pack(
        direction,
        "下一交易日",
        restored["baseNavDate"],
        factors,
        daily=True,
        limited=len(restored["branches"]) < 2,
        legacy=not ternary,
    )


def multi_facts(prediction):
    """按原快照核对多周期方向和贡献；未知方法或缺失原参数均停止生成，不反向编理由。"""
    snapshot = prediction["featureSnapshot"]
    if snapshot["fundCode"] != prediction["fundCode"] or snapshot["dataAsOf"] != prediction["dataAsOf"]:
        raise ValueError("SNAPSHOT_MISMATCH")
    period = PERIODS[prediction["horizonId"]]
    direction = prediction["direction"]
    manifest = prediction["modelManifest"]
    params, values = manifest["parameters"], snapshot["features"]
    adapter = manifest["adapter"]
    baseline = adapter in {"NAV_MOMENTUM_THREE_STATE_V2", "NAV_MOMENTUM_V1"}
    if baseline:
        momentum, lookback = number(values["momentum"]), number(values["actualLookbackReturns"])
        threshold = 0 if adapter == "NAV_MOMENTUM_V1" else float(params["momentumThreshold"])
        if lookback < 1 or lookback != int(lookback) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("INVALID_BASELINE")
        if adapter != "NAV_MOMENTUM_V1" and threshold == 0:
            raise ValueError("INVALID_BASELINE")
        expected = (
            ("UP" if momentum > 0 else "NON_UP")
            if adapter == "NAV_MOMENTUM_V1"
            else ("UP" if momentum > threshold else "DOWN" if momentum < -threshold else "FLAT")
        )
        change = "上涨" if momentum > 0 else "下跌" if momentum < 0 else "涨跌"
        factors = [
            {
                "key": "momentum",
                "text": f"近 {int(lookback)} 个交易日累计{change} {abs(momentum * 100):.2f}%",
                "impact": 1,
            }
        ]
    elif adapter == "LOGISTIC_MULTICLASS_V2":
        keys = manifest["features"]
        means, scales = params["mean"], params["scale"]
        coefficients, intercepts, classes = params["coefficients"], params["intercepts"], params["classes"]
        order = manifest["directionPolicySnapshot"]["tieBreakOrder"]
        if (
            len(keys) != 7
            or set(keys) != set(FEATURES)
            or classes != ["DOWN", "FLAT", "UP"]
            or sorted(order) != classes
            or len(order) != 3
            or len(means) != 7
            or len(scales) != 7
            or len(coefficients) != 3
            or len(intercepts) != 3
            or any(len(row) != 7 for row in coefficients)
            or any(number(s) <= 0 for s in scales)
        ):
            raise ValueError("INVALID_ORIGINAL_PARAMETERS")
        normalized = [(number(values[k]) - number(m)) / s for k, m, s in zip(keys, means, scales, strict=True)]
        logits = [
            number(b) + sum(number(w) * v for w, v in zip(row, normalized, strict=True))
            for row, b in zip(coefficients, intercepts, strict=True)
        ]
        for value in logits:
            number(value)
        rank = sorted(range(3), key=lambda i: (-logits[i], order.index(classes[i])))
        winner, runner = rank[:2]
        expected = classes[winner]
        factors = [
            {
                "key": key,
                "text": fact_text(key, values[key]),
                "impact": number(normalized[j] * (coefficients[winner][j] - coefficients[runner][j])),
                "value": values[key],
                "reasoning": {
                    "referenceValue": means[j],
                    "relativeWeight": coefficients[winner][j] - coefficients[runner][j],
                    "comparisonDirection": classes[runner],
                    # 把标准化后的单项置零即还原到原均值，仍按原三分类及并列规则决定结果。
                    "directionAtReference": classes[
                        min(
                            range(3),
                            key=lambda i: (-(logits[i] - coefficients[i][j] * normalized[j]), order.index(classes[i])),
                        )
                    ],
                },
            }
            for j, key in enumerate(keys)
        ]
    else:
        raise ValueError("UNSUPPORTED_EXPLANATION_METHOD")
    if expected != direction:
        raise ValueError("PREDICTION_RESTORE_MISMATCH")
    return pack(direction, period, prediction["dataAsOf"], factors, baseline=baseline)
