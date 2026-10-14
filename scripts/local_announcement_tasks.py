"""本地公告阅读的可审计任务拆分；不裁剪原文，不让模型自改答案。

只拆明确语法模板，未识别的问题原样保留。子问题默认分别调用同一模型；可选的当前总担保余额按原文规则提取。结果按原题
归组，逐字保留各次答复及引用。此试验入口不被生产预测模块引用。
"""

import argparse
import json
import re
from pathlib import Path

from scripts.local_announcement_reader import call_local
from scripts.local_announcement_structured_reader import read_document

VERSION = "local-announcement-tasks-v3"


def plan_question(question: str, *, focus_details: bool = False, quote_balance: bool = False) -> dict:
    """只做不含业务答案的确定性拆分，无法保证主语继承时保持原题。"""
    q = question.strip()
    rules = [
        (r"(.+价格)如何调整，触发调整的具体事项是什么[？?]", "price_change"),
        (r"(.+)是否通过，其(.+)是多少[？?]", "vote_result"),
        (r"(.+)调整后是多少，(.+)是多少，两者是否相同[？?]", "limit_vs_balance"),
        (r"哪些(.+)属于(.+)，哪些(?:\1)?属于(.+)，各自通过门槛是什么[？?]", "decision_types"),
        (r"(.+考核的)(.+)、(.+)及(.+)要求是什么[？?](.+)", "assessment_metrics"),
    ]
    if focus_details:
        rules.append((r"(.+?)(?:需|须)同时满足哪些条件[？?]", "conditions_with_exceptions"))
    for pattern, rule in rules:
        match = re.fullmatch(pattern, q)
        if not match:
            continue
        fields = match.groups()
        if rule == "price_change":
            subject = fields[0]
            parts = [
                (
                    f"请依据公告的实际数字或数值计算式，回答本次{subject}调整前的实际价格是多少？不要用符号变量代替。"
                    if focus_details
                    else f"本次{subject}调整前是多少？"
                ),
                f"本次{subject}调整后是多少？",
                f"触发本次{subject}调整的具体事项是什么？",
            ]
        elif rule == "vote_result":
            subject, field = fields
            parts = [f"{subject}是否通过？", f"{subject}的{field}是多少？"]
        elif rule == "limit_vs_balance":
            limit, balance = fields
            parts = [f"{limit}调整后是多少？", f"{balance}是多少？", f"调整后的{limit}与{balance}是否相同？"]
            if focus_details:
                parts[1] += "只答本项金额，逐字保留原文金额前的限定词，不展开其他金额。"
                if quote_balance:
                    # 只供 ids_first 候选试验；沿用旧模式时保持已验证过的原问题。
                    parts[1] = f"请逐字引用说明{balance}的完整原句，保留金额前的限定词和单位，不列举其他金额。"
                parts[2] = f"{limit}与{balance}属于同一种口径吗？仅说明区别，不重新计算或列举分项。"
        elif rule == "conditions_with_exceptions":
            subject = fields[0]
            parts = [
                f"请逐字列出{subject}需同时满足的条件，保留括号中的内容。",
                f"关于{subject}，原文与这些条件相连的括号例外、另行执行规定是什么？逐字引用相关原句；若没有则明确说明未见。",
            ]
        elif rule == "decision_types":
            subject, first, second = fields
            parts = [
                f"哪些{subject}属于{first}？",
                f"哪些{subject}属于{second}？",
                f"{first}的通过门槛是什么？",
                f"{second}的通过门槛是什么？",
            ]
        else:
            prefix, first, second, third, tail = fields
            parts = [f"{prefix}{field}要求是什么？" for field in (first, second, third)]
            # 比较基年的尾问保留原语义，并承接同一考核主语，避免不指明年份。
            parts.append(prefix + tail)
        return {"original_question": q, "rule": rule, "questions": parts}
    return {"original_question": q, "rule": "unchanged", "questions": [q]}


def plan_tasks(
    questions: list[str], *, focus_details: bool = False, quote_balance: bool = False
) -> tuple[list[dict], list[dict]]:
    """普通题保持题组语境，只拆有明确主语继承规则的复杂题。

    不能为解释单题中的指代再附一整组待答问题，否则小模型可能答错目标题。
    bindings 是原题号、子题号到本次答案的映射，不依赖模型生成内容或正确答案。
    """
    if not isinstance(questions, list) or not 1 <= len(questions) <= 8:
        raise ValueError("每篇需要1至8个问题。")
    plans = [plan_question(q, focus_details=focus_details, quote_balance=quote_balance) for q in questions]
    ordinary = [(n, p) for n, p in enumerate(plans, 1) if p["rule"] == "unchanged"]
    tasks = []
    if ordinary:
        tasks.append(
            {
                "id": "batch",
                "questions": [p["original_question"] for _, p in ordinary],
                "bindings": [[n, 1] for n, _ in ordinary],
            }
        )
    for n, plan in enumerate(plans, 1):
        if plan["rule"] == "unchanged":
            continue
        tasks.extend(
            {"id": f"q{n}-s{s}", "questions": [q], "bindings": [[n, s]]} for s, q in enumerate(plan["questions"], 1)
        )
    if len(tasks) > 32:
        raise ValueError("拆分后的本地请求超过32次上限。")
    return plans, tasks


def read_tasks(
    title: str,
    packet: dict,
    questions: list[str],
    *,
    transport=call_local,
    citation_mode: str = "ids",
    focus_details: bool = False,
    focus_balance: bool = False,
) -> dict:
    """复杂题独立读取，普通题成组；失败保留，不追加尝试或付费回退。"""
    plans, tasks = plan_tasks(questions, focus_details=focus_details, quote_balance=citation_mode == "ids_first")
    # 调用前一次性验证全部输入，避免后面的非法问题导致前面的调用白做。
    from scripts.local_announcement_balance import balance_evidence, extract_balance, handles_balance
    from scripts.local_announcement_structured_reader import structured_request

    for task in tasks:
        if focus_balance and handles_balance(task["questions"]):
            balance_evidence(packet)
        else:
            structured_request(title, packet, task["questions"], citation_mode=citation_mode)
    runs, mapped = [], {}
    for task in tasks:
        if focus_balance and handles_balance(task["questions"]):
            run = extract_balance(title, packet, task["questions"])
        else:
            run = read_document(title, packet, task["questions"], transport=transport, citation_mode=citation_mode)
        runs.append({"task": task, **run})
        for q, binding, answer in zip(task["questions"], task["bindings"], run["answers"], strict=True):
            mapped[tuple(binding)] = {"question": q, "run_number": len(runs), **answer}
    results = []
    for number, plan in enumerate(plans, 1):
        parts = [mapped[(number, s)] for s in range(1, len(plan["questions"]) + 1)]
        answers = parts
        issues = [f"S{i}:{issue}" for i, answer in enumerate(answers, 1) for issue in answer["issues"]]
        joined = "\n".join(f"{q} {a['answer']}" for q, a in zip(plan["questions"], answers, strict=True))
        results.append(
            {
                "question_number": number,
                "plan": plan,
                "parts": parts,
                "answer": joined,
                "issues": issues,
                "safe_answer": joined if not issues else None,
                "review_state": "needs_review" if issues else "checks_passed",
                "semantic_verified": False,
            }
        )
    return {
        "version": VERSION,
        "answers": results,
        "runs": runs,
        "semantic_verified": False,
        "calls": sum(len(run["attempts"]) for run in runs),
        "source_sha256": packet["source_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--citation-mode", choices=("ids", "quotes", "ids_first"), default="ids")
    parser.add_argument("--focus-details", action="store_true", help="候选细化提问：价格实数、金额限定词和条件例外")
    parser.add_argument(
        "--focus-balance", action="store_true", help="当前总担保余额用原文规则提取，保留限定词；歧义时复核"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("结果已存在，请使用新路径保留原答复。")
    case = json.loads(args.input.read_text(encoding="utf-8"))
    result = read_tasks(
        case["title"],
        case["document"],
        case["questions"],
        citation_mode=args.citation_mode,
        focus_details=args.focus_details,
        focus_balance=args.focus_balance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "calls": result["calls"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
