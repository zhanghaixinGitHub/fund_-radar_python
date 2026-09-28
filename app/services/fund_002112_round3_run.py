"""第三轮唯一运行与永久拟合账本。所有写入限定于 round3-runs。

主拟合/复现各占唯一槽位，先持久化账本再进入分类器。检查点缺失即停止，不能
更换运行编号重试。每个模型在独立子进程重新加载推理后才生成有效检查点。
"""

import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime

import joblib

from app.services import fund_002112_round3_data as data
from app.services import fund_002112_round3_model as model

MAX_FITS = 24
SLOTS = tuple(
    f"{stage}-{variant}-{role}"
    for stage in (*data.FOLDS, "FULL")
    for variant in model.VARIANTS
    for role in ("main", "replay")
)


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": 1,
        "libraries": {
            n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")
        },
    }


def code_manifest():
    paths = [
        *data.PROJECT.glob("app/services/fund_002112_round3*.py"),
        data.PROJECT / "scripts/fund_002112_round3.py",
        data.PROJECT / "app/services/direction_1d_protocol.py",
        data.PROJECT / "app/services/trading_calendar.py",
        data.PROJECT / "requirements.txt",
    ]
    paths.extend(data.PROJECT.glob("app/data/calendars/*.json"))
    return {str(p.relative_to(data.PROJECT)): data.file_hash(p) for p in sorted(paths)}


@contextmanager
def lock():
    data.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (data.RUN_ROOT / ".execution.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def directory(run_id):
    if not re.fullmatch(r"002112-r3-[a-f0-9]{24}", run_id):
        raise ValueError("INVALID_RUN_ID")
    p = (data.RUN_ROOT / run_id).resolve()
    if p.parent != data.RUN_ROOT.resolve():
        raise ValueError("WRITE_PATH_OUTSIDE_ROUND3")
    return p


def verify_sources(sources):
    for path, expected in sources["files"].items():
        if data.file_hash(path) != expected:
            raise ValueError("SOURCE_FILE_CHANGED:" + path)
    for r in sources["receipts"]:
        if r.get("expires_at") and datetime.fromisoformat(r["expires_at"]) <= datetime.now(data.ZONE):
            raise ValueError("SOURCE_EXPIRED")


def preflight():
    """首次完整重建后冻结代码、环境、输入、来源和原考试身份；无真实拟合。"""
    with lock():
        pointer = data.RUN_ROOT / "run-binding.json"
        if pointer.exists():
            p = directory(data.read(pointer)["run_id"])
            load_frozen(p)
            return p
        if any(data.RUN_ROOT.glob("002112-r3-*/attempts/*.json")):
            raise ValueError("EXISTING_ATTEMPTS_CANNOT_REBIND")
        built, folds, eligibility, sources = data.construct(lambda s: print(s, flush=True))
        authorization = data.read(data.RUN_ROOT / "source-authorization.json")
        rights = authorization["metadata"]
        if not rights["enabled"] or not {"fund_nav", "daily", "index_daily"}.issubset(rights["authorized_api_names"]):
            raise ValueError("SOURCE_SCOPE_UNAVAILABLE")
        sources["authorization"] = authorization
        sources["files"][str(data.RUN_ROOT / "source-authorization.json")] = data.file_hash(
            data.RUN_ROOT / "source-authorization.json"
        )
        spec = {
            "protocol": "FUND_002112_ROUND3_PRIOR_PUBLIC_V1",
            "cohort": list(data.COHORT),
            "features": list(data.FEATURES),
            "variants": model.VARIANTS,
            "logistic": model.LOGISTIC,
            "tree": model.TREE,
            "class_order": list(data.CLASSES),
            "tie_order": list(data.TIE_ORDER),
            "maximum_fits": MAX_FITS,
            "slots": list(SLOTS),
            "code": code_manifest(),
            "environment": environment(),
            "inputs_hash": data.digest(built),
            "sources_hash": data.digest(sources),
            "folds_hash": data.digest(folds),
            "eligibility_hash": data.digest(eligibility),
            "old_fits": 34,
            "selection": "strict_total_gt_N7_L20_constants_all_classes_ge_N7_L20_2_quarters_ge_L20",
            "no_adoption": True,
            "no_future_observation": True,
        }
        run_id = "002112-r3-" + data.digest(spec)[:24]
        p = directory(run_id)
        for name, v in (
            ("inputs", built),
            ("sources", sources),
            ("folds", folds),
            ("eligibility", eligibility),
            ("protection-before", data.read(data.RUN_ROOT / "protection-start.json")),
        ):
            data.write_once(p / (name + ".json"), v)
        data.write_once(p / "protocol.json", spec)
        data.write_once(p / "created.json", {"at": data.now(), "fits": 0})
        data.write_once(pointer, {"run_id": run_id, "protocol_hash": data.digest(spec)})
        return p


def load_frozen(p, *, check_sources=True):
    spec = data.read(p / "protocol.json")
    if spec["code"] != code_manifest() or spec["environment"] != environment():
        raise ValueError("FROZEN_CODE_OR_ENVIRONMENT_CHANGED")
    if spec["tree"] != model.TREE or spec["logistic"] != model.LOGISTIC or spec["slots"] != list(SLOTS):
        raise ValueError("FROZEN_RECIPE_CHANGED")
    values = {n: data.read(p / (n + ".json")) for n in ("inputs", "sources", "folds", "eligibility")}
    for n, v in values.items():
        if data.digest(v) != spec[n + "_hash"]:
            raise ValueError("FROZEN_ARTIFACT_CHANGED:" + n)
    if check_sources:
        verify_sources(values["sources"])
    return values


def fold_data(values, stage):
    fold = next(f for f in values["folds"] if f["name"] == stage)
    ids = {tuple(r) for r in fold["train_ids"]}
    train = [r for r in values["inputs"]["train"] if (r["fund_code"], r["target"]) in ids]
    exams = [
        r
        for r in values["inputs"]["development" if stage == "FULL" else "train"]
        if r["fund_code"] == data.FUND and r["target"] in fold["expected_dates"]
    ]
    if data.digest(train) != fold["train_hash"] or data.digest(exams) != fold["exam_hash"]:
        raise ValueError("FOLD_IDENTITY_CHANGED")
    if data.weights(train) != fold["weights"] or any(not data.eligible_at(r, fold["start"]) for r in train):
        raise ValueError("FOLD_WEIGHTS_OR_MATURITY_CHANGED")
    return train, exams, fold["weights"]


def attempts(p):
    return sorted((p / "attempts").glob("*.json"))


def reserve(p, slot):
    """排他写入是不可退还的预算承诺；中断后没有完整检查点绝不重训。"""
    if slot not in SLOTS:
        raise ValueError("UNKNOWN_SLOT")
    path = p / "attempts" / (slot + ".json")
    if path.exists():
        raise ValueError("SLOT_ALREADY_CONSUMED")
    if len(attempts(p)) >= MAX_FITS:
        raise ValueError("FIT_BUDGET_EXHAUSTED")
    data.write_once(
        path,
        {
            "slot": slot,
            "at": data.now(),
            "status": "PERMANENTLY_RESERVED",
            "protocol_hash": data.digest(data.read(p / "protocol.json")),
        },
    )


def checkpoint(p, slot):
    value = data.read(p / "checkpoints" / (slot + ".json"))
    if value["slot"] != slot or value["attempt_hash"] != data.file_hash(p / "attempts" / (slot + ".json")):
        raise ValueError("CHECKPOINT_ATTEMPT_MISMATCH")
    for rel, sha in value["files"].items():
        path = (p / rel).resolve()
        if not path.is_relative_to(p.resolve()) or data.file_hash(path) != sha:
            raise ValueError("CHECKPOINT_FILE_CHANGED")
    return value


def child(p, command, slot):
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-m",
            "scripts.fund_002112_round3",
            command,
            "--run-id",
            p.name,
            "--slot",
            slot,
        ],
        cwd=data.PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        timeout=180,
    )
    if result.returncode:
        # 本地数值错误不含请求凭据。仅保存有界末尾，避免外部异常全文扩散。
        raise ValueError("CHILD_FAILED:" + command + ":" + result.stderr[-1500:])


def fit_worker(p, slot):
    """内部子进程仅能消费预留但从未开始的槽位，不能直接绕过主编排运行。"""
    if slot not in SLOTS or not (p / "attempts" / (slot + ".json")).exists():
        raise ValueError("UNRESERVED_FIT")
    started = p / "started" / (slot + ".json")
    if started.exists():
        raise ValueError("FIT_ALREADY_STARTED_NO_RETRY")
    values = load_frozen(p, check_sources=False)
    if not values["eligibility"]["passed"]:
        raise ValueError("DATA_GATE_FAILED")
    stage, variant, _ = slot.split("-")
    if stage == "FULL" and not data.read(p / "historical-decision.json")["passed"]:
        raise ValueError("HISTORICAL_GATE_FAILED")
    train, exams, w = fold_data(values, stage)
    data.write_once(started, {"at": data.now(), "slot": slot})
    fitted, predictions = model.fit(train, exams, w, variant)
    model_path = p / "models" / (slot + ".joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("xb") as stream:
        joblib.dump(fitted, stream, compress=0)
        stream.flush()
        os.fsync(stream.fileno())
    data.write_once(p / "predictions" / (slot + ".json"), predictions)
    data.write_once(
        p / "model-manifests" / (slot + ".json"),
        {
            "file_sha256": data.file_hash(model_path),
            "state_hash": model.state_hash(fitted),
            "variant": variant,
            "recipe": fitted["recipe"],
            "features": fitted["features"],
            "classes": fitted["classes"],
            "mean": fitted["scaler"].mean_.tolist(),
            "scale": fitted["scaler"].scale_.tolist(),
            "training_hash": fitted["training_hash"],
            "weights_hash": fitted["weights_hash"],
        },
    )


def restore_worker(p, slot):
    """只加载本轮按固定文件名生成、SHA 已验证的模型；CLI 不接收外部模型路径。"""
    if slot not in SLOTS:
        raise ValueError("UNKNOWN_SLOT")
    values = load_frozen(p, check_sources=False)
    manifest = data.read(p / "model-manifests" / (slot + ".json"))
    path = p / "models" / (slot + ".joblib")
    if data.file_hash(path) != manifest["file_sha256"]:
        raise ValueError("UNTRUSTED_MODEL_FILE")
    fitted = joblib.load(path)
    if model.state_hash(fitted) != manifest["state_hash"]:
        raise ValueError("MODEL_STATE_CHANGED")
    train, exams, _ = fold_data(values, slot.split("-")[0])
    saved = data.read(p / "predictions" / (slot + ".json"))
    delta = {
        s: model.compare_predictions(saved[s], model.predict(fitted, rows))
        for s, rows in (("train", train), ("exam", exams))
    }
    data.write_once(
        p / "restored" / (slot + ".json"),
        {"score_differences": delta, "directions_equal": True, "separate_process": True},
    )


def ensure_slot(p, slot):
    if (p / "checkpoints" / (slot + ".json")).exists():
        return checkpoint(p, slot)
    if (p / "attempts" / (slot + ".json")).exists():
        raise ValueError("INCOMPLETE_CONSUMED_SLOT_TECHNICAL_FAILURE:" + slot)
    reserve(p, slot)
    try:
        child(p, "_fit", slot)
        child(p, "_restore", slot)
        files = [
            f"models/{slot}.joblib",
            f"predictions/{slot}.json",
            f"model-manifests/{slot}.json",
            f"restored/{slot}.json",
            f"started/{slot}.json",
        ]
        value = {
            "slot": slot,
            "completed_at": data.now(),
            "attempt_hash": data.file_hash(p / "attempts" / (slot + ".json")),
            "files": {rel: data.file_hash(p / rel) for rel in files},
        }
        data.write_once(p / "checkpoints" / (slot + ".json"), value)
        return value
    except Exception as exc:
        data.write_once(p / "failures" / (slot + ".json"), {"at": data.now(), "reason": str(exc)})
        raise


def stage_result(p, stage):
    predictions = {}
    proofs = {}
    for v in model.VARIANTS:
        slots = [f"{stage}-{v}-{r}" for r in ("main", "replay")]
        for slot in slots:
            checkpoint(p, slot)
        a, b = [data.read(p / "model-manifests" / (s + ".json")) for s in slots]
        if a["state_hash"] != b["state_hash"]:
            raise ValueError("INDEPENDENT_MODEL_STATE_MISMATCH")
        left, right = [data.read(p / "predictions" / (s + ".json")) for s in slots]
        proofs[v] = {s: model.compare_predictions(left[s], right[s]) for s in ("train", "exam")}
        predictions[v] = left["exam"]
    data.write_once(p / "reproduction" / (stage + ".json"), proofs)
    return predictions


def run(p, *, interrupt_after=None):
    with lock():
        values = load_frozen(p)
        if not values["eligibility"]["passed"]:
            data.write_once(
                p / "historical-decision.json",
                {
                    "passed": False,
                    "status": "NOT_RUN_DATA_GATE_FAILED",
                    "fits": 0,
                    "eligibility_hash": data.digest(values["eligibility"]),
                },
            )
            return
        incomplete = [x.stem for x in attempts(p) if not (p / "checkpoints" / x.name).exists()]
        if incomplete:
            raise ValueError("INCOMPLETE_CONSUMED_SLOT_TECHNICAL_FAILURE:" + ",".join(incomplete))
        for stage in data.FOLDS:
            for v in model.VARIANTS:
                for role in ("main", "replay"):
                    slot = f"{stage}-{v}-{role}"
                    ensure_slot(p, slot)
                    print(f"已保存 {slot}，占用拟合 {len(attempts(p))}/{MAX_FITS}", flush=True)
                    if interrupt_after == slot and not (p / "interruption.json").exists():
                        data.write_once(
                            p / "interruption.json",
                            {"at": data.now(), "slot": slot, "checkpoint_complete": True, "requested_exit_code": 75},
                        )
                        os._exit(75)
        by_variant = {v: [] for v in model.VARIANTS}
        for stage in data.FOLDS:
            for v, rows in stage_result(p, stage).items():
                by_variant[v].extend(rows)
        result = model.comparison(by_variant, historical=True)
        historical = {
            "passed": result["numerical_passed"],
            "checks": {**result["checks"], "reproduction_and_restore": True},
            "status": "PASS" if result["numerical_passed"] else "FAIL",
            "comparison_hash": data.digest(result),
            "fits": 18,
        }
        data.write_once(p / "historical-comparison.json", result)
        data.write_once(p / "historical-decision.json", historical)
        # 决定文件一旦写下不可变。只有这里的通过分支才会创建 FULL 槽位。
        full = None
        if historical["passed"]:
            for v in model.VARIANTS:
                for role in ("main", "replay"):
                    ensure_slot(p, f"FULL-{v}-{role}")
            full = model.comparison(stage_result(p, "FULL"), historical=False)
        data.write_once(
            p / "comparison.json",
            {
                "historical": result,
                "development": full,
                "development_status": "COMPLETED" if full else "NOT_RUN_HISTORICAL_GATE_FAILED",
            },
        )
        data.write_once(
            p / "completion.json",
            {
                "real_fits": len(attempts(p)),
                "status": "COMPLETED",
                "development_run": full is not None,
                "adopted": False,
            },
        )


def verify(p):
    """零拟合复核：全部主模型与复现模型分别重新加载，重算比较和阶段决定。"""
    with lock():
        load_frozen(p)
        before = len(attempts(p))
        for attempt in attempts(p):
            checkpoint(p, attempt.stem)
            child(p, "_restore", attempt.stem)
        if before:
            by_variant = {v: [] for v in model.VARIANTS}
            for stage in data.FOLDS:
                for v, rows in stage_result(p, stage).items():
                    by_variant[v].extend(rows)
            if model.comparison(by_variant, historical=True) != data.read(p / "historical-comparison.json"):
                raise ValueError("RECOMPUTED_COMPARISON_CHANGED")
            stored = data.read(p / "comparison.json")
            if (
                stored["development"]
                and model.comparison(stage_result(p, "FULL"), historical=False) != stored["development"]
            ):
                raise ValueError("RECOMPUTED_FULL_CHANGED")
        if len(attempts(p)) != before:
            raise ValueError("VERIFY_MUST_NOT_FIT")
        return {"verified_attempts": before, "new_fits": 0, "separate_process_restore": True}


def report(p):
    """只用已保存结果写通俗报告；没有运行的阶段绝不填零分。"""
    eligibility = data.read(p / "eligibility.json")
    fits = len(attempts(p))
    lines = [
        "# 002112 第三轮结果",
        "",
        f"本轮真实拟合 {fits} 次；前两轮仍为 34 次，合计 {34 + fits} 次。",
        "",
        "本轮只有一种新增方法 T20；N7/L20 是新日期口径的匹配对照。所有成绩均为开发检查，不是未来效果。",
        "",
        "## 资料与固定日期",
        "",
        "| 阶段 | 训练行 | 独立日期 | 持平独立日期 | 考试日期 | 资料通过 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, f in eligibility["folds"].items():
        g = f["gate"]["pooled"]
        lines.append(
            f"| {name} | {g['rows']} | {g['dates']} | {g['class_dates']['FLAT']} | {f['exam_count']} | "
            f"{f['gate']['passed'] and not f['missing_dates']} |"
        )
    lines.extend(
        [
            "",
            "全包三类的独立日期可能重叠，不能相加。"
            "逐行排除原因与 S/T/U、滞后日数分别见 eligibility.json、inputs.json。",
            "",
            "## 结果",
        ]
    )
    if (p / "comparison.json").exists():
        result = data.read(p / "comparison.json")
        for key, title in (("historical", "2023 年原 161 天"), ("development", "2024 年原 230 天")):
            lines.extend(["", "### " + title, ""])
            c = result[key]
            if c is None:
                lines.append("未执行：2023 年预定筛选未通过，按约定停止，未为查看 2024 年成绩额外训练。")
                continue
            lines.extend(
                [
                    "| 配置 | 正确/总天数 | 上涨正确/实际 | 持平正确/实际 | 下跌正确/实际 |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for name, score in {**c["models"], **{"始终" + k: v for k, v in c["constants"].items()}}.items():
                cols = [f"{score['class_correct'][k]}/{score['actual'][k]}" for k in ("UP", "FLAT", "DOWN")]
                lines.append(f"| {name} | {score['correct']}/{score['days']} | " + " | ".join(cols) + " |")
            for control, pair in c["pairs"].items():
                lines.extend(
                    [
                        "",
                        f"T20 相比 {control}：新增猜对 {pair['gained']} 天，丢失 {pair['lost']} 天，"
                        f"净变化 {pair['net']:+d} 天。",
                    ]
                )
            lines.extend(["", "| 季度 | N7 | L20 | T20 |", "| --- | ---: | ---: | ---: |"])
            for q, scores in c["quarters"].items():
                lines.append(f"| {q} | " + " | ".join(str(scores[v]["correct"]) for v in model.VARIANTS) + " |")
            failed = [k for k, passed in c["checks"].items() if not passed]
            lines.extend(["", "预定规则：" + ("通过。" if not failed else "未通过：" + "、".join(failed) + "。")])
    else:
        lines.extend(["", "尚无模型效果成绩。资料未通过或发生技术中断，详见 eligibility.json、failures 和账本。"])
    lines.extend(
        [
            "",
            "## 证据与限制",
            "",
            "- 每日方向、三类分数及真实答案：predictions/；配对得失、季度、混淆表：comparison.json。",
            "- 永久拟合账本：attempts/；保存检查点：checkpoints/；"
            "独立训练复现：reproduction/；独立进程重算：restored/。",
            "- 历史决定不可改写：historical-decision.json。真实中断证据：interruption.json（仅在实际执行后存在）。",
            "- 新闻仅整理原有 12 份卡片、5 家公司、5 个示例目标日，未拟合、未外呼，不能判断新闻是否提升预测。",
            "- 2023 年检查持平只有 8 天，2024 年原清单只有 2 天，缺少持平稳定识别及未来效果证据。",
            "- 未登记或采用、未启动未来观察、未补历史方向、未写数据库、未改页面或同步链。",
            "",
            "失败仅针对本轮固定做法；没有运行的阶段不能称效果差，也不能据此推断新闻或其他算法无效。",
            "",
        ]
    )
    text = "\n".join(lines)
    path = p / "report.md"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("IMMUTABLE_REPORT_CHANGED")
    if not path.exists():
        path.write_text(text, encoding="utf-8", newline="\n")
    return str(path)


def status(p):
    return {
        "run_id": p.name,
        "reserved_fits": len(attempts(p)),
        "started_fits": len(list((p / "started").glob("*.json"))),
        "completed_fits": len(list((p / "checkpoints").glob("*.json"))),
        "maximum_fits": MAX_FITS,
        "historical": data.read(p / "historical-decision.json") if (p / "historical-decision.json").exists() else None,
    }
