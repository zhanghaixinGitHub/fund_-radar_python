"""线性/浅层树 × 原特征/成交活跃度的固定四组实验，无新增来源和自动调参。"""

import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from time import perf_counter
from uuid import UUID

from app.services import direction_volume_data as data
from app.services.direction_algorithm_models import predict_model
from app.services.direction_linear_analysis import accuracy_delta, evaluate
from app.services.direction_linear_models import validate_job
from app.services.direction_linear_protocol import (
    ALGORITHM_BRANCHES as BRANCHES,
)
from app.services.direction_linear_protocol import (
    ALGORITHM_VERSION as VERSION,
)
from app.services.direction_linear_protocol import (
    BASELINES,
    evaluation_asof,
    planned_dates,
    study_rules,
)
from app.services.direction_market_runner import NUMERIC_FIELDS, build_window, prior, window_predictions
from app.services.direction_training_artifacts import (
    ROOT,
    digest,
    file_hash,
    new_folder,
    now,
    read_json,
    read_jsonl,
    read_seal,
    run_folder,
    seal,
    write_json,
    write_jsonl,
)
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_evaluation import complete_blocks, grouped_metrics, paired_interval
from app.services.direction_training_protocol import TREE, runtime, source_fingerprint
from app.services.direction_volume_runner import STAGES as VOLUME_STAGES
from app.services.trading_calendar import load_calendar

SOURCE_RUN = "9e89b98a-898f-4518-8e6b-e9f2ca9cfbe1"
SOURCE_HASH = "b2f272a7c5062aba54f53348f5dc80358dc78a6cc9fd37f16d137cdcb4db92ba"
STAGES = ("frozen", "prepared", "predicted", "scored", "complete")
PAIRS = {"B_vs_A": (1, 0), "C_vs_A": (2, 0), "D_vs_B": (3, 1), "D_vs_C": (3, 2)}


def check_source(folder):
    """验证旧包封存链，允许旧代码版本存在；不在拟合前解析EXAM标签。"""
    previous = None
    manifests = {}
    for stage in VOLUME_STAGES:
        current = read_seal(folder, f"volume-{stage}.json")
        if previous and current[f"{previous[0]}_hash"] != previous[1]["manifest_hash"]:
            raise ValueError("ALGORITHM_SOURCE_CHAIN")
        previous = stage, current
        manifests[stage] = current
    if current["manifest_hash"] != SOURCE_HASH:
        raise ValueError("ALGORITHM_SOURCE_CHANGED")
    return manifests


def specification(old):
    """只保留有效契约，避免继承上一轮三组预算或交互特征的说明。"""
    protocol = {
        k: deepcopy(old[k])
        for k in (
            "windows",
            "funds",
            "start",
            "end",
            "minimum",
            "minimum_windows",
            "minimum_coverage",
            "bootstrap",
            "budget",
            "selection",
            "independent",
            "features",
            "logistic",
            "threshold",
            "mapping_hash",
            "activity_formula",
            "volume_information_rule",
        )
    }
    protocol.update(
        version=VERSION,
        branches=list(BRANCHES),
        baselines=list(BASELINES),
        tree=dict(TREE),
        feature_indices={b: list(v) for b, v in study_rules(VERSION)[1].items()},
        added_feature="amount_mean_5d_over_mean_20d",
        pairs=PAIRS,
        fit_budget={"main_maximum": 32, "replay_maximum": 32},
        training_scope="POOLED_EQUAL_FUND_IDENTICAL_FIT_AND_EXAM_KEYS_ALL_FOUR_BRANCHES",
        source_run=SOURCE_RUN,
        source_complete_hash=SOURCE_HASH,
        source_code_hash=source_fingerprint(),
        runtime=runtime(),
        dependency_hash=file_hash(ROOT / "requirements.txt"),
        calendar_hash=load_calendar().content_hash,
        network_calls=0,
        database_writes=0,
        parameter_search=False,
        calibration="NONE",
        snapshot_search="ONE_FIXED_32_FIT_BATCH_AND_ONE_32_FIT_REPLAY_NO_ADAPTIVE_CHANGES",
        diagnostic_scope="FIT_IN_SAMPLE_VS_OBSERVED_EXAM_NOT_ADDITIONAL_SELECTION_GATE",
        independent_test=False,
        model_released=False,
    )
    return protocol


def freeze():
    source, folder = run_folder(UUID(SOURCE_RUN)), new_folder()
    manifests, files = check_source(source), {}
    # 把旧证据完整复制进新包，来源发生修改时不能仅凭新摘要绕过已冻结的旧完整摘要。
    names = {f"volume-{s}.json" for s in VOLUME_STAGES}
    names.update(n for m in manifests.values() for n in m["files"])
    for name in sorted(names):
        shutil.copyfile(source / name, folder / name)
        files[name] = file_hash(folder / name)
    protocol = specification(read_json(source / "volume-plan.json"))
    files["algorithm-plan.json"] = write_json(folder / "algorithm-plan.json", protocol)
    frozen = seal(folder, "algorithm-frozen.json", files, status="FIXED_FOUR_ARMS_BEFORE_ANY_FIT")
    return {"run": folder.name.removeprefix("direction-training-"), "folder": str(folder), **frozen}


def load_plan(folder):
    frozen = read_seal(folder, "algorithm-frozen.json")
    check_source(folder)
    protocol = read_json(folder / "algorithm-plan.json")
    if protocol != json.loads(json.dumps(specification(read_json(folder / "volume-plan.json")))):
        raise ValueError("ALGORITHM_FROZEN_CODE_OR_PROTOCOL_CHANGED")
    prior()
    return protocol, frozen


def rebuild_window(folder, window):
    """从原NAV题目和冻结成交额重算旧输入；再仅替换算法，不改变样本和标签。"""
    name, source = window["name"], prior()
    old, coverage = build_window(
        read_json(source / f"linear-prepared-{name}.json"),
        read_json(folder / "volume-mapping.json"),
        read_json(folder / "volume-data.json"),
        read_json(folder / "volume-plan.json"),
        read_json(source / "linear-coverage.json")[name],
        augment_inputs=data.augment_inputs,
    )
    if (
        old != read_json(folder / f"volume-prepared-{name}.json")
        or coverage != read_json(folder / "volume-coverage.json")[name]
    ):
        raise ValueError("ALGORITHM_SOURCE_INPUT_REBUILD")
    if not coverage["fit_unchanged"] or not coverage["exam_unchanged"]:
        raise ValueError("ALGORITHM_SOURCE_POPULATION_CHANGED")
    jobs = {}
    for i, branch in enumerate(BRANCHES):
        jobs[branch] = {**deepcopy(old["jobs"][BRANCHES[i % 2]]), "version": VERSION, "branch": branch}
        if coverage["status"] == "READY":
            validate_job(jobs[branch])
    return {"window": window, "planned": old["planned"], "jobs": jobs}, coverage


def prepare(folder):
    protocol, frozen = load_plan(folder)
    files, coverage = {}, {}
    for window in protocol["windows"]:
        bundle, coverage[window["name"]] = rebuild_window(folder, window)
        name = f"algorithm-prepared-{window['name']}.json"
        files[name] = write_json(folder / name, bundle)
    files["algorithm-coverage.json"] = write_json(folder / "algorithm-coverage.json", coverage)
    return seal(
        folder, "algorithm-prepared.json", files, frozen_hash=frozen["manifest_hash"], worker_exam_answers=False
    )


def control_parity(folder, name, outputs):
    old = read_json(folder / f"volume-models-{name}.json")
    result = {}
    for branch in BRANCHES[:2]:
        if outputs[branch]["status"] != "PREDICTED":
            result[branch] = {"status": "NOT_COMPARABLE_JOB_NOT_PREDICTED"}
            continue
        a, b = old[branch]["models"]["POOLED"], outputs[branch]["models"]["POOLED"]
        if (
            any(a[k] != b[k] for k in (*NUMERIC_FIELDS, "C", "penalty", "solver_iterations"))
            or old[branch]["scores"] != outputs[branch]["scores"]
        ):
            raise ValueError("ALGORITHM_CONTROL_NUMERIC_CHANGED")
        result[branch] = {
            "status": "IDENTICAL_MODEL_NUMBERS_AND_SCORES",
            "scores": len(outputs[branch]["scores"]),
            "maximum_difference": 0,
        }
    return result


def predict(folder):
    protocol, _ = load_plan(folder)
    prepared = read_seal(folder, "algorithm-prepared.json")
    coverage = read_json(folder / "algorithm-coverage.json")
    write_json(folder / "algorithm-predict-started.json", {"started_at": now(), "maximum_fits": 32})
    from app.services.direction_training_process import run_process

    files, execution, parity, fits = {}, {}, {}, 0
    start = perf_counter()
    for window in protocol["windows"]:
        name, outputs = window["name"], {}
        bundle = read_json(folder / f"algorithm-prepared-{name}.json")
        for branch in BRANCHES:
            if coverage[name]["status"] != "READY":
                result = {"status": "INSUFFICIENT_DATA"}
            elif fits >= 32 or perf_counter() - start > protocol["budget"]["total_seconds"]:
                result = {"status": "FAILED", "reason": "ALGORITHM_TRAINING_BUDGET"}
            else:
                fits += 1
                result = run_process(
                    bundle["jobs"][branch],
                    seconds=protocol["budget"]["candidate_seconds"],
                    memory_bytes=protocol["budget"]["process_memory_bytes"],
                    command=[sys.executable, "-m", "scripts.direction_algorithm_worker"],
                )
            outputs[branch] = result
            print(json.dumps({"window": name, "branch": branch, "status": result["status"]}), flush=True)
        parity[name] = control_parity(folder, name, outputs)
        for filename, value, writer in (
            (f"algorithm-models-{name}.json", outputs, write_json),
            (
                f"algorithm-predictions-{name}.jsonl",
                window_predictions(bundle, outputs, branches=BRANCHES),
                write_jsonl,
            ),
        ):
            files[filename] = writer(folder / filename, value)
        execution[name] = {b: {k: v for k, v in o.items() if k not in ("scores", "models")} for b, o in outputs.items()}
    for name, value in (("algorithm-execution.json", execution), ("algorithm-control-parity.json", parity)):
        files[name] = write_json(folder / name, value)
    files["algorithm-predict-started.json"] = file_hash(folder / "algorithm-predict-started.json")
    return seal(
        folder,
        "algorithm-predicted.json",
        files,
        prepared_hash=prepared["manifest_hash"],
        model_fit_count=fits,
        status="PREDICTIONS_SEALED_BEFORE_EXAM_ANSWERS",
    )


def all_predictions(folder, protocol):
    return [r for w in protocol["windows"] for r in read_jsonl(folder / f"algorithm-predictions-{w['name']}.jsonl")]


def diagnostics(folder, protocol, predictions, answers):
    """C/A和D/B检查换算法，D/C检查树上的成交额增益；FIT只作记忆差距诊断。"""
    answer_map = {(a["window"], a["sample_key"]): a["answer"] for a in answers}
    windows = {w["name"]: w for w in protocol["windows"]}
    rows = {b: {} for b in BRANCHES}
    for p in predictions:
        a = answer_map[(p["window"], p["sample_key"])]
        if (
            p["branch"] in rows
            and p["score"] is not None
            and a
            and a["available_at"] <= evaluation_asof(windows[p["window"]])
        ):
            rows[p["branch"]][(p["window"], p["sample_key"])] = {
                **p,
                "y": a["y"],
                "correct": int(p["predicted_up"] == a["y"]),
            }
    common = set.intersection(*(set(v) for v in rows.values()))
    blocks, _ = complete_blocks(
        protocol, common, planned_by_window={n: planned_dates(VERSION, w) for n, w in windows.items()}
    )
    comparisons, gaps = {}, {}
    for pair, (i, j) in PAIRS.items():
        left, right = rows[BRANCHES[i]], rows[BRANCHES[j]]
        comparisons[pair] = {
            "left": BRANCHES[i],
            "right": BRANCHES[j],
            "time_blocks": paired_interval(protocol, blocks, left, right),
            "full_accuracy_delta": accuracy_delta(
                grouped_metrics([left[k] for k in sorted(common)], FUNDS),
                grouped_metrics([right[k] for k in sorted(common)], FUNDS),
            ),
            "wrong_to_right": sum(not right[k]["correct"] and left[k]["correct"] for k in common),
            "right_to_wrong": sum(right[k]["correct"] and not left[k]["correct"] for k in common),
        }
    for name in windows:
        bundle = read_json(folder / f"algorithm-prepared-{name}.json")
        outputs = read_json(folder / f"algorithm-models-{name}.json")
        gaps[name] = {}
        for b, output in outputs.items():
            if output["status"] != "PREDICTED":
                gaps[name][b] = {"status": output["status"]}
                continue
            fit, _ = validate_job(bundle["jobs"][b])
            scores = predict_model(output["models"]["POOLED"], [i for i, _ in fit])
            fit_metrics = grouped_metrics(
                [{"fund": i.fund, "y": a.y, "score": s} for (i, a), s in zip(fit, scores, strict=True)], FUNDS
            )
            exam_metrics = grouped_metrics([r for (w, _), r in rows[b].items() if w == name], FUNDS)
            gaps[name][b] = {
                "FIT": fit_metrics,
                "EXAM": exam_metrics,
                "accuracy_gap_fit_minus_exam": accuracy_delta(fit_metrics, exam_metrics),
            }
    return {
        "same_questions": len(common),
        "comparisons": comparisons,
        "train_exam": gaps,
        "fit_in_sample": True,
        "exam_previously_observed": True,
        "independent_test": False,
        "diagnostic_only_not_new_selection_gate": True,
    }


def score(folder):
    protocol, _ = load_plan(folder)
    predicted = read_seal(folder, "algorithm-predicted.json")
    predictions = all_predictions(folder, protocol)
    answers = read_jsonl(folder / "volume-answers.jsonl")
    result = evaluate(protocol, predictions, answers)
    files = {
        "algorithm-metrics.json": write_json(folder / "algorithm-metrics.json", result),
        "algorithm-diagnostics.json": write_json(
            folder / "algorithm-diagnostics.json", diagnostics(folder, protocol, predictions, answers)
        ),
    }
    return seal(
        folder,
        "algorithm-scored.json",
        files,
        predicted_hash=predicted["manifest_hash"],
        selected_candidate=result["selected_candidate"],
        independent_test=False,
        model_released=False,
    )


def decision(result):
    return {
        "version": VERSION,
        "selected_candidate": result["selected_candidate"],
        "candidate_status": result["candidate_status"],
        "current_snapshot_search": "CLOSED",
        "automatic_followup_training": False,
        "model_released": False,
        "independent_test_run": False,
        "independent_status": "AWAITING_INDEPENDENCE_AND_FIRST_VERSION_EVIDENCE"
        if result["selected_candidate"]
        else "NOT_RUN_NO_QUALIFIED_CANDIDATE",
    }


def finalize(folder):
    protocol, _ = load_plan(folder)
    scored, result = read_seal(folder, "algorithm-scored.json"), read_json(folder / "algorithm-metrics.json")
    files = {"algorithm-decision.json": write_json(folder / "algorithm-decision.json", decision(result))}
    if result["selected_candidate"]:
        branch = result["selected_candidate"]
        files["algorithm-frozen-candidate.json"] = write_json(
            folder / "algorithm-frozen-candidate.json",
            {
                "branch": branch,
                "protocol_hash": digest(protocol),
                "scored_hash": scored["manifest_hash"],
                "models": {
                    w["name"]: read_json(folder / f"algorithm-models-{w['name']}.json")[branch]["models"]
                    for w in protocol["windows"]
                },
                "release_authorized": False,
            },
        )
    return seal(
        folder,
        "algorithm-complete.json",
        files,
        scored_hash=scored["manifest_hash"],
        status="FINITE_BATCH_COMPLETE",
        model_released=False,
        independent_test=False,
    )


def verify(folder):
    """离线重建所有输入、保存模型分数与评价，零拟合；同时检查模型绑定的FIT摘要。"""
    protocol, _ = load_plan(folder)
    stages = {s: read_seal(folder, f"algorithm-{s}.json") for s in STAGES}
    for left, right in zip(STAGES[:-1], STAGES[1:], strict=True):
        if stages[right][f"{left}_hash"] != stages[left]["manifest_hash"]:
            raise ValueError("ALGORITHM_STAGE_CHAIN")
    activity, prices = read_json(folder / "volume-data.json"), read_json(folder / "volume-price-reference.json")
    for code in activity["amounts"]:
        raw = [r for year in data.YEARS for r in read_json(folder / f"volume-response-{code}-{year}.json")["rows"]]
        amounts, cov = data.validate_rows(raw, prices[code])
        if amounts != activity["amounts"][code] or any(activity["coverage"][code][k] != v for k, v in cov.items()):
            raise ValueError("ALGORITHM_RAW_REBUILD")
    coverage, parity, fits = {}, {}, 0
    for window in protocol["windows"]:
        name = window["name"]
        bundle, coverage[name] = rebuild_window(folder, window)
        if bundle != read_json(folder / f"algorithm-prepared-{name}.json"):
            raise ValueError("ALGORITHM_INPUT_REBUILD")
        outputs = read_json(folder / f"algorithm-models-{name}.json")
        if set(outputs) != set(BRANCHES):
            raise ValueError("ALGORITHM_OUTPUT_BRANCHES")
        for branch, output in outputs.items():
            if output["status"] == "PREDICTED":
                fit, exam = validate_job(bundle["jobs"][branch])
                model = output["models"]["POOLED"]
                if (
                    model["version"] != VERSION
                    or model["branch"] != branch
                    or model["fit_end"] != window["fit_end"]
                    or model["train_counts"] != dict(Counter(i.fund for i, _ in fit))
                    or model["train_hash"] != digest(bundle["jobs"][branch]["fit"])
                ):
                    raise ValueError("ALGORITHM_MODEL_TRAIN_BINDING")
                if predict_model(model, exam) != output["scores"] or output["model_fit_count"] != 1:
                    raise ValueError("ALGORITHM_MODEL_SCORE_REBUILD")
                fits += 1
        if window_predictions(bundle, outputs, branches=BRANCHES) != read_jsonl(
            folder / f"algorithm-predictions-{name}.jsonl"
        ):
            raise ValueError("ALGORITHM_PREDICTIONS_REBUILD")
        parity[name] = control_parity(folder, name, outputs)
    if fits > stages["predicted"]["model_fit_count"] or stages["predicted"]["model_fit_count"] > 32:
        raise ValueError("ALGORITHM_FIT_BUDGET")
    if coverage != read_json(folder / "algorithm-coverage.json") or parity != read_json(
        folder / "algorithm-control-parity.json"
    ):
        raise ValueError("ALGORITHM_COVERAGE_OR_PARITY")
    predictions, answers = all_predictions(folder, protocol), read_jsonl(folder / "volume-answers.jsonl")
    result = evaluate(protocol, predictions, answers)
    if (
        result != read_json(folder / "algorithm-metrics.json")
        or decision(result) != read_json(folder / "algorithm-decision.json")
        or diagnostics(folder, protocol, predictions, answers) != read_json(folder / "algorithm-diagnostics.json")
    ):
        raise ValueError("ALGORITHM_RESULT_REBUILD")
    return stages["complete"]


def replay(folder):
    verify(folder)
    protocol, _ = load_plan(folder)
    # 一份原包只能申请一次复跑，复跑包不能递归发起新一轮训练。
    if (folder / "algorithm-replay-origin.json").exists():
        raise ValueError("ALGORITHM_REPLAY_OF_REPLAY")
    target = new_folder()
    receipt_origin = {"original_run": folder.name, "replay_run": target.name, "created_at": now()}
    write_json(folder / "algorithm-replay-started.json", receipt_origin)
    for stage in STAGES[:2]:
        manifest = f"algorithm-{stage}.json"
        for name in (manifest, *read_seal(folder, manifest)["files"]):
            if not (target / name).exists():
                shutil.copyfile(folder / name, target / name)
    write_json(target / "algorithm-replay-origin.json", receipt_origin)
    predict(target)
    score(target)
    finalize(target)
    verify(target)
    differences, identical = [], True
    for w in protocol["windows"]:
        a = read_json(folder / f"algorithm-models-{w['name']}.json")
        b = read_json(target / f"algorithm-models-{w['name']}.json")
        for branch in BRANCHES:
            if a[branch]["status"] != b[branch]["status"]:
                raise ValueError("ALGORITHM_REPLAY_STATUS")
            if a[branch]["status"] == "PREDICTED":
                differences.extend(abs(x - y) for x, y in zip(a[branch]["scores"], b[branch]["scores"], strict=True))
                identical &= a[branch]["models"] == b[branch]["models"]
    if max(differences, default=0) != 0 or not identical:
        raise ValueError("ALGORITHM_REPLAY_CHANGED")
    receipt = {
        "run": target.name.removeprefix("direction-training-"),
        "original_run": folder.name,
        "scores": len(differences),
        "maximum_difference": max(differences, default=0),
        "models_identical": identical,
        "status": "REPRODUCED_NOT_NEW_EFFECT_EVIDENCE",
    }
    seal(
        target,
        "algorithm-replay.json",
        {"algorithm-replay-receipt.json": write_json(target / "algorithm-replay-receipt.json", receipt)},
    )
    return receipt
