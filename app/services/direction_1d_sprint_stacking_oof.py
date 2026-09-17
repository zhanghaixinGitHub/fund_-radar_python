"""组合训练的时间留出特征：每道训练题只能用该季度之前拟合的基础模型预测。"""

import hashlib
from collections import defaultdict

import joblib
import numpy as np

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_sector_return as ridge


def root():
    return b.ROOT / "round-58/oof"


def quarter(day):
    year, month = map(int, day[:7].split("-"))
    return f"{year}-{3 * ((month - 1) // 3) + 1:02d}-01"


def chosen(rows, cutoff):
    """沿用原始504个已成熟可用目标日期；筛选发生在标签参与拟合之前。"""
    return ridge.adaptive.training_rows(
        [r for r in rows if r["u"] < cutoff and r["mature"] < cutoff and ridge.available(r["z"])],
        cutoff,
        "MONTHLY_BAL504",
    )


def base_scores(values, head):
    """直接计算基础回归分数，不使用待预测题的标签或重算未来净值。"""
    if (
        tuple(head["code_order"]) != ridge.CODE_ORDER
        or head["shrinkage"] != 0.25
        or head["model"].n_features_in_ != 155
        or head["mean"] != [0.0] * 5
    ):
        raise ValueError("STACK_BASE_SCHEMA_CHANGED")
    x = ridge.design(values, head["scale"], ridge.CANDIDATES[1])
    scores = np.asarray(head["model"].predict(x), dtype=float)
    if scores.shape != (len(values),) or not np.isfinite(scores).all():
        raise ValueError("STACK_BASE_SCORE_INVALID")
    return scores


def meta_vectors(values, scores):
    """组合特征是三票的带符号差额和基础幅度分数，绝不包含真实涨跌幅。"""
    if len(values) != len(scores):
        raise ValueError("STACK_FEATURE_COUNT_CHANGED")
    result = []
    for z, score in zip(values, scores, strict=True):
        ridge.selected_features(z, ridge.CANDIDATES[1])
        if not ridge.available(z) or not np.isfinite(score):
            raise ValueError("STACK_FEATURE_NOT_AVAILABLE")
        votes = sum(z[i] >= 0 for i in (0, 12, 17))
        result.append([(2 * votes - 3) / 3, float(score)])
    return result


def checkpoint(rows, spec):
    group, cutoff = spec["group"], spec["cutoff"]
    train = chosen([r for r in rows if r["group"] == group], cutoff)
    if b.digest(train) != spec["training_hash"]:
        raise ValueError("STACK_INNER_TRAINING_CHANGED")
    label = f"{group}-{cutoff}"
    receipt_path = root() / f"checkpoints/{label}.json"
    if receipt_path.exists():
        receipt = b.read(receipt_path)
        path = b.PROJECT / receipt["path"]
        if (
            receipt["training_hash"] != spec["training_hash"]
            or receipt["cutoff"] != cutoff
            or receipt["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("STACK_INNER_CHECKPOINT_CHANGED")
        model = joblib.load(path)
        return (model["us_etf"] if receipt["reused"] else model), receipt
    if spec["reuse_existing_R57_2025"]:
        q = (int(cutoff[5:7]) - 1) // 3 + 1
        path = ridge.root() / f"checkpoints/{q}-{group}-{ridge.CANDIDATES[1]}.joblib"
        old = b.read(path.with_suffix(".json"))
        if old["cutoff"] != cutoff or old["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("STACK_REUSED_BASE_CHANGED")
        head = joblib.load(path)["us_etf"]
    else:
        attempt = root() / f"checkpoints/{label}.attempt.json"
        if attempt.exists():
            raise ValueError("STACK_INNER_INTERRUPTED_NO_AUTORETRY")
        b.save(attempt, {"at": b.now().isoformat(), "cutoff": cutoff, "training_hash": spec["training_hash"]})
        head = ridge.fit([r for r in rows if r["group"] == group], ridge.CANDIDATES[1], cutoff)
        path = root() / f"checkpoints/{label}.joblib"
        joblib.dump(head, path)
    if head["fit_hash"] != spec["training_hash"] or head["max_mature_date"] >= cutoff or head["fit_end"] >= cutoff:
        raise ValueError("STACK_INNER_FUTURE_TRAINING")
    receipt = {
        "at": b.now().isoformat(),
        "group": group,
        "cutoff": cutoff,
        "training_hash": spec["training_hash"],
        "reused": spec["reuse_existing_R57_2025"],
        "path": path.relative_to(b.PROJECT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    b.save(receipt_path, receipt)
    return head, receipt


def prepare(rows):
    """按预先核对的51组窗口产生训练特征；2026部分只作当前训练，不计算审核期成绩。"""
    if (root() / "manifest.json").exists():
        return load()[1]
    spec = b.read(b.ROOT / "stacking-oof-feasibility-v1/result.json")
    needed = {}
    for window in spec["outer_windows"]:
        selected = chosen([r for r in rows if r["group"] == window["group"]], window["cutoff"])
        if b.digest(selected) != window["training_hash"]:
            raise ValueError("STACK_OUTER_WINDOW_CHANGED")
        for row in selected:
            needed[row["code"], row["u"]] = row
    groups = defaultdict(list)
    for row in needed.values():
        groups[row["group"], quarter(row["u"])].append(row)
    if set(groups) != {(w["group"], w["cutoff"]) for w in spec["inner_windows"]}:
        raise ValueError("STACK_INNER_SCOPE_CHANGED")
    references = {}
    for window in spec["inner_windows"]:
        ridge.active()
        group, cutoff = window["group"], window["cutoff"]
        head, receipt = checkpoint(rows, window)
        part = sorted(groups[group, cutoff], key=lambda r: (r["u"], r["code"]))
        if any(cutoff > r["u"] or quarter(r["u"]) != cutoff for r in part):
            raise ValueError("STACK_FEATURE_LOOKAHEAD")
        path = root() / f"features/{group}-{cutoff}.json"
        if not path.exists():
            vectors = meta_vectors([r["z"] for r in part], base_scores([r["z"] for r in part], head))
            records = [
                {"code": r["code"], "u": r["u"], "base_cutoff": cutoff, "row_hash": b.digest(r), "z2": z}
                for r, z in zip(part, vectors, strict=True)
            ]
            b.save(
                path,
                {
                    "at": b.now().isoformat(),
                    "role": "CURRENT_TRAIN_FEATURE_ONLY_NOT_AUDIT"
                    if cutoff.startswith("2026")
                    else "TIME_HELD_OUT_TRAIN_FEATURE",
                    "base_receipt_hash": b.digest(receipt),
                    "records": records,
                },
            )
        value = b.read(path)
        if value["base_receipt_hash"] != b.digest(receipt):
            raise ValueError("STACK_FEATURE_BASE_CHANGED")
        references[path.relative_to(root()).as_posix()] = {"hash": b.digest(value), "receipt": receipt}
        b.save(
            root() / "progress.json",
            {"at": b.now().isoformat(), "completed_inner_windows": len(references)},
            replace=True,
        )
    result = {
        "at": b.now().isoformat(),
        "references": references,
        "questions": len(needed),
        "new_base_fits": 39,
        "reused_base_checkpoints": 12,
        "new_2026_audit_metrics": False,
    }
    b.save(root() / "manifest.json", result)
    load()
    return result


def load():
    """每次读取都核对特征、基础模型和逐题来源，不接受改哈希后的另一批样本。"""
    manifest = b.read(root() / "manifest.json")
    records = {}
    for name, ref in manifest["references"].items():
        value = b.read(root() / name)
        receipt = ref["receipt"]
        if (
            b.digest(value) != ref["hash"]
            or value["base_receipt_hash"] != b.digest(receipt)
            or hashlib.sha256((b.PROJECT / receipt["path"]).read_bytes()).hexdigest() != receipt["sha256"]
        ):
            raise ValueError("STACK_OOF_MANIFEST_CHANGED")
        for record in value["records"]:
            key = record["code"], record["u"]
            if (
                key in records
                or record["base_cutoff"] != quarter(record["u"])
                or record["base_cutoff"] != receipt["cutoff"]
            ):
                raise ValueError("STACK_OOF_ROW_LINEAGE_CHANGED")
            records[key] = record
    if len(records) != manifest["questions"]:
        raise ValueError("STACK_OOF_QUESTIONS_CHANGED")
    return records, manifest


def training_features(rows):
    records, manifest = load()
    values = []
    for row in rows:
        record = records[row["code"], row["u"]]
        if record["row_hash"] != b.digest(row) or record["base_cutoff"] > row["u"]:
            raise ValueError("STACK_META_ROW_CHANGED")
        values.append(record["z2"])
    result = np.asarray(values, dtype=float)
    if result.shape != (len(rows), 2) or not np.isfinite(result).all():
        raise ValueError("STACK_META_FEATURE_INVALID")
    # 即使特征JSON连同摘要被改写，也必须能由该季度前的真实基础模型重算。
    by_window = defaultdict(list)
    for i, row in enumerate(rows):
        by_window[row["group"], quarter(row["u"])].append(i)
    refs = {(r["receipt"]["group"], r["receipt"]["cutoff"]): r["receipt"] for r in manifest["references"].values()}
    for key, indices in by_window.items():
        receipt = refs[key]
        obj = joblib.load(b.PROJECT / receipt["path"])
        head = obj["us_etf"] if receipt["reused"] else obj
        if head["cutoff"] != key[1] or head["max_mature_date"] >= key[1] or head["fit_end"] >= key[1]:
            raise ValueError("STACK_BASE_TIME_CHANGED")
        raw = [rows[i]["z"] for i in indices]
        expected = np.asarray(meta_vectors(raw, base_scores(raw, head)))
        if not np.allclose(result[indices], expected, rtol=0, atol=1e-12):
            raise ValueError("STACK_OOF_RECOMPUTATION_CHANGED")
    return result, b.digest(manifest)
