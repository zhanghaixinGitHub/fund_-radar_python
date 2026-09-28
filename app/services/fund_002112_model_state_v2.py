"""002112 独立序列化修复：文件字节完整性与模型数值状态分别验证。

本模块不拟合、不读取研究资料、不登记模型。仅接受本项目固定环境和已知
估计器状态；未知字段/类型直接失败，避免漏掉未来库版本新增的预测状态。
旧第三轮代码和失败检查点均不修改。摘要 v2 不能冒充旧 joblib 对象摘要。
"""

import hashlib
import io
import json
import math
import os
import re
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn._loss._loss import CyHalfMultinomialLoss
from sklearn._loss.link import Interval, MultinomialLogit
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.services.fund_002112_round3_data import CLASSES, FEATURES, TIE_ORDER
from app.services.fund_002112_round3_model import LOGISTIC, TREE, VARIANTS, compare_predictions, predict

SCHEMA = "002112-model-numeric-state-v2"
ENVIRONMENT = {
    "python": "3.13.5",
    "numpy": "2.5.3",
    "scikit-learn": "1.9.0",
    "joblib": "1.6.0",
    "scipy": "1.18.1",
    "threadpoolctl": "3.6.0",
}
MODEL_FIELDS = {
    "variant",
    "features",
    "classes",
    "tie_order",
    "recipe",
    "training_hash",
    "weights_hash",
    "scaler",
    "classifier",
}
# 全量覆盖本地固定配方产生的对象字段，包括分箱、随机状态和训练计数。
# 采用精确集合，不用“遍历能识别的字段，其余忽略”的宽松策略。
OBJECT_FIELDS = {
    StandardScaler: set("with_mean with_std copy n_features_in_ n_samples_seen_ mean_ var_ scale_".split()),
    LogisticRegression: set(LOGISTIC) | set("n_features_in_ classes_ n_iter_ coef_ intercept_".split()),
    HistGradientBoostingClassifier: set(TREE)
    | set(
        "is_categorical_ _preprocessor _is_categorical_remapped n_features_in_ _label_encoder "
        "classes_ n_trees_per_iteration_ _fitted_with_sw _random_seed _feature_subsample_rng "
        "_n_features _loss do_early_stopping_ _use_validation_data _bin_mapper "
        "_baseline_prediction _predictors _scorer train_score_ validation_score_".split()
    ),
    LabelEncoder: {"classes_"},
    _BinMapper: set(
        "n_bins subsample is_categorical known_categories random_state n_threads is_categorical_ "
        "missing_values_bin_idx_ bin_thresholds_ n_bins_non_missing_".split()
    ),
    TreePredictor: {"nodes", "binned_left_cat_bitsets", "raw_left_cat_bitsets"},
    HalfMultinomialLoss: set(
        "closs link n_classes xp device approx_hessian constant_hessian interval_y_true "
        "interval_y_pred class_indexing_offsets y_true_int y_true_one_hot".split()
    ),
    MultinomialLogit: set(),
    Interval: {"low", "high", "low_inclusive", "high_inclusive"},
}


def runtime_environment():
    """返回实际数值环境；不安装、升级或切换任何依赖。"""
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        **{name: version(name) for name in ENVIRONMENT if name != "python"},
    }


def check_environment():
    if runtime_environment() != ENVIRONMENT:
        raise ValueError("NUMERIC_ENVIRONMENT_CHANGED")


def _dtype(dtype):
    """保留字段顺序、数值类型、位宽和子数组形状；排除内存偏移和填充字节。"""
    if dtype.metadata:
        raise ValueError("DTYPE_METADATA_UNSUPPORTED")
    if dtype.fields:
        if any(len(dtype.fields[name]) != 2 for name in dtype.names):
            raise ValueError("DTYPE_TITLES_UNSUPPORTED")
        return ["record", [[name, _dtype(dtype.fields[name][0])] for name in dtype.names]]
    if dtype.subdtype:
        base, shape = dtype.subdtype
        return ["subarray", _dtype(base), list(shape)]
    if dtype.kind not in "biufcSU":
        raise ValueError("DTYPE_UNSUPPORTED")
    return [dtype.kind, dtype.itemsize]


def _array(value):
    """按逻辑元素顺序取值；空数组、切片、C/F 布局和端序使用相同规范。

    结构化树节点逐字段读取，不能将不确定的 C 结构体填充字节当作预测值。
    所有数组禁止 NaN/无穷；损失函数区间的无穷端点作为普通标量单独编码。
    """
    descriptor = _dtype(value.dtype)
    if value.dtype.fields:
        values = [[name, _array(value[name])] for name in value.dtype.names]
    else:
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError("NONFINITE_MODEL_ARRAY")
        normalized = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
        values = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
    return ["array", descriptor, list(value.shape), values]


def _canonical(value):
    """仅处理明确支持的类型，不调用任意对象的 repr 或通用 pickle。"""
    kind = type(value)
    if kind is np.ndarray:
        return _array(value)
    if isinstance(value, np.generic):
        return ["numpy-scalar", _array(np.asarray(value))]
    if value is None:
        return ["none"]
    if kind in (bool, str, int):
        return [kind.__name__, value]
    if kind is float:
        if math.isnan(value):
            raise ValueError("NAN_MODEL_SCALAR")
        return ["float", value.hex()]
    if kind in (list, tuple):
        return [kind.__name__, [_canonical(item) for item in value]]
    if kind is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("STATE_KEY_MUST_BE_STRING")
        return ["dict", [[key, _canonical(value[key])] for key in sorted(value)]]
    if kind is np.random.Generator:
        if type(value.bit_generator) is not np.random.PCG64:
            raise ValueError("RANDOM_GENERATOR_CHANGED")
        return ["PCG64", _canonical(value.bit_generator.state)]
    if kind is CyHalfMultinomialLoss:
        if value.__getstate__() is not None:
            raise ValueError("CYTHON_LOSS_STATE_CHANGED")
        return ["sklearn._loss._loss.CyHalfMultinomialLoss", "stateless"]
    if kind in OBJECT_FIELDS:
        fields = vars(value)
        if set(fields) != OBJECT_FIELDS[kind]:
            raise ValueError("ESTIMATOR_FIELDS_CHANGED:" + kind.__name__)
        return [kind.__module__ + "." + kind.__qualname__, _canonical(fields)]
    raise ValueError("STATE_TYPE_UNSUPPORTED:" + kind.__module__ + "." + kind.__qualname__)


def numeric_digest(value):
    """生成带版本域的 SHA-256；不同类型/形状/值不因存储规范化而混淆。"""
    encoded = json.dumps([SCHEMA, _canonical(value)], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def model_state_digest(model):
    """验证固定研究契约，并覆盖完整已知估计器状态；不拟合或修改模型。"""
    check_environment()
    if type(model) is not dict or set(model) != MODEL_FIELDS:
        raise ValueError("MODEL_FIELDS_CHANGED")
    variant = model["variant"]
    if variant not in VARIANTS:
        raise ValueError("MODEL_VARIANT_CHANGED")
    size = VARIANTS[variant]
    recipe = TREE if variant == "T20" else LOGISTIC
    classifier_type = HistGradientBoostingClassifier if variant == "T20" else LogisticRegression
    classifier, scaler = model["classifier"], model["scaler"]
    if (
        type(classifier) is not classifier_type
        or type(scaler) is not StandardScaler
        or model["features"] != list(FEATURES[:size])
        or model["classes"] != list(CLASSES)
        or model["tie_order"] != list(TIE_ORDER)
        or model["recipe"] != recipe
        or classifier.get_params(deep=False) != recipe
        or scaler.get_params(deep=False) != {"copy": True, "with_mean": True, "with_std": True}
        or classifier.classes_.tolist() != list(CLASSES)
        or classifier.n_features_in_ != size
        or scaler.n_features_in_ != size
    ):
        raise ValueError("MODEL_CONTRACT_CHANGED")
    for name in ("training_hash", "weights_hash"):
        if not isinstance(model[name], str) or not re.fullmatch(r"[0-9a-f]{64}", model[name]):
            raise ValueError("MODEL_INPUT_IDENTITY_INVALID")
    return numeric_digest(model)


def _slot_paths(directory, slot):
    # 只接收槽位标识，不让槽位成为任意模型路径或覆盖旧运行目录的相对路径。
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", slot):
        raise ValueError("INVALID_SLOT")
    root = Path(directory)
    return root / (slot + ".joblib"), root / (slot + ".manifest.json")


def _write_exclusive(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def save_model(model, directory, slot):
    """只保存调用方已有模型，不拟合；返回供独立进程核验的清单 SHA-256。

    目录必须是调用方拥有的独立受控目录。两个文件均独占创建，已有文件或
    中断遗留文件一律拒绝覆盖；没有清单的半成品不能加载为合格模型。
    """
    state = model_state_digest(model)
    model_path, manifest_path = _slot_paths(directory, slot)
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("MODEL_SLOT_ALREADY_EXISTS")
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=0)
    raw = buffer.getvalue()
    manifest = {
        "schema": SCHEMA,
        "environment": runtime_environment(),
        "slot": slot,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "state_sha256": state,
        "variant": model["variant"],
        "training_hash": model["training_hash"],
        "weights_hash": model["weights_hash"],
    }
    encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    _write_exclusive(model_path, raw)
    _write_exclusive(manifest_path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def restore_model(directory, slot, expected_manifest_sha256):
    """先验证外部固定的清单摘要和文件字节，再反序列化同一份已验字节。

    SHA-256 是完整性校验，不是签名；仅限本项目自行生成的受控本地模型，
    禁止外部上传文件。调用方须把预期清单摘要放在不可变检查点/任务证据中。
    """
    check_environment()
    model_path, manifest_path = _slot_paths(directory, slot)
    encoded = manifest_path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_manifest_sha256:
        raise ValueError("MANIFEST_BYTES_CHANGED")
    manifest = json.loads(encoded)
    if manifest.get("schema") != SCHEMA or manifest.get("environment") != ENVIRONMENT or manifest.get("slot") != slot:
        raise ValueError("MANIFEST_CONTRACT_CHANGED")
    raw = model_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["file_sha256"]:
        raise ValueError("MODEL_FILE_BYTES_CHANGED")
    model = joblib.load(io.BytesIO(raw))
    if model_state_digest(model) != manifest["state_sha256"]:
        raise ValueError("MODEL_NUMERIC_STATE_CHANGED")
    if any(model[name] != manifest[name] for name in ("variant", "training_hash", "weights_hash")):
        raise ValueError("MODEL_MANIFEST_IDENTITY_CHANGED")
    return model


def verify_all_predictions(model, training, exams, reference):
    """重算所有训练行和考试行；方向完全相同且分数最大差不超过 1e-12。

    验证参照分数的维数和有限性，避免 NaN 与阈值比较返回 False 后被放行。
    输入行的哈希和实际类别同时核对，不能只选几行探针证明模型等价。
    """
    model_state_digest(model)
    result = {}
    for name, rows in (("train", training), ("exam", exams)):
        expected = reference[name]
        scores = np.asarray([row["scores"] for row in expected], dtype=float)
        if scores.shape != (len(rows), 3) or not np.isfinite(scores).all():
            raise ValueError("REFERENCE_SCORES_INVALID")
        actual = predict(model, rows)
        result[name] = {"rows": len(rows), "max_score_delta": compare_predictions(expected, actual)}
    return result
