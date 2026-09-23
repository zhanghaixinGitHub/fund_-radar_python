"""多周期模型/研究操作入口；仅本机配置库，历史任务不得伪装LIVE。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import get_engine  # noqa: E402
from app.services.prediction_contract import prediction_policy  # noqa: E402
from app.services.prediction_models import (  # noqa: E402
    FEATURES,
    bootstrap_models,
    model_directory,
    model_status,
    register_model,
)


def import_legacy():
    """旧目标只登记，不设为新周期主模型；缺标签截止明确保留未知。"""
    from app.services.direction_experiment import load_experiment_models

    for old in load_experiment_models():
        register_model(
            {
                "adapter": "LEGACY_LINEAR_V1",
                "recipeVersion": old["branch"],
                "horizonId": "LEGACY_T20_V1",
                "targetDefinitionId": "LEGACY_CUTOFF_CASH_RETURN_20_V1",
                "assetGroup": "LEGACY_THREE_FUNDS",
                "featureSchemaVersion": "LEGACY_NAV_7_V1",
                "features": list(FEATURES),
                "featureUnits": ["RATIO"] * 6 + ["SESSIONS"],
                "missingPolicy": "REQUIRE_61_POINTS",
                "threshold": 0.5,
                "labelEndMax": None,
                "trainedAt": None,
                "legacyFitEnd": old["fit_end"],
                "codeVersion": old["version"],
                "dependencies": {},
                "evidenceLevel": "DEVELOPMENT_ONLY",
                "parameters": old,
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "status", "backup-manifest"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    url = get_engine().url
    if args.command == "bootstrap":
        if url.host not in {"localhost", "127.0.0.1", "::1"} or url.database != "fund_ai":
            raise RuntimeError("此本地初始化命令只允许已核验的本机 fund_ai 库")
        bootstrap_models()
        import_legacy()
    result = model_status()
    if args.command == "backup-manifest":
        result.update(
            modelDirectory=str(model_directory()),
            policy=prediction_policy(),
            recovery="恢复相同数据库和完整模型目录；逐包校验hash，再以CAS恢复路由，不删除历史预测",
        )
    raw = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw, encoding="utf-8")
        print(f"Saved {len(result['models'])} models, {len(result['routes'])} routes")
    else:
        print(raw)


if __name__ == "__main__":
    main()
