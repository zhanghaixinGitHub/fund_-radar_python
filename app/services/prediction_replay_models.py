"""冻结已采用/并行候选的多周期组合，回放不得用基础方法冒充失败的训练模型。"""

from datetime import datetime, time, timedelta

from app.services.prediction_contract import PredictionFailure, fingerprint, prediction_policy
from app.services.prediction_features import ZONE
from app.services.prediction_models import freeze_routes, load_model, route_key
from app.services.prediction_research import validate_historical_model

RECIPE_LABELS = {
    "NAV_MOMENTUM_THREE_STATE_V2": "三分类走势基础模型",
    "TOTAL_RETURN_LOGISTIC_THREE_STATE_V2": "三分类历史学习模型",
    "NAV_MOMENTUM_BASELINE_V1": "净值趋势基础模型",
    "TOTAL_RETURN_LOGISTIC_V1": "统计学习模型",
}


def replay_model_bundles(start):
    """只取当前已登记的主模型与并行候选；三个周期齐全才参加同条件比较。

    候选来自事先登记的路由，不按本次回放收益挑选。相同配方每周期取最新登记训练版本，
    最多三组候选。今天的模型回看历史属于开发实验；训练答案必须早于整个比较区间。
    """
    policy, routes = prediction_policy(), freeze_routes()
    horizons = [h["horizon_id"] for h in policy["horizons"]]
    cutoff = datetime.combine(start, time.min, ZONE) - timedelta(microseconds=1)
    current, candidates, excluded = {}, {}, []
    candidate_recipes = set()
    for horizon in horizons:
        route = routes.get(route_key(horizon))
        if not route:
            excluded.append({"label": "当前采用模型组合", "reason": f"{horizon}尚未登记主模型"})
            continue
        for model_id in dict.fromkeys([route["model_id"], *route.get("shadow_ids", [])]):
            try:
                model = load_model(model_id)
                package = model["manifest"]
                from app.services.prediction_direction import validate_identity

                validate_identity(package, horizon)
                if (
                    package["horizonId"] != horizon
                    or package["assetGroup"] != "ALL"
                    or package["targetDefinitionId"] != policy["target_definition_id"]
                    or package["featureSchemaVersion"] != policy["feature_schema_version"]
                ):
                    raise PredictionFailure("REPLAY_MODEL_INCOMPATIBLE", "REPLAY", "模型目标或周期与回放不一致")
                validate_historical_model(package, cutoff)
                model = model | {"activationRevision": route["revision"]}
                if model_id == route["model_id"]:
                    current[horizon] = model
                # 候选按同配方跨周期组成组合，也允许复用该配方已经成为主模型的周期。
                recipe = package["recipeVersion"]
                if model_id != route["model_id"]:
                    candidate_recipes.add(recipe)
                bucket = candidates.setdefault(recipe, {})
                previous = bucket.get(horizon)

                def rank(item):
                    return item["manifest"].get("trainedAt") or "", item["modelId"]

                if previous is None or rank(model) > rank(previous):
                    bucket[horizon] = model
            except PredictionFailure as error:
                excluded.append(
                    {
                        "label": "当前采用模型" if model_id == route["model_id"] else "候选模型",
                        "modelId": model_id,
                        "horizonId": horizon,
                        "reason": error.payload["summary"],
                    }
                )
    bundles, seen = [], set()

    def append_bundle(models, role, label):
        if set(models) != set(horizons):
            excluded.append({"label": label, "reason": "缺少可用于本段历史的完整五日、二十日、半年模型"})
            return
        refs = [
            {
                "modelId": models[h]["modelId"],
                "modelHash": models[h]["modelHash"],
                "horizonId": h,
                "recipeVersion": models[h]["manifest"]["recipeVersion"],
                "labelEndMax": models[h]["manifest"].get("labelEndMax"),
                "activationRevision": models[h]["activationRevision"],
            }
            for h in horizons
        ]
        digest = fingerprint(refs)
        if digest in seen:
            return
        seen.add(digest)
        bundles.append(
            {"id": "MODEL_" + digest[:16], "label": label, "role": role, "modelRefs": refs, "models": models}
        )

    recipes = {m["manifest"]["recipeVersion"] for m in current.values()}
    name = RECIPE_LABELS.get(next(iter(recipes)), "多周期模型组合") if len(recipes) == 1 else "多周期模型组合"
    append_bundle(current, "CURRENT", name + "（当前采用）")
    for recipe, models in sorted(candidates.items()):
        # 主组合中的单个组成部分不是一个额外候选；只有实际登记为并行候选的配方才列出。
        if recipe not in candidate_recipes:
            continue
        if len(bundles) >= 4:
            break
        append_bundle(models, "CANDIDATE", RECIPE_LABELS.get(recipe, "训练模型 " + recipe) + "（候选）")
    return bundles, excluded
