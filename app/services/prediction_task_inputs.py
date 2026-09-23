"""冻结实时和人工研究来源，复用自动周期的时间/小数编码；不重复读取失败输入。"""

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories.prediction_store import encode, one
from app.services.auto_model_contract import freeze_value, thaw_value
from app.services.prediction_contract import PredictionFailure, fingerprint


def task_input(kind, task_id, fund_code, loader):
    # SQL列名只从内部枚举获取；值始终绑定参数。
    column = {"LIVE": "generation_task_id", "RESEARCH": "research_run_id"}[kind]
    query = f"SELECT payload,content_hash FROM prediction_task_input WHERE {column}=:id AND fund_code=:code"
    with get_engine().connect() as c:
        saved = one(c, query, id=task_id, code=fund_code)
    if not saved:
        try:
            value = loader()
        except PredictionFailure as error:
            value = {"frozenFailure": error.payload}
        payload = freeze_value(value)
        with get_engine().begin() as c:
            c.execute(
                text(f"""INSERT INTO prediction_task_input({column},fund_code,content_hash,payload)
              VALUES(:id,:code,:hash,CAST(:payload AS jsonb)) ON CONFLICT DO NOTHING"""),
                {"id": task_id, "code": fund_code, "hash": fingerprint(payload), "payload": encode(payload)},
            )
            saved = one(c, query, id=task_id, code=fund_code)
    if fingerprint(saved["payload"]) != saved["content_hash"]:
        raise PredictionFailure("TASK_INPUT_HASH_MISMATCH", "RECOVERY", "任务冻结来源校验失败", retryable=False)
    return thaw_value(saved["payload"])
