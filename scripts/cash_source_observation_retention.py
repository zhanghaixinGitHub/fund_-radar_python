"""本机观察日志保留期维护：默认只预览，加--execute才清理一批已到期记录。"""

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.repositories.cash_source_observation import expired_observation_ids, purge_expired_observations  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="明确执行不可恢复的过期观察日志清理，不影响源表/样本/模型"
    )
    parser.add_argument("--max-records", type=int, default=1000, help="单批最多删除1..1000条已到期日志")
    args = parser.parse_args()
    if not 1 <= args.max_records <= 1000:
        parser.error("--max-records must be 1..1000")
    url = make_url(get_settings().ai_database_url)
    if url.host not in {"localhost", "127.0.0.1", "::1"} or url.database != "fund_ai":
        raise RuntimeError("retention maintenance is restricted to local fund_ai")
    engine = create_engine(
        url,
        hide_parameters=True,
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000 -c lock_timeout=3000"},
    )
    try:
        with Session(engine) as session, session.begin():
            if args.execute:
                deleted = purge_expired_observations(session, limit=args.max_records)
                result = {
                    "mode": "PURGE_EXPIRED_OBSERVATIONS",
                    "deleted_count": deleted,
                    "source_samples_models_changed": False,
                }
            else:
                session.execute(text("SET TRANSACTION READ ONLY"))
                ids = expired_observation_ids(session, limit=args.max_records)
                result = {"mode": "DRY_RUN", "eligible_count_in_batch": len(ids), "deleted_count": 0}
        print(json.dumps(result))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
