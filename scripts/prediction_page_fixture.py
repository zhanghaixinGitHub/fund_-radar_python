"""仅供隔离三端页面验收的人工响应入口；正常app.main绝不导入此文件。

不签发发布凭证、不改真实模型/结果。人工数值仅用于验证Java/Vue展示，不能当作真实预测验收。
必须由prediction_page_acceptance的显式--synthetic-contract-fixture启动，使用随机服务Token和隔离账号库。
"""

import os
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from app.api.routes import watchlist_prediction as route
from app.main import app
from app.schemas.cash_forecast import CashForecastView
from app.services.watchlist_prediction import get_watchlist_prediction, project_cash_forecast


def create_app():
    root = Path(__file__).resolve().parents[1] / ".local-runs"
    raw = os.environ.get("PREDICTION_PAGE_FIXTURE_DIR")
    if not raw:
        raise RuntimeError("explicit isolated fixture directory required")
    directory = Path(raw).resolve()
    if directory.parent != root.resolve() or not re.fullmatch(r"prediction_e2e_[0-9a-f]{32}", directory.name):
        raise RuntimeError("fixture must belong to this local acceptance run")
    if not directory.is_dir():
        raise RuntimeError("fixture run directory missing")

    def read(fund_code: str):
        real = get_watchlist_prediction(fund_code)
        control = directory / "fixture-mode.txt"
        mode = control.read_text(encoding="utf-8").strip() if control.exists() else "real"
        if fund_code != "008888" or mode == "real":
            return real
        if mode not in {"available", "stale", "revoked"}:
            raise ValueError("unknown isolated fixture mode")
        status = {"available": "AVAILABLE", "stale": "STALE", "revoked": "MODEL_NOT_RELEASED"}[mode]
        reasons = {
            "available": (),
            "stale": ("FORECAST_WINDOW_ENDED",),
            "revoked": ("PUBLICATION_AUTHORIZATION_CHANGED",),
        }[mode]
        result = CashForecastView(
            fund_code=fund_code,
            cutoff_date=date(2026, 9, 8),
            forecast_id=UUID("ff8f2cbd-696b-42f8-91db-4937bf571574"),
            status=status,
            target_base_date=date(2026, 9, 8),
            target_end_date=date(2026, 10, 14),
            generated_at=datetime(2026, 9, 9, 2, tzinfo=UTC),
            model_hash="a" * 64,
            up_probability=Decimal("0.68") if mode == "available" else None,
            direction="UP" if mode == "available" else None,
            reason_codes=reasons,
        )
        projected = project_cash_forecast(
            result, latest_date=real.latest_nav_date, research_run_id=real.research_run_id
        )
        return projected.model_copy(
            update={
                "message": "【隔离验收数据】仅验证页面分支，当前真实模型仍未发布。",
                "disclaimer": "人工协议样本，不是真实基金预测；不作为模型发布通过的证据。",
            }
        )

    route.get_watchlist_prediction = read
    return app
