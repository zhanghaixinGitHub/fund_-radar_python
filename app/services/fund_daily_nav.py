"""按实际缺项重试基准日净值；复用既有规范入库，不填前值、不重提全量作业。"""

import hashlib
import json
from datetime import date
from pathlib import Path

from app.core.config import get_settings
from app.services.fund_exposure_common import now
from app.services.fund_materials_store import versioned_save
from app.services.fund_prediction_nav_identity import current_nav_identity, nav_identity
from app.services.tushare_fund_sync import TushareFundSyncService


def refresh_missing_nav(code, base_date, output, *, service_factory=TushareFundSyncService):
    day = date.fromisoformat(str(base_date))
    before = current_nav_identity(code, day)
    if before != nav_identity([], day):
        return {"status": "BASE_NAV_PRESENT", "nav_input_hash": before}
    output = Path(output)
    receipts = []
    secret = get_settings().tushare_token.get_secret_value().encode()
    service = service_factory()

    def capture(response):
        response.read()
        request = json.loads(response.request.content)
        raw = response.content
        if len(raw) > 2_000_000 or (secret and secret in raw):
            raise ValueError("BASE_NAV_RESPONSE_NOT_SAFE_TO_ARCHIVE")
        key = hashlib.sha256(raw).hexdigest()
        path = output / "raw" / (key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(raw)
        try:
            body = json.loads(raw)
            data = body.get("data") or {}
            items = data.get("items") if isinstance(data, dict) else None
            api_code, row_count = body.get("code"), len(items) if isinstance(items, list) else None
        except (ValueError, AttributeError):
            api_code, row_count = None, None
        receipts.append({"api": request["api_name"], "params": request.get("params", {}),
                         "http_status": response.status_code, "sha256": key,
                         "api_code": api_code, "rows": row_count,
                         "raw_path": str(path.resolve()), "received_at": now().isoformat()})

    result = {"fund_code": code, "base_nav_date": str(day), "attempted_at": now().isoformat()}
    try:
        service._provider._client.event_hooks["response"].append(capture)
        outcome = service.sync_market_nav_history((code + ".OF",), start_date=day, end_date=day)
        result["sync"] = outcome.to_payload()
    except Exception as error:
        result["failure"] = type(error).__name__
    finally:
        service.close()
    after = current_nav_identity(code, day)
    last = receipts[-1] if receipts else {}
    confirmed_empty = (last.get("http_status") == 200 and last.get("api_code") == 0 and last.get("rows") == 0)
    result.update(status="AVAILABLE" if after != nav_identity([], day)
                  else "WAITING_NAV" if confirmed_empty else "SOURCE_READ_FAILED",
                  nav_input_hash=after, receipts=receipts)
    versioned_save(output / "receipt.json", result)
    return result
