"""招商官网匿名公开基金曲线协议；只取公开页面/客户端，签名只驻留内存。"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from app.services.fund_exposure_common import now
from app.services.fund_materials_store import versioned_save

INCOME_URL = "https://www.cmfchina.com/ws-business-server/fund/getIncomeList"
CLIENT_URL = "https://www.cmfchina.com/ws6/base-service.js"
CHART_URL = "https://www.cmfchina.com/ws6/FundChatCom.vue.js"


def _record(folder, label, response):
    raw = response.content
    if len(raw) > 2_000_000:
        raise ValueError("CMF_PUBLIC_PROTOCOL_RESPONSE_BOUND")
    digest = hashlib.sha256(raw).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (label + "-" + digest + ".raw")
    if not path.exists():
        path.write_bytes(raw)
    receipt = {
        "url": str(response.request.url),
        "status": response.status_code,
        "received_at": now().isoformat(),
        "sha256": digest,
        "bytes": len(raw),
        "file": str(path.resolve()),
    }
    versioned_save(folder / (label + "-receipt.json"), receipt)
    response.raise_for_status()
    return response.text, receipt


def _public_query(page, client_source, params, *, node_binary=None):
    node = Path(node_binary) if node_binary else Path(__file__).resolve().parents[3] / ".tools/node/node.exe"
    if not node.is_file():
        raise ValueError("CMF_D_NODE_RUNTIME_REQUIRED")
    runner = Path(__file__).with_name("cmf_public_sign.cjs")
    result = subprocess.run(
        [str(node), str(runner)],
        input=json.dumps({"page": page, "client": client_source, "params": params}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        # 不把Node stderr或公开客户端封装材料倒进业务日志。
        raise ValueError("CMF_PUBLIC_CLIENT_PROTOCOL_NOT_REPLAYABLE")
    return json.loads(result.stdout)


def income_list(client, params, *, audit_directory, node_binary=None):
    """同正常官网会话重放自定义日期控件，返回真正上游原响应，不重新序列化。

    当前官网近36月控件会覆盖开始日；手选日期的公开脚本明确清空queryMonth。
    本入口仅支持这种已核准的显式窗口，不读登录令牌或执行浏览器校验脚本。
    """
    code = params.get("productCode", "")
    if (
        not re.fullmatch(r"[0-9]{6}", str(code))
        or params.get("queryMonth") != ""
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", params.get("startDate", ""))
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", params.get("endDate", ""))
    ):
        raise ValueError("CMF_PUBLIC_EXPLICIT_DATE_WINDOW_REQUIRED")
    folder = Path(audit_directory)
    page_url = "https://www.cmfchina.com/web/fundDetail/" + code + "/index.html"
    page, page_ref = _record(folder, "page", client.get(page_url))
    source, client_ref = _record(folder, "client", client.get(CLIENT_URL))
    chart, chart_ref = _record(folder, "chart", client.get(CHART_URL))
    if 'n.queryMonth=""' not in chart or 'format("YYYY-MM-DD")' not in chart:
        raise ValueError("CMF_PUBLIC_CUSTOM_RANGE_PROTOCOL_CHANGED")
    signed = _public_query(page, source, params, node_binary=node_binary)
    response = client.get(INCOME_URL, params=signed["params"], headers={**signed["headers"], "Referer": page_url})
    # 只记录解包前原响应摘要与普通参数身份，不存签名、密文或会话cookie。
    versioned_save(
        folder / "anonymous-request-receipt.json",
        {
            "url": INCOME_URL,
            "method": "GET",
            "params": params,
            "status": response.status_code,
            "received_at": now().isoformat(),
            "response_sha256": hashlib.sha256(response.content).hexdigest(),
            "response_bytes": len(response.content),
            "public_protocol_sources": [page_ref, client_ref, chart_ref],
            "protocol": "CMF_OFFICIAL_PUBLIC_ANONYMOUS_SM3_SM4_QUERY_V1",
            "login_token_used": False,
            "response_reencoded": False,
        },
    )
    return response
