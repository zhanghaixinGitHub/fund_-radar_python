"""DeepSeek 仅负责原预测事实的中文组织；固定结构、低随机性、有限输出、无自动重试。"""

import json
import re
import time
from urllib.parse import urlparse

import httpx

SYSTEM_PROMPT = """你是基金预测说明的中文编辑。请用 json 输出一段连贯的综合分析。
输入 facts 是从原预测核对的观察与局限；summary、日期、周期和 limitations 由程序展示。
输出结构只有 {"analysis":"正文"}。正文一至两段，把相关观察合并，突出输入中已有的矛盾和解释局限。
每个 requiredRefs 占位符必须完整出现一次，顺序可调整，允许在同一句合并多个指标。
optionalRefs 是次要观察，可不引用，最多择一；正文每段最多包含两个 F 观察，避免堆叠指标。
不得逐项套“事实—关系—替换结果”的句式，不要列表、标题、编号或复述所有指标支持某个方向。
占位符锁定观察的指标、数值、定义或边界；不得拆分、否定、改写或加修饰来改变其含义。
自由文字仅用于串联、组织和提醒区分历史观察与未来判断，不能引入新的事实、行情判断或原因。
不得另写数字、日期、方向结论、百分比、概率、胜率、交易建议、因果关系或历史规律。
不要解释或重复占位符内的时间区间和指标；事实较少时用短文即可，不要补足篇幅。
没有本基金历史分布时不得说波动高低、常态、罕见或同类排名；波动不是上涨原因。
不得把深回撤或区间下跌解释成后续更容易上涨，不得把参数均值当本基金历史常态。
不要用“因为、所以、因此”解释涨跌。不要补写市场、资金、估值、消息等原因。
正常的局限连接句可以使用，例如“因此对后续的判断仍需与这段历史观察区分开来”或“这说明现有依据仍有限”。
B1 的解释边界和 Q1（若提供）的矛盾必须保留；不能弱化成支持证据或另行反驳。
示例结构：{"analysis":"{{F1}}。{{Q1}}。\\n另需留意，{{F2}}。{{B1}}。"}
实际占位符以输入为准，不照抄示例编号。
"""
TOKEN = re.compile(r"\{\{([FQB][1-9])\}\}")
# 数值、方向和证据边界锁在原子事实中，自由文字不能重新下判断。
# 规则校验不能证明任意自然语言语义；仍需人工审阅真实生成及对抗样例。
FORBIDDEN = re.compile(
    r"概率|置信|命中率|胜率|必然|必定|肯定|保证|稳赚|买入|卖出|加仓|减仓|抄底|止损|止盈|"
    r"资金|政策|新闻|公告|消息|利好|利空|市场情绪|超卖|超买|估值|反弹|反转|拐点|企稳|见底|"
    r"历史相似|类似情形|通常会|往往会|一定会|没有风险|持续下跌|每天.*跌|"
    r"准确率|胜算|显著|强烈|强劲|赚钱|获利|促使|导致|推动|带来|意味|说明|证明|表明|代表|"
    r"因为|所以|因此|故而|由此|从而|源于|得益|归因|越.+越|容易|有利|支持|占优|削弱|"
    r"上涨|下跌|持平|走强|走弱|承压|乐观|悲观|涨势|跌势|高位|低位|偏高|偏低|较高|较低|"
    r"更高|更低|很高|很低|不高|不低|高波动|低波动|大幅|剧烈|温和|平稳|常态|罕见|排名|分位|均值|平均|"
    r"DeepSeek|模型|算法|参数|接口|训练|特征|系数|评分|版本|后端|前端|\bAPI\b",
    re.I,
)

# 只豁免完整、经过复验的“证据不足”逻辑句，不能通过拼接半句获得市场因果豁免。
# 这些句子不包含具体行情、数值或方向；额外推断仍接受下方原规则检查。
LIMITATION_CLAUSES = {
    "因此对后续的判断仍需与这段历史观察区分开来",
    "因此该方向判断的适用范围限于已有观察",
    "所以仍需区分历史观察与未来判断",
    "这说明现有依据仍有限",
    "这表明现有依据仍有限",
    "这并不代表未来会延续",
    "因此不能仅凭这些观察确认后续方向",
    "因为缺少可靠的后续表现证据",
    "所以本次判断的解释依据仍有限",
}


def response_issues(value, facts):
    """返回稳定错误分类，区分自由数字、引用、密度与不受支持的推断，便于离线审阅。"""
    if not isinstance(value, dict) or set(value) != {"analysis"}:
        return ["NARRATIVE_SCHEMA_INVALID"]
    raw = value["analysis"]
    if not isinstance(raw, str):
        return ["NARRATIVE_TYPE_INVALID"]
    refs, issues = TOKEN.findall(raw), []
    required, optional = set(facts["requiredRefs"]), set(facts.get("optionalRefs", []))
    if len(refs) != len(set(refs)) or not required <= set(refs) or not set(refs) <= required | optional:
        issues.append("NARRATIVE_REFERENCE_INVALID")
    if len(set(refs) & optional) > 1 or any(
        sum(ref.startswith("F") for ref in TOKEN.findall(paragraph)) > 2 for paragraph in raw.split("\n")
    ):
        issues.append("NARRATIVE_OBSERVATION_DENSITY")
    plain = TOKEN.sub("", raw)
    if re.search(r"[\d%％]", plain) or re.search(
        r"百分之|千分之|[零〇一二两三四五六七八九十百千万壹贰叁肆伍陆柒捌玖拾]+[成点年月日天倍％%]", plain
    ):
        issues.append("NARRATIVE_FREE_NUMBER_OR_DATE")
    if not raw.strip() or len(raw) > 1000 or raw.count("\n") > 1 or re.search(r"[{}<>#*\r]", plain):
        issues.append("NARRATIVE_CONTENT_INVALID")
    unapproved = []
    for clause in re.split(r"[。；，！？\n]", raw):
        clause = clause.strip()
        if clause in LIMITATION_CLAUSES or re.fullmatch(r"(?:因此|所以|这说明|这表明)\{\{B1\}\}", clause):
            continue
        unapproved.append(TOKEN.sub("", clause))
    if FORBIDDEN.search("。".join(unapproved)):
        issues.append("NARRATIVE_UNSUPPORTED_ASSERTION")
    if re.search(
        r"(?:并非|不是|不再|没有|并未|未曾|否认|不等于|不应|不认为|并不)[^。；，\n]{0,12}\{\{", raw
    ) or re.search(r"\}\}[^。；，\n]{0,8}(?:不成立|有误|不准确|不可信|并不属实|不正确)", raw):
        issues.append("NARRATIVE_FACT_NEGATED")
    return issues


def validate_response(value, facts):
    """引用可重排合并，事实不可改写、遗漏、重复或否定；保留原方向和必要局限。"""
    issues = response_issues(value, facts)
    if issues:
        raise ValueError(issues[0])
    raw = value["analysis"]
    refs = TOKEN.findall(raw)
    rendered = TOKEN.sub(lambda m: facts["facts"][m[1]], raw).strip()
    if len(rendered) > 1400:
        raise ValueError("NARRATIVE_TOO_LONG")
    return {
        "styleVersion": facts["styleVersion"],
        "summary": facts["summary"],
        "context": rendered,
        # 保持 Java 透传和已有存储字段兼容；综合分析不再按系数分成支持/相反清单。
        "supporting": "",
        "opposing": "",
        "limitations": facts["limitations"],
        "evidenceRefs": {"context": refs, "supporting": [], "opposing": []},
    }


def generate(facts, settings):
    """单次有界请求；仅发送事实白名单，不发账户、密钥以外配置、原模型参数或原始响应日志。"""
    endpoint = settings.deepseek_base_url.rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or parsed.hostname != "api.deepseek.com" or parsed.username or parsed.query:
        raise ValueError("DEEPSEEK_ENDPOINT_INVALID")
    # 日期、周期和结论在固定摘要中展示，不重复发给编辑，避免被自由文字再次改写。
    prompt = {key: facts[key] for key in ("facts", "requiredRefs", "optionalRefs")}
    started = time.monotonic()
    with httpx.Client(
        timeout=httpx.Timeout(settings.deepseek_timeout_seconds, connect=3, write=3, pool=1), follow_redirects=False
    ) as client:
        with client.stream(
            "POST",
            endpoint + "/chat/completions",
            headers={"Authorization": "Bearer " + settings.deepseek_api_key.get_secret_value()},
            json={
                "model": settings.deepseek_model,
                "stream": False,
                "thinking": {"type": "disabled"},
                "temperature": 0.2,
                "max_tokens": 1800,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            },
        ) as response:
            if response.status_code != 200:
                raise ValueError(f"DEEPSEEK_HTTP_{response.status_code}")
            body = bytearray()
            for part in response.iter_bytes():
                body.extend(part)
                if len(body) > 100_000 or time.monotonic() - started > settings.deepseek_timeout_seconds + 3:
                    raise ValueError("DEEPSEEK_RESPONSE_LIMIT")
    data = json.loads(body)
    choice = data["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("DEEPSEEK_OUTPUT_INCOMPLETE")
    return validate_response(json.loads(choice["message"]["content"]), facts)
