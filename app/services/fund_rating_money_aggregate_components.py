"""可独立移植的M2货币收益/稳定性原件编译器（无网络、数据库或写入）。

每次入口先复核所有原字节，再从原始收益JS、实际PDF和官方JSON响应重放。
只更新return/stability；未知周末不拆分、不落入日序列，标准差不参与任何约束。
固定参照、原窗口、M2量尺和子权重均冻结；完整总分与等级永远不由本模块返回。
"""

from __future__ import annotations

import calendar
import json
import math
import re
import unicodedata
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from decimal import localcontext
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

# 原始响应统一归档外接盘；历史路径只作为数据库别名，不生成C盘源文件。
from app.services import material_content as _sources
from app.services.material_content import (
    content_digest,
    location_exists,
    same_content,
    source_bytes,
    source_exists,
    source_size,
)

VERSION = "M2_MONEY_AGGREGATE_COMPONENTS_20261030_1"
AS_OF, START = "2026-09-30", "2023-09-30"
PEERS = ("000539", "000647", "000693", "000575", "482002")
NAMES = {
    "000009": "易方达天天理财货币A",
    "000539": "中银活期宝货币A",
    "000647": "易方达财富快线货币A",
    "000693": "建信现金添利货币A",
    "000575": "兴全添利宝货币",
    "482002": "工银货币A",
}
LEGAL_NAME = "易方达天天理财货币市场基金"
PEER_LEGAL_NAME = "工银瑞信货币市场基金"
M2_SHA = "f0bd82de435d3e4bea9273151a5abfc91e9ca64301cf857f3f007de68c2784ae"
COHORT_SHA = "a0fe49ed9494e47acf559303a2f2342c9a206362b389c72d8e9afe7a68236abc"
FIELD_SHA = "975921b1e0d0dbd5ae660dcc2fd70982aa5139af6194dcd0551a74f1b5f23c37"
CONTRACTS = {
    "2023-07-04": "5fff1ad9dcbf89153ab312bb68f82842f23b6d45c561d666fca0714e789b4117",
    "2026-06-11": "05fd5d5d5fe7cac86441ffdd525ed289c261d96bf5c95e1e9d14876543d7a4f7",
}
ALLOWED_KEYS = {
    "version",
    "fund_code",
    "legal_name",
    "share_class",
    "currency",
    "as_of_date",
    "window_start",
    "fixed_peers",
    "rules_ref",
    "cohort_ref",
    "field_definition_ref",
    "primary_ref",
    "target_income_refs",
    "peer_income_refs",
    "incomplete_peer",
    "sources",
}


def canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fail(code: str) -> None:
    raise ValueError("M2_AGG_" + code)


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


RESEARCH_ON = date(2026, 10, 30)


def _source_dates(ref: dict) -> None:
    """取得日和原发表日分别保留；不把研究日取得的历史收益说成截止日当时已知。

    未知取得日仍为null。已披露的日期必须可解析，且不能来自本次研究日之后；
    合同、法律产品和定期报告另须在9/30前公开。收益序列只在原自然日窗口取值。
    """
    for field in ("acquired_at", "published_on"):
        value = ref.get(field)
        if value is None:
            continue
        try:
            observed = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except (ValueError, TypeError, AttributeError):
            fail("SOURCE_DATE")
        if observed > RESEARCH_ON:
            fail("SOURCE_FUTURE_RESEARCH_DATE")
        if (
            field == "published_on"
            and ref.get("role")
            in {"STATUTORY_AGGREGATE_RETURN", "STATUTORY_SETTLEMENT_CONTRACT", "TARGET_LEGAL_PRODUCT"}
            and observed > date.fromisoformat(AS_OF)
        ):
            fail("SOURCE_FUTURE_FUNDAMENTAL_PUBLICATION")


class Originals:
    """来源限定root/sources，逐件检查真实字节；元数据不能代替原件身份。"""

    def __init__(self, inputs: dict, root: Path):
        self.root = Path(root).resolve()
        self.sources = inputs.get("sources", {})
        if not self.sources or len(self.sources) > 200:
            fail("SOURCE_COUNT")
        self.contents = {}
        for digest, ref in self.sources.items():
            path = (self.root / ref["file"]).resolve()
            if (
                not re.fullmatch("[0-9a-f]{64}", digest)
                or not path.is_relative_to(self.root / "sources")
                or not source_exists(path)
                or not 0 < source_size(path) <= 25_000_000
            ):
                fail("SOURCE_PATH")
            content = source_bytes(path)
            if content_digest(content) != digest or ref.get("sha256") != digest:
                fail("SOURCE_HASH")
            _source_dates(ref)
            self.contents[digest] = content

    def raw(self, digest: str, kind: str | None = None) -> bytes:
        if digest not in self.contents or (kind and self.sources[digest]["kind"] != kind):
            fail("SOURCE_ROLE")
        return self.contents[digest]

    def json(self, digest: str) -> dict:
        return json.loads(self.raw(digest, "json").decode("utf-8-sig"), parse_float=D)

    def text(self, digest: str) -> str:
        return self.raw(digest).decode("utf-8-sig")

    def pdf(self, digest: str) -> str:
        return _pdf_text(digest, self.raw(digest, "pdf"))


@lru_cache(maxsize=32)
def _pdf_text(digest: str, content: bytes) -> str:
    """缓存以原字节为键；每次公开入口仍先验全部原字节，不缓存校验结论。"""
    from app.services.material_content import document_reader as PdfReader

    if not content.startswith(b"%PDF"):
        fail("PDF_BYTES")
    return normalized("\n".join(page.extract_text() for page in PdfReader(content).pages))


def _cn(n: int) -> str:
    digits = "零一二三四五六七八九"
    if n < 10:
        return digits[n]
    return (digits[n // 10] if n >= 20 else "") + "十" + (digits[n % 10] if n % 10 else "")


def _date_in_text(value: str, text: str) -> bool:
    d = date.fromisoformat(value)
    chinese = "".join("〇" if c == "0" else _cn(int(c)) for c in str(d.year))
    variants = [
        f"{d.year}年{d.month}月{d.day}日",
        chinese + "年" + _cn(d.month) + "月" + _cn(d.day) + "日",
        chinese.replace("〇", "零") + "年" + _cn(d.month) + "月" + _cn(d.day) + "日",
    ]
    return any(v in text for v in variants)


def _rules(inputs: dict, src: Originals) -> None:
    if (
        set(inputs) != ALLOWED_KEYS
        or inputs.get("version") != VERSION
        or inputs.get("fund_code") != "000009"
        or inputs.get("legal_name") != LEGAL_NAME
        or inputs.get("share_class") != "A"
        or inputs.get("currency") != "CNY"
        or inputs.get("as_of_date") != AS_OF
        or inputs.get("window_start") != START
        or tuple(inputs.get("fixed_peers", [])) != PEERS
    ):
        fail("FROZEN_SCOPE")
    if inputs["rules_ref"] != M2_SHA or inputs["cohort_ref"] != COHORT_SHA:
        fail("FROZEN_RULE_OR_COHORT_HASH")
    protocol, cohort = src.json(M2_SHA), src.json(COHORT_SHA)
    if (
        protocol["version"] != "MONEY8-M2-20261023"
        or protocol["as_of"] != AS_OF
        or protocol["target"] != "000009"
        or tuple(cohort["admitted"]) != PEERS
        or cohort["method_sha256"] != M2_SHA
        or any(cohort["rows"][p]["status"] != "ADMITTED" for p in PEERS)
        or protocol["return"]["inner_weights"] != {"one_year": ".30", "three_year_annual": ".70"}
        or protocol["return"]["annual_gap_unit"] != ".001"
        or protocol["stability"]["windows"] != {"monthly": 36, "rolling_annual": 25}
        or protocol["stability"]["inner_weights"] != {"monthly": ".30", "rolling_annual": ".70"}
    ):
        fail("FROZEN_MATH")
    if inputs["field_definition_ref"] != FIELD_SHA or "EnumSourceTypeIncomeUnit" not in src.text(FIELD_SHA):
        fail("TARGET_INCOME_FIELD_DEFINITION")
    primary = src.pdf(inputs["primary_ref"])
    if (
        LEGAL_NAME not in primary
        or "000009" not in primary
        or "人民币" not in primary
        or not _date_in_text("2026-06-30", primary[:6000])
        or "本基金利润分配是按日结转份额" not in primary
    ):
        fail("TARGET_LEGAL_SETTLEMENT_IDENTITY")


def _append(rows: dict, day: str, value: D) -> None:
    if day in rows and rows[day] != value:
        fail("CONFLICTING_DAILY_RECORD")
    if not value.is_finite() or value <= -10000:
        fail("INVALID_DAILY_INCOME")
    rows[day] = value


def _efunds(src: Originals, refs: list[str]) -> dict[str, D]:
    rows = {}
    for ref in refs:
        text = src.text(ref)
        match = re.fullmatch(r'\s*mk_000009(?:_all|_1y)?\s*=\s*"([^"]*)";?\s*', text)
        if not match:
            fail("TARGET_RAW_JS_IDENTITY")
        records = match[1].split(";")
        header = records[0].split("_")
        if header != ["0", "1", "5", "6", "7"]:
            fail("TARGET_RAW_FIELD_ORDER")
        for record in records[1:]:
            if not record:
                continue
            parts = record.split("_")
            # 官网行尾保留一个空分隔符；只容忍实际空尾，不猜新增字段或换列。
            if len(parts) < len(header) or any(parts[len(header) :]):
                fail("TARGET_RAW_ROW")
            day = datetime.strptime(parts[0], "%Y%m%d").date().isoformat()
            _append(rows, day, D(parts[header.index("6")]))
    return rows


def _eastmoney(src: Originals, ref: str, code: str) -> dict[str, D]:
    text = src.text(ref)
    identity = re.search(r'fS_code\s*=\s*"(\d{6})"', text)
    name = re.search(r'fS_name\s*=\s*"([^"]+)"', text)
    array = re.search(r"Data_millionCopiesIncome\s*=\s*(\[.*?\]);", text, re.S)
    if (
        not identity
        or identity[1] != code
        or not name
        or name[1] != NAMES[code]
        or not re.search(r"ishb\s*=\s*true", text)
        or not array
    ):
        fail("PEER_RAW_JS_IDENTITY")
    rows = {}
    for stamp, value in json.loads(array[1], parse_float=D):
        day = datetime.fromtimestamp(int(stamp) / 1000, timezone(timedelta(hours=8))).date().isoformat()
        if day in rows:
            fail("DUPLICATE_RAW_DAY")
        _append(rows, day, D(value))
    return rows


def _wealth(rows: dict[str, D]) -> dict:
    first, last = date.fromisoformat(START), date.fromisoformat(AS_OF)
    dates = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
    if any(d not in rows for d in dates):
        fail("COMPLETE_PEER_MISSING_NATURAL_DAY")
    growth = {START: D(1)}
    for previous, day in zip(dates, dates[1:]):  # noqa: B905 -- 相邻日期窗口按较短序列配对，保留冻结计算语义。
        growth[day] = growth[previous] * (1 + rows[day] / 10000)
    ends = _month_ends()
    return {
        "one_year": growth[AS_OF] / growth["2025-09-30"] - 1,
        "three_year_annual": growth[AS_OF] ** (D("365.25") / 1096) - 1,
        "monthly": [{"start": a, "end": b, "return": growth[b] / growth[a] - 1} for a, b in zip(ends, ends[1:])],  # noqa: B905 -- 相邻日期窗口按较短序列配对，保留冻结计算语义。
        "annual": [
            {"start": ends[i - 12], "end": ends[i], "return": growth[ends[i]] / growth[ends[i - 12]] - 1}
            for i in range(12, 37)
        ],
        "actual_record_count": len(dates),
    }


def _month_ends() -> list[str]:
    return [
        date(
            2023 + (8 + i) // 12, (8 + i) % 12 + 1, calendar.monthrange(2023 + (8 + i) // 12, (8 + i) % 12 + 1)[1]
        ).isoformat()
        for i in range(37)
    ]


def _contracts(src: Originals, peer: dict) -> None:
    if set(peer["contracts"]) != set(CONTRACTS):
        fail("CONTRACT_PERIODS")
    catalog = src.json(peer["catalog_ref"])
    records = catalog["data"]["content"]
    for published, ref in peer["contracts"].items():
        if ref != CONTRACTS[published]:
            fail("SETTLEMENT_CONTRACT_HASH")
        descriptor = src.sources[ref]
        records_matching = [
            r
            for r in records
            if r["announcementDate"] == published
            and "482002" in r["fundCode"]
            and "招募说明书" in r["announcementName"]
            and unquote(descriptor["url"]).endswith(r["fileUrl"])
        ]
        text = src.pdf(ref)
        # 两版招募书均只列法律全名，未印代码；份额绑定来自官网实际公告目录的fundCode数组。
        # 不能把PDF中没有代码误判为缺原件，也不借另一产品的结转条款。
        if (
            len(records_matching) != 1
            or published > AS_OF
            or PEER_LEGAL_NAME not in text
            or "当日收益均参与下一日" not in text
            or "小数点后四位" not in text
            or "截尾" not in text
            or "小数点后三位" not in text
            or "最近第i公历日" not in text
        ):
            fail("LEGAL_REINVESTMENT_OR_PRECISION")


def _chart(src: Originals, ref: str, top_type: str) -> dict[str, D]:
    url = urlparse(src.sources[ref]["url"])
    query = parse_qs(url.query)
    if (
        url.hostname != "www.icbcubs.com.cn"
        or url.path != "/gyrx-product/app/fund/netValue/lineChart/482002"
        or query.get("topType") != [top_type]
        or query.get("buttonType") != ["15"]
    ):
        fail("OFFICIAL_RESPONSE_REQUEST_BINDING")
    raw = src.json(ref)
    if raw.get("code") != "0000" or not isinstance(raw.get("data"), list) or not raw["data"]:
        fail("OFFICIAL_RESPONSE_FAILURE_OR_EMPTY")
    rows = {}
    for row in raw["data"]:
        day = date.fromisoformat(row["lineDate"]).isoformat()
        if day in rows:
            fail("DUPLICATE_OFFICIAL_DAY")
        value = D(str(row["data"]))
        if not value.is_finite():
            fail("OFFICIAL_NONFINITE_VALUE")
        rows[day] = value
    return rows


def _reports(src: Originals, refs: list[dict]) -> list[dict]:
    expected = [
        date(
            2023 + (8 + i * 3) // 12,
            (8 + i * 3) % 12 + 1,
            calendar.monthrange(2023 + (8 + i * 3) // 12, (8 + i * 3) % 12 + 1)[1],
        ).isoformat()
        for i in range(12)
    ]
    out = []
    for item in refs:
        text = src.pdf(item["ref"])
        end, published = item["period_end"], item["published_on"]
        d = date.fromisoformat(end)
        if (
            PEER_LEGAL_NAME not in text[:6000]
            or "482002" not in text
            or not _date_in_text(end, text[:6000])
            or published > AS_OF
            or not _date_in_text(published, text[:6000])
            or f"{d.year}年第{d.month // 3}季度报告" not in text[:2000]
            or "本基金收益分配是按日结转份额" not in text
        ):
            fail("STATUTORY_REPORT_IDENTITY_OR_PERIOD")
        table = re.search(r"过去三个月([\d.]+)%[\d.]+%", text)
        cumulative = re.search(r"自基金合同生效起至今([\d.]+)%", text)
        if not table or not cumulative:
            fail("STATUTORY_RETURN_TABLE")
        before = text[: table.start()]
        if before.rfind("工银货币A") <= before.rfind("工银货币B"):
            fail("REPORT_SHARE_CLASS")
        out.append(
            {
                "period_end": end,
                "published_on": published,
                "ref": item["ref"],
                "quarter_return_percent": str(D(table[1])),
                "cumulative_percent": str(D(cumulative[1])),
            }
        )
    out.sort(key=lambda r: r["period_end"])
    if [r["period_end"] for r in out] != expected:
        fail("STATUTORY_REPORT_COVERAGE")
    return out


def _aggregate_bounds(income: dict, annual: dict, reports: list[dict]) -> tuple[list[dict], dict]:
    """自然日对数增长仅作为消元变量；不给缺失日生成任何收益值。"""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    first, last = date(2023, 9, 24), date.fromisoformat(AS_OF)
    days = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
    ix = {day: i for i, day in enumerate(days)}
    bounds = [(None, None) for _ in days]
    singleton_count = 0
    for day in days:
        previous = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        if day in income and previous in income:
            q = income[day] / 10000
            a, b = (q, q + D(".00000001")) if q >= 0 else (q - D(".00000001"), q)
            bounds[ix[day]] = (math.log1p(float(a)) * 10000, math.log1p(float(b)) * 10000)
            singleton_count += 1
    rows, limits = [], []

    def add(a, b, lo, hi, role):
        rows.extend([(a, b, 1, role), (a, b, -1, role)])
        limits.extend([hi * 10000, -lo * 10000])

    for day, q in annual.items():
        if day not in ix or ix[day] < 6:
            continue
        a, b = (q - D(".0005")) / 100, (q + D(".0005")) / 100
        add(ix[day] - 6, ix[day] + 1, math.log1p(float(a)) * 7 / 365, math.log1p(float(b)) * 7 / 365, "seven_day")
    initial = D(reports[0]["cumulative_percent"])
    il, ih = (initial - D(".00005")) / 100, (initial + D(".00005")) / 100
    for report in reports[1:]:
        value, end = D(report["cumulative_percent"]), report["period_end"]
        a, b = (value - D(".00005")) / 100, (value + D(".00005")) / 100
        add(
            ix[START] + 1,
            ix[end] + 1,
            math.log1p(float(a)) - math.log1p(float(ih)),
            math.log1p(float(b)) - math.log1p(float(il)),
            "statutory_cumulative",
        )
        d = date.fromisoformat(end)
        qstart = date(d.year, (d.month - 1) // 3 * 3 + 1, 1).isoformat()
        value = D(report["quarter_return_percent"])
        a, b = (value - D(".00005")) / 100, (value + D(".00005")) / 100
        add(ix[qstart], ix[end] + 1, math.log1p(float(a)), math.log1p(float(b)), "statutory_quarter")
    matrix = lil_matrix((len(rows), len(days)), dtype=float)
    for i, (a, b, sign, _) in enumerate(rows):
        matrix[i, a:b] = sign
    matrix = matrix.tocsr()
    tolerance = 1e-9

    def solve(objective):
        return linprog(
            objective,
            A_ub=matrix,
            b_ub=np.array(limits),
            bounds=bounds,
            method="highs",
            options={"primal_feasibility_tolerance": tolerance, "dual_feasibility_tolerance": tolerance},
        )

    feasible = solve(np.zeros(len(days)))
    if not feasible.success:
        fail("AGGREGATE_OBSERVATIONS_INFEASIBLE")
    # 对浮点求解结果向外留数值保护界；它只扩大不确定范围，不改变M2判档阈值。
    guard = max(len(rows), len(days)) * tolerance * 10 / 10000
    ends = _month_ends()
    specs = [("one_year", "2025-09-30", AS_OF), ("three_year", START, AS_OF)]
    specs += [("monthly", a, b) for a, b in zip(ends, ends[1:])]  # noqa: B905 -- 相邻日期窗口按较短序列配对，保留冻结计算语义。
    specs += [("annual", ends[i - 12], ends[i]) for i in range(12, 37)]
    projected = []
    for kind, a, b in specs:
        objective = np.zeros(len(days))
        objective[ix[a] + 1 : ix[b] + 1] = 1
        lo, hi = solve(objective), solve(-objective)
        if lo.status not in (0, 3) or hi.status not in (0, 3):
            fail("WINDOW_SOLVER_FAILED")
        projected.append(
            {
                "kind": kind,
                "start": a,
                "end": b,
                "lower_status": lo.status,
                "upper_status": hi.status,
                "return_lower": str(D(str(math.expm1(lo.fun / 10000 - guard)))) if lo.success else None,
                "return_upper": str(D(str(math.expm1(-hi.fun / 10000 + guard)))) if hi.success else None,
            }
        )
    return projected, {
        "singleton_observations": singleton_count,
        "constraint_count": len(rows),
        "numeric_log_outward_guard": str(guard),
        "statistics_constraints_applied": False,
        "missing_daily_values_generated": False,
        "feasible": True,
    }


def _G(delta: D) -> D:
    return D(75) + 25 * delta / (delta * delta + D(".001") ** 2).sqrt()


def _median(values: list[D]) -> D:
    return sorted(values)[2]


def _bucket(gap: D, band: D) -> D:
    return D(100) if gap > band else D(50) if gap < -band else D(75)


def _interval(lo: D, hi: D, **extra) -> dict:
    return {"lower": str(lo), "upper": str(hi), "candidate": str(lo) if lo == hi else None, **extra}


def _score(target: dict, known: dict, windows: list[dict]) -> dict:
    parts = {}
    for name, kind in [("one_year", "one_year"), ("three_year_annual", "three_year")]:
        row = next(r for r in windows if r["kind"] == kind)
        if row["return_lower"] is None or row["return_upper"] is None:
            fail("RETURN_REFERENCE_UNBOUNDED")
        lo, hi = D(row["return_lower"]), D(row["return_upper"])
        if kind == "three_year":
            lo, hi = ((1 + v) ** (D("365.25") / 1096) - 1 for v in (lo, hi))
        a, b = (_median([v[name] for v in known.values()] + [bound]) for bound in (lo, hi))
        parts[name] = _interval(
            _G(target[name] - b),
            _G(target[name] - a),
            peer_lower=str(lo),
            peer_upper=str(hi),
            median_lower=str(a),
            median_upper=str(b),
        )
    ret = _interval(
        D(".3") * D(parts["one_year"]["lower"]) + D(".7") * D(parts["three_year_annual"]["lower"]),
        D(".3") * D(parts["one_year"]["upper"]) + D(".7") * D(parts["three_year_annual"]["upper"]),
        parts=parts,
    )
    tables, parts = {}, {}
    for kind, band in [("monthly", D(".001") / 12), ("annual", D(".001"))]:
        projected = [r for r in windows if r["kind"] == kind]
        table = []
        for i, row in enumerate(projected):
            actual = target[kind][i]
            if (row["start"], row["end"]) != (actual["start"], actual["end"]):
                fail("COMMON_WINDOW_ALIGNMENT")
            values = [v[kind][i]["return"] for v in known.values()]
            if row["return_lower"] is None or row["return_upper"] is None:
                a, b = sorted(values)[1:3]
            else:
                a, b = (_median(values + [D(row[key])]) for key in ["return_lower", "return_upper"])
            table.append(
                _interval(
                    _bucket(actual["return"] - b, band),
                    _bucket(actual["return"] - a, band),
                    start=row["start"],
                    end=row["end"],
                    median_lower=str(a),
                    median_upper=str(b),
                )
            )
        tables[kind] = table
        parts[kind] = _interval(
            sum(D(r["lower"]) for r in table) / len(table),
            sum(D(r["upper"]) for r in table) / len(table),
            unique_windows=sum(r["candidate"] is not None for r in table),
            count=len(table),
        )
    stability = _interval(
        D(".3") * D(parts["monthly"]["lower"]) + D(".7") * D(parts["annual"]["lower"]),
        D(".3") * D(parts["monthly"]["upper"]) + D(".7") * D(parts["annual"]["upper"]),
        parts=parts,
        windows=tables,
    )
    return {"return": ret, "stability": stability}


def _confirm_nonempty_domain(income: dict, annual: dict, reports: list[dict]) -> dict:
    """以共同成立日财富基准检查原聚合约束非空，防止松弛域或空集产生伪唯一点。

    变量是累计对数财富，约束都是两个日期累计值之差的上下界。这里仅求数学
    存在性，不把见证解释为实测收益，不填补或发布缺日序列，也不改变原LP投影。
    Decimal 90位计算并向内缩1e-55，实际数据须有严格内部见证才能通过；
    它远小于原披露精度，不能扩大不确定域或改变原M2阈值。
    """
    with localcontext() as context:
        context.prec = 90
        first, last = date(2023, 9, 23), date.fromisoformat(AS_OF)
        days = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
        ix = {day: i for i, day in enumerate(days)}
        anchor, edges, observed = len(days), [], []
        inward = D("1e-55")

        def add(a, b, lo, hi):
            if not lo < hi:
                fail("AGGREGATE_EMPTY_PRECISION_INTERVAL")
            observed.append((a, b, lo, hi))
            edges.extend([(a, b, hi - inward), (b, a, -lo - inward)])

        def ln1p(value):
            if value <= -1:
                fail("AGGREGATE_LOG_DOMAIN")
            return (1 + value).ln()

        for day in days[1:]:
            previous = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
            if day in income and previous in income:
                q = income[day] / 10000
                a, b = (q, q + D("1e-8")) if q >= 0 else (q - D("1e-8"), q)
                add(ix[previous], ix[day], ln1p(a), ln1p(b))
        for day, q in annual.items():
            if day not in ix or ix[day] < 7:
                continue
            add(ix[day] - 7, ix[day], ln1p((q - D(".0005")) / 100) * 7 / 365, ln1p((q + D(".0005")) / 100) * 7 / 365)
        for report in reports:
            q = D(report["cumulative_percent"])
            add(anchor, ix[report["period_end"]], ln1p((q - D(".00005")) / 100), ln1p((q + D(".00005")) / 100))
        for report in reports[1:]:
            end = date.fromisoformat(report["period_end"])
            start = date(end.year, (end.month - 1) // 3 * 3 + 1, 1) - timedelta(days=1)
            q = D(report["quarter_return_percent"])
            add(
                ix[start.isoformat()],
                ix[report["period_end"]],
                ln1p((q - D(".00005")) / 100),
                ln1p((q + D(".00005")) / 100),
            )
        potential = [D(0)] * (len(days) + 1)
        for _iteration in range(len(potential)):
            changed = False
            for a, b, bound in edges:
                if potential[b] > potential[a] + bound:
                    potential[b] = potential[a] + bound
                    changed = True
            if not changed:
                break
        else:
            fail("FULL_COMMON_BASELINE_DOMAIN_INFEASIBLE")
        slack = min(
            (min(potential[b] - potential[a] - lo, hi - (potential[b] - potential[a])) for a, b, lo, hi in observed),
            default=D(0),
        )
        if slack <= 0:
            fail("FULL_COMMON_BASELINE_WITNESS_NOT_STRICT")
        return {
            "full_common_baseline_domain_nonempty": True,
            "interval_constraints": len(observed),
            "precision_digits": 90,
            "minimum_original_interval_slack": str(slack),
            "daily_observation_imputation_used": False,
        }


def calculate(inputs: dict, *, root: Path) -> dict:
    """公开纯计算入口；所有原件每次逐byte哈希核对，只有解析/复算结果可缓存。"""
    Originals(inputs, root)
    return deepcopy(_calculate_cached(canonical(inputs), str(Path(root).resolve())))


@lru_cache(maxsize=8)
def _calculate_cached(serialized: str, root_name: str) -> dict:
    inputs, root = json.loads(serialized), Path(root_name)
    src = Originals(inputs, root)
    _rules(inputs, src)
    peer = inputs["incomplete_peer"]
    if set(inputs["peer_income_refs"]) != set(PEERS[:-1]) or peer.get("code") != "482002":
        fail("FIXED_PEER_BINDING")
    _contracts(src, peer)
    with localcontext() as context:
        context.prec = 80
        target = _wealth(_efunds(src, inputs["target_income_refs"]))
        known = {p: _wealth(_eastmoney(src, inputs["peer_income_refs"][p], p)) for p in PEERS[:-1]}
        peer_raw = _eastmoney(src, peer["identity_income_ref"], "482002")
        income, seven = _chart(src, peer["income_ref"], "4"), _chart(src, peer["seven_day_ref"], "3")
        for day, value in income.items():
            if START <= day <= AS_OF and (day not in peer_raw or value != peer_raw[day]):
                fail("OFFICIAL_PEER_CROSS_CHECK")
        reports = _reports(src, peer["quarterly_reports"])
        domain = _confirm_nonempty_domain(income, seven, reports)
        windows, solver = _aggregate_bounds(income, seven, reports)
        solver["nonempty_domain_proof"] = domain
        dimensions = _score(target, known, windows)
        return {
            "version": VERSION,
            "fund_code": "000009",
            "as_of_date": AS_OF,
            "dimensions": dimensions,
            "fixed_peers": list(PEERS),
            "window_bounds": windows,
            "solver": solver,
            "report_observations": reports,
            "source_hashes": sorted(src.sources),
            "target_primary_ref": inputs["primary_ref"],
            "actual_complete_income_counts": {
                "000009": target["actual_record_count"],
                **{p: v["actual_record_count"] for p, v in known.items()},
            },
            "unknown_peer_observed_records": sum(START <= d <= AS_OF for d in income),
            "complete_score": None,
            "grade": None,
            "bounds_midpoint_used": False,
            "missing_day_imputation_used": False,
            "statistics_assumption_used": False,
        }


# 以下为原件包的标准prepare/snapshot薄适配，不依赖worker模块或绝对路径。
KEY = "money_aggregate_components_calculation"
UNCHANGED = ("risk", "execution", "management", "holdings", "liquidity", "fee")


def archive_json(root: Path, value: dict) -> dict:
    """按实际UTF-8字节追加不可变文件；此函数不写队列指针或业务表。"""
    raw = canonical(value).encode("utf-8")
    digest = content_digest(raw)
    relative = f"sources/money_m2/{digest}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if location_exists(path):
        if not same_content(source_bytes(path), raw):
            fail("IMMUTABLE_ARCHIVE_CHANGED")
    else:
        _sources.write_source(path, raw)
    return {"file": relative, "sha256": digest, "kind": "json"}


def _read(root: Path, ref: dict) -> dict:
    root = root.resolve()
    path = (root / ref["file"]).resolve()
    if not path.is_relative_to(root / "sources") or not source_exists(path) or not 0 < source_size(path) <= 25_000_000:
        fail("SNAPSHOT_ARCHIVE_PATH")
    raw = source_bytes(path)
    if content_digest(raw) != ref["sha256"]:
        fail("SNAPSHOT_ARCHIVE_HASH")
    return json.loads(raw.decode("utf-8-sig"))


def _bind(bundle: dict, inputs: dict, baseline: dict) -> None:
    if (
        bundle["fund_code"] != inputs["fund_code"]
        or bundle["as_of_date"] != inputs["as_of_date"]
        or bundle["identity"]["legal_name"] != inputs["legal_name"]
        or bundle["share_class"] != inputs["share_class"]
        or bundle["original_report_archive"]["sha256"] != inputs["primary_ref"]
        or baseline["fund_code"] != bundle["fund_code"]
        or baseline["as_of_date"] != bundle["as_of_date"]
        or baseline["identity"] != bundle["identity"]
        or baseline.get("money_inputs") != bundle.get("money_inputs")
    ):
        fail("SNAPSHOT_PRODUCT_OR_BASELINE_BINDING")
    for field in (
        "verified_at",
        "share_class",
        "holding_date",
        "report_published_on",
        "original_report_archive",
        "identity_archive",
        "money_input_archive",
    ):
        if baseline.get(field) != bundle.get(field):
            fail("UNCHANGED_IDENTITY_OR_VALIDITY")
    for key in UNCHANGED:
        if baseline["reviewed_document"]["dimensions"][key] != bundle["reviewed_document"]["dimensions"][key]:
            fail("UNRELATED_DIMENSION_CHANGED")
    if baseline["reviewed_document"]["outer_weights_percent"] != bundle["reviewed_document"]["outer_weights_percent"]:
        fail("OUTER_WEIGHTS_CHANGED")


def validate_input_archive(bundle: dict, *, root: Path) -> None:
    """标准validate_sources草案钩子：输入、基线和原件均绑定同一产品/份额。"""
    raw = bundle[KEY]
    if raw.get("version") != VERSION or _read(root, raw["input_archive"]) != raw["inputs"]:
        fail("SNAPSHOT_INPUT_ARCHIVE")
    baseline = _read(root, raw["base_bundle_archive"])
    _bind(bundle, raw["inputs"], baseline)
    # 此入口也校验全部原字节；不会把一次prepare通过当成以后永远可信。
    calculate(raw["inputs"], root=root)


def replay_snapshot(bundle: dict, dimensions: dict | None = None, *, root: Path) -> dict:
    """标准calculate/历史重放草案钩子；原始观察重算后逐界拒绝点数篡改。"""
    validate_input_archive(bundle, root=root)
    actual = calculate(bundle[KEY]["inputs"], root=root)
    presented = dimensions if dimensions is not None else bundle["reviewed_document"]["dimensions"]
    for key in ("return", "stability"):
        for field in ("lower", "upper"):
            if D(presented[key][field]) != D(actual["dimensions"][key][field]):
                fail("SNAPSHOT_COMPONENT_RECALCULATION")
        if presented[key].get("candidate") != actual["dimensions"][key]["candidate"]:
            fail("SNAPSHOT_COMPONENT_POINT_CHANGED")
    if (
        actual["dimensions"]["stability"]["candidate"] is not None
        or actual["complete_score"] is not None
        or actual["statistics_assumption_used"]
    ):
        fail("SNAPSHOT_UNKNOWN_SCOPE_PROMOTED")
    return actual


def prepare(
    request: dict, *, current_bundle: dict, current_ref: str, root: Path, frozen: datetime
) -> tuple[dict, dict]:
    """接收同一事务读得的当前bundle及批次；调用方发布时仍须执行当前指针CAS检查。"""
    if (
        request.get("fund_code") != "000009"
        or request.get("expected_current_ref") != str(current_ref)
        or set(request) != {"fund_code", "expected_current_ref", "input_archive"}
    ):
        fail("PREPARE_CURRENT_REF_OR_REQUEST")
    inputs = _read(root, request["input_archive"])
    # 只允许在当前五个已确认分项上做这次增量；历史snapshot回放不使用此新增前置。
    dimensions = current_bundle["reviewed_document"]["dimensions"]
    existing = {
        key for key, value in dimensions.items() if isinstance(value, dict) and value.get("candidate") is not None
    }
    if existing != {"risk", "execution", "management", "liquidity", "fee"}:
        fail("PREPARE_EXISTING_FIVE_POINTS")
    actual = calculate(inputs, root=root)
    bundle = deepcopy(current_bundle)
    baseline_ref = archive_json(root, current_bundle)
    doc = deepcopy(bundle["reviewed_document"])
    for key in ("return", "stability"):
        dim = actual["dimensions"][key]
        doc["dimensions"][key] = {
            **doc["dimensions"][key],
            **{k: dim[k] for k in ("lower", "upper", "candidate")},
            "reason": "固定五产品参照及原M2量尺；按真实收益和原披露精度传播聚合区间，不补缺日。",
            "bound_endpoints_are_not_imputed_scores": dim["candidate"] is None,
            "new_evidence": request["input_archive"],
            "method_status": "M2原公式不变，逐byte原件编译/聚合区间/中位数唯一性回放",
            "subcomponents": dim.get("parts", {}),
        }
    doc["changed_dimensions"] = ["return", "stability"]
    doc["unchanged_dimensions"] = list(UNCHANGED)
    doc["finalized_at"] = frozen.isoformat()
    doc["remaining_gaps"] = [
        "稳定性仍缺482002于2024-11-30的单日收益或同口径可唯一归属原件；不使用未核标准差日历。",
        "持仓最大同银行存款+存单敞口及未评级短融/中票信用贡献未唯一。",
    ]
    bundle["reviewed_document"] = doc
    bundle[KEY] = {
        "version": VERSION,
        "input_archive": request["input_archive"],
        "inputs": inputs,
        "base_bundle_archive": baseline_ref,
        "expected_previous_ref": str(current_ref),
    }
    # 原五个点、持仓区间及其有效期保持原值；本次研究时间只记在doc.finalized_at。
    bundle["public_explanations"] = {
        **bundle.get("public_explanations", {}),
        "return": "按同一评价期实际收益比较；已确定该项得分。",
        "stability": "部分跨月收益尚不能唯一确定，保留分数范围。",
    }
    # source/read/calculate均是调用方传入root上的纯回放；没有register_update/publish调用。
    _bind(bundle, inputs, current_bundle)
    replay_snapshot(bundle, root=root)
    return bundle, actual


def materialize_document(bundle: dict, *, root: Path, arithmetic: dict) -> dict:
    """算术通过后封存新研究文档；旧原件和旧研究哈希均保持，只追加新引用。"""
    doc = deepcopy(bundle["reviewed_document"])
    doc["research_score"] = {
        "point_weight_percent": int(arithmetic["point_weight"]),
        "point_dimension_count": arithmetic["point_dimensions"],
        "informative_interval_weight_percent": int(arithmetic["interval_weight"]),
        "missing_weight_percent": int(arithmetic["missing_weight"]),
        "partial_research_score": arithmetic["partial_score"],
        "complete_point_score": None,
        "grade": None,
        "interval_midpoint_used": False,
        "eight_dimension_interval": {"lower": arithmetic["lower"], "upper": arithmetic["upper"]},
    }
    bundle = deepcopy(bundle)
    bundle["reviewed_document"] = doc
    bundle["research_archive"] = archive_json(root, doc)
    extra = {r["sha256"]: r for r in bundle.get("extra_archives", [])}
    for digest, source in bundle[KEY]["inputs"]["sources"].items():
        extra[digest] = {"file": source["file"], "sha256": digest, "kind": source["kind"]}
    for ref in [bundle[KEY]["input_archive"], bundle[KEY]["base_bundle_archive"]]:
        extra[ref["sha256"]] = ref
    bundle["extra_archives"] = list(extra.values())
    return bundle


def prepare_dossier(
    request: dict,
    *,
    current_bundle: dict,
    current_ref: str,
    root: Path,
    frozen: datetime,
    calculate_snapshot,
    validate_snapshot_sources,
) -> dict:
    """pipeline prepare草案：调用已有标准纯校验/合成，返回单基金清单但不登记。

    两个回调由root传入当前标准模块，避免本候选耦合共享服务的导入路径。
    root正式发布时应在既有基金锁内比对expected_previous_ref，防止排队期间覆盖新批次。
    """
    bundle, actual = prepare(request, current_bundle=current_bundle, current_ref=current_ref, root=root, frozen=frozen)
    arithmetic = calculate_snapshot(bundle, root=root)
    if (
        arithmetic["point_dimensions"] != 6
        or D(arithmetic["point_weight"]) != 80
        or arithmetic["score"] is not None
        or arithmetic["grade"] is not None
    ):
        fail("PREPARE_PARTIAL_RESULT_SCOPE")
    bundle = materialize_document(bundle, root=root, arithmetic=arithmetic)
    validate_snapshot_sources(bundle, frozen, root=root)
    replay_snapshot(bundle, root=root)
    entry = {key: value for key, value in bundle.items() if key != "reviewed_document"}
    return {"basis": bundle["basis"], "entries": [entry]}


def check_binding(bundle: dict, inputs: dict | None = None, *, root: Path) -> None:
    """公开薄入口：当前产品、份额、旧证据及有效期绑定，随后校全部真实原字节。

    expected_previous_ref由发布方在同一事务锁内与最新指针核对，本函数不能替代CAS。
    """
    if inputs is not None and inputs != bundle[KEY]["inputs"]:
        fail("SNAPSHOT_TYPED_INPUT_CHANGED")
    validate_input_archive(bundle, root=root)
