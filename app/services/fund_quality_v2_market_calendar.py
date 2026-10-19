"""核原件休市事实并引用最近实际指数收盘；不创造收盘或人民币基准收益。"""

import re
from bisect import bisect_left
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

VERSION = 'QUALITY_MARKET_CLOSURE_20261017_1'
CALENDAR = 'component-market-calendar'
WEATHER_RULE = 'component-weather-rule'
WEATHER_TIMELINE = 'component-weather-timeline'
ROLES = {CALENDAR, WEATHER_RULE, WEATHER_TIMELINE}
CLOSED = 'Securities Market is closed on Saturdays, Sundays and Public Holidays as specified below:'
HALF = 'The following trading days are half-day trading day and non-settlement day:'
MONTH = '(?:January|February|March|April|May|June|July|August|September|October|November|December)'
WEEK = '(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'


def calendar_endpoint(ref: dict) -> None:
    """只适配现有港交所四年证券通告及2024-09-06天气的两个实际官网端点。"""
    address = urlsplit(ref['url'])
    if (address.scheme != 'https' or address.username or address.password or address.fragment
            or address.port not in {None, 443} or ref.get('label') not in ROLES):
        raise ValueError('CLOSURE_ORIGINAL_ENDPOINT')
    if ref['label'] == CALENDAR:
        match = re.fullmatch(r'/-/media/HKEX-Market/Services/Circulars-and-Notices/'
            r'Participant-and-Members-Circulars/SEHK/(20[0-9]{2})/ce_SEHK_CT_[0-9]{3}_\1\.pdf', address.path)
        year = str(ref.get('calendar_year', ''))
        valid = (address.hostname == 'www.hkex.com.hk' and not address.query and match
                 and re.fullmatch('[0-9]{4}', year) and int(year) == int(match[1])+1)
    elif ref['label'] == WEATHER_RULE:
        valid = (address.hostname == 'www.hkex.com.hk'
                 and address.path == '/News/Market-Communications/2024/240906news'
                 and parse_qs(address.query, keep_blank_values=True) == {'sc_lang': ['en']})
    else:
        valid = (address.hostname == 'www.hko.gov.hk'
                 and address.path == '/en/publica/tc/tc2024/section3_3rpt.html' and not address.query)
    if not valid:
        raise ValueError('CLOSURE_ORIGINAL_ENDPOINT_YEAR_OR_QUERY')


def holiday_circular(pages: list[str], *, year: int, as_of: date) -> dict:
    """从整日假日表读取日期和星期，再独立读取半日市；第二页续行不能遗漏。"""
    if not 1 <= len(pages) <= 4:
        raise ValueError('CLOSURE_CIRCULAR_PAGE_COUNT')
    text = re.sub(r'\s+', ' ', '\n'.join(pages))
    if (f'Subject: Hong Kong Securities Market Holiday Schedule for Year {year}' not in text
            or 'THE STOCK EXCHANGE OF HONG KONG LIMITED' not in text or text.count(CLOSED) != 1):
        raise ValueError('CLOSURE_CIRCULAR_TITLE_MARKET_OR_TABLE')
    publications = set(re.findall(r'Date:\s*([0-9]{2}/[0-9]{2}/20[0-9]{2})', text))
    if len(publications) != 1:
        raise ValueError('CLOSURE_CIRCULAR_PUBLICATION_NOT_UNIQUE')
    published = datetime.strptime(publications.pop(), '%d/%m/%Y').date()
    if published > as_of:
        raise ValueError('CLOSURE_CIRCULAR_NOT_AVAILABLE_BY_CUTOFF')
    sections = text.split(CLOSED, 1)[1].split(HALF)
    if len(sections) > 2:
        raise ValueError('CLOSURE_HALF_DAY_SECTION_NOT_UNIQUE')
    pattern = re.compile(r'\b([0-9]{1,2} '+MONTH+' '+str(year)+r')\s+('+WEEK+r')')
    holidays = []
    for match in pattern.finditer(sections[0]):
        moment = datetime.strptime(match[1], '%d %B %Y')
        if moment.strftime('%A') != match[2]:
            raise ValueError('CLOSURE_CIRCULAR_WEEKDAY_CONFLICT')
        holidays.append({'date': str(moment.date()), 'excerpt': match[0]})
    if not holidays or len({r['date'] for r in holidays}) != len(holidays):
        raise ValueError('CLOSURE_CIRCULAR_EMPTY_OR_DUPLICATE_HOLIDAY')
    half_days = []
    if len(sections) == 2:
        half_pattern = re.compile(r'\b([0-9]{1,2} '+MONTH+' '+str(year)+r')\s*\(('+WEEK+r')\)')
        for match in half_pattern.finditer(sections[1]):
            moment = datetime.strptime(match[1], '%d %B %Y')
            if moment.strftime('%A') != match[2]:
                raise ValueError('CLOSURE_HALF_DAY_WEEKDAY_CONFLICT')
            half_days.append(str(moment.date()))
        if not half_days or len(set(half_days)) != len(half_days):
            raise ValueError('CLOSURE_HALF_DAY_ROWS_UNPARSED')
    if set(half_days) & {r['date'] for r in holidays}:
        raise ValueError('CLOSURE_FULL_AND_HALF_DAY_CONFLICT')
    return {'year': year, 'document_dated_on': str(published), 'holiday_rows': holidays,
            'half_days': half_days, 'weekend_clause': CLOSED,
            'scope': 'ACTUAL_DATED_HKEX_CIRCULAR_NOT_FIRST_HTTP_VERSION_PROOF'}


def weather_closure(rule: bytes, timeline: bytes) -> dict:
    """只推导2024-09-06：中午截止规则＋实际12:40降信号；不倒用9月23日新制。"""
    texts = [re.sub(r'\s+', ' ', BeautifulSoup(raw, 'html.parser').get_text(' ', strip=True))
             for raw in (rule, timeline)]
    rule_excerpt = ('If Typhoon Signal No. 8 or above, Black Rainstorm Warning or any announcement of Extreme '
        'Conditions, remains issued at 12:00 noon, all trading sessions today will be cancelled.')
    actual = ('The No. 8 Northeast Gale or Storm Signal was replaced by the No. 3 Strong Wind Signal '
              'at 12:40 p.m. on 6 September.')
    if (rule_excerpt not in texts[0] or '06 Sep 2024' not in texts[0]
            or 'Securities and Derivatives Market Trading Arrangements under Typhoon Signal No. 8' not in texts[0]
            or actual not in texts[1] or 'Super Typhoon Yagi (2411)' not in texts[1]):
        raise ValueError('CLOSURE_WEATHER_ACTUAL_PREMISES_NOT_BOUND')
    return {'date': '2024-09-06', 'state': 'WEATHER_CLOSURE_INFERRED_FROM_TWO_PRIMARY_PREMISES',
            'premises': [rule_excerpt, actual], 'inference': 'SIGNAL_LOWERED_AFTER_NOON_CANCELLATION_CUTOFF',
            'not_a_direct_final_cancellation_announcement': True}


def closure_evidence(*, refs: list[dict], root: Path, as_of: date, frozen: datetime) -> dict:
    """再次核各原件及真实收据，错误按件保留，不让坏日历抹掉已有真实指数价格。"""
    from app.services.fund_quality_v2_component_prices import _verified_original

    calendars, weather, rejected = {}, {}, []
    conflicting_years, conflicting_weather_roles = set(), set()
    for ref in refs:
        if ref.get('label') not in ROLES:
            continue
        try:
            calendar_endpoint(ref)
            raw, receipt_sha = _verified_original(ref, refs, root, frozen)
            hashes = [ref['sha256'], receipt_sha]
            if ref['label'] == CALENDAR:
                reader = PdfReader(BytesIO(raw))
                parsed = holiday_circular([page.extract_text() for page in reader.pages],
                    year=int(ref['calendar_year']), as_of=as_of)
                year = parsed['year']
                if year in calendars or year in conflicting_years:
                    calendars.pop(year, None)
                    conflicting_years.add(year)
                    raise ValueError('CLOSURE_CALENDAR_YEAR_NOT_UNIQUE')
                calendars[year] = parsed | {'source_hashes': hashes, 'source_url': ref['url']}
            else:
                if ref['label'] in weather or ref['label'] in conflicting_weather_roles:
                    weather.pop(ref['label'], None)
                    conflicting_weather_roles.add(ref['label'])
                    raise ValueError('CLOSURE_WEATHER_ROLE_NOT_UNIQUE')
                weather[ref['label']] = {'raw': raw, 'source_hashes': hashes, 'source_url': ref['url']}
        except (ValueError, KeyError, OSError, TypeError, PdfReadError) as exc:
            rejected.append({'source_hashes': [ref.get('sha256')], 'source_url': ref.get('url'), 'reason': str(exc)})
    by_day = {}
    for item in calendars.values():
        for row in item['holiday_rows']:
            by_day[row['date']] = row | {'state': 'HKEX_EXPLICIT_FULL_DAY_HOLIDAY',
                                        'source_hashes': item['source_hashes']}
    if WEATHER_RULE in weather and WEATHER_TIMELINE in weather and as_of >= date(2024, 9, 6):
        try:
            event = weather_closure(weather[WEATHER_RULE]['raw'], weather[WEATHER_TIMELINE]['raw'])
            by_day[event['date']] = event | {'source_hashes': sorted({h for w in weather.values()
                                                                    for h in w['source_hashes']})}
        except ValueError as exc:
            rejected.append({'reason': str(exc), 'source_hashes': sorted({h for w in weather.values()
                                                                        for h in w['source_hashes']})})
    # JSON归档会把整数键转为字符串；在生成时统一，避免再次核验时因键类型不同失败。
    return {'version': VERSION, 'calendars': {str(k): v for k, v in calendars.items()},
            'explicit_closed_dates': by_day, 'rejected_originals': rejected,
            'conflicting_calendar_years': sorted(conflicting_years),
            'conflicting_weather_roles': sorted(conflicting_weather_roles)}


def price_references(component: dict, required_dates: list[str], evidence: dict) -> dict:
    """给每个基金估值日定位实际收盘或带休市依据的最近收盘引用，保持两类字段不同。

    仅在前一真实收盘至目标日的每个间隔日都已证实休市时允许引用。若中间
    有一个未解释的工作日、缺年度日历、半日市缺价格或反向日期，保留缺口。
    不产生当天新收盘，不假设人民币收益为零，历史无基金日历也不能报成功。
    """
    if not required_dates:
        return {'version': VERSION, 'status': 'NO_VERIFIED_FUND_VALUATION_CALENDAR', 'rows': [],
                'unresolved_dates': [], 'new_price_observations_created': 0, 'complete_cny_benchmark': False}
    if required_dates != sorted(set(required_dates)):
        raise ValueError('CLOSURE_REQUIRED_DATES_ORDER')
    points = dict(component['prices'])
    predecessor = component.get('window_predecessor_observation')
    if predecessor:
        points[predecessor['date']] = predecessor['close']
    days = sorted(points)
    price_hashes = component['source_hashes']

    def closed(day: str) -> dict | None:
        if day in evidence['explicit_closed_dates']:
            return evidence['explicit_closed_dates'][day]
        moment = date.fromisoformat(day)
        calendar = evidence['calendars'].get(str(moment.year))
        if moment.weekday() >= 5 and calendar:
            return {'date': day, 'state': 'WEEKEND_CLOSED_BY_HKEX_CIRCULAR',
                    'source_hashes': calendar['source_hashes'], 'excerpt': calendar['weekend_clause']}
        return None

    rows, unresolved = [], []
    for day in required_dates:
        date.fromisoformat(day)
        if day in points:
            if closed(day):
                row = {'date': day, 'state': 'PRICE_PRESENT_ON_DOCUMENTED_CLOSED_DAY_REVIEW_REQUIRED'}
                unresolved.append(day)
            else:
                row = {'date': day, 'state': 'ACTUAL_REPORTED_CLOSE', 'actual_close_date': day,
                       'actual_close': points[day], 'source_hashes': price_hashes}
        else:
            index = bisect_left(days, day)-1
            previous = days[index] if index >= 0 else None
            proof = closed(day)
            gap = (date.fromisoformat(day)-date.fromisoformat(previous)).days if previous else None
            interval_proofs = []
            if proof and gap is not None and 0 < gap <= 31:
                for offset in range(1, gap+1):
                    intervening = str(date.fromisoformat(previous)+timedelta(days=offset))
                    item = closed(intervening)
                    if item is None:
                        interval_proofs = []
                        break
                    interval_proofs.append(item)
            if not interval_proofs:
                row = {'date': day, 'state': 'MISSING_CLOSE_OR_INTERVENING_OPEN_DAY_UNRESOLVED',
                       'previous_observed_close_date': previous}
                unresolved.append(day)
            else:
                row = {'date': day, 'state': 'CLOSED_MARKET_LAST_ACTUAL_CLOSE_REFERENCE',
                    'observed_close_on_valuation_day': None, 'actual_close_date': previous,
                    'reference_close': points[previous], 'closure_evidence': interval_proofs,
                    'source_hashes': sorted(set(price_hashes + [h for p in interval_proofs
                                                              for h in p['source_hashes']])),
                    'scope': 'DATED_REFERENCE_NOT_NEW_CLOSE_OR_ASSUMED_ZERO_CNY_RETURN'}
        rows.append(row)
    return {'version': VERSION, 'status': 'ALL_REQUESTED_DATES_EXPLAINED' if not unresolved
                else 'CLOSURE_OR_PRICE_GAPS_REMAIN', 'rows': rows, 'unresolved_dates': unresolved,
            'actual_close_rows': sum(r['state'] == 'ACTUAL_REPORTED_CLOSE' for r in rows),
            'closed_reference_rows': sum(r['state'] == 'CLOSED_MARKET_LAST_ACTUAL_CLOSE_REFERENCE' for r in rows),
            'new_price_observations_created': 0, 'complete_cny_benchmark': False}
