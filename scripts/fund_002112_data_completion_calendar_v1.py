"""补齐历史交易日依据并纠正覆盖报告；不触碰服务使用的交易日文件。"""
import re
from datetime import date, timedelta
from bs4 import BeautifulSoup
from scripts.fund_002112_data_completion_v1 import OUT, read, save, fetch, now

SOURCES = {
    2016: 'https://www.sse.com.cn/disclosure/announcement/general/c/c_20151224_4027676.shtml',
    2017: 'https://www.sse.com.cn/disclosure/announcement/general/c/c_20161222_4218613.shtml',
    2018: 'https://www.sse.com.cn/disclosure/announcement/general/c/c_20171222_4438363.shtml',
    2019: 'https://www.sse.com.cn/disclosure/announcement/general/c/c_20181220_4696473.shtml',
    2020: 'https://www.sse.com.cn/disclosure/announcement/general/c/c_20191220_4969627.shtml',
}
# 每个区间逐字来自交易所全文；周末始终休市，国务院调休上班不等于开市。
HOLIDAYS = {
    2016: [('0101','0103'),('0207','0213'),('0402','0404'),('0430','0502'),('0609','0611'),('0915','0917'),('1001','1007')],
    2017: [('0101','0102'),('0127','0202'),('0402','0404'),('0429','0501'),('0528','0530'),('1001','1008')],
    2018: [('0101','0101'),('0215','0221'),('0405','0407'),('0429','0501'),('0616','0618'),('0922','0924'),('1001','1007'),('1231','1231')],
    2019: [('0101','0101'),('0204','0210'),('0405','0407'),('0501','0504'),('0607','0609'),('0913','0915'),('1001','1007')],
    2020: [('0101','0101'),('0124','0202'),('0404','0406'),('0501','0505'),('0625','0627'),('1001','1008')],
}
EXTRA = [
    ('2019劳动节调整','https://www.sse.com.cn/disclosure/announcement/general/c/c_20190418_4771364.shtml'),
    ('2020春节延长','https://www.sse.com.cn/disclosure/announcement/general/c/c_20200127_4991582.shtml'),
]

def run():
    records=[]
    for label,url in list(SOURCES.items())+EXTRA:
        receipt=fetch(url,'industry',2_000_000)
        text=''
        if receipt.get('ok'):
            from pathlib import Path
            soup=BeautifulSoup(Path(receipt['path']).read_bytes(),'html.parser')
            text=soup.get_text('\n',strip=True)
        records.append({'label':label,'receipt':receipt,'text':text})
    save(OUT/'industry/calendar-official-evidence.json',{'sources':records,'at':now()})
    if not all(r['text'] and '休市' in r['text'] for r in records):
        return
    sessions=set()
    for year,ranges in HOLIDAYS.items():
        closed=set()
        for lo,hi in ranges:
            day=date.fromisoformat(f'{year}-{lo[:2]}-{lo[2:]}');end=date.fromisoformat(f'{year}-{hi[:2]}-{hi[2:]}')
            while day<=end:closed.add(day);day+=timedelta(days=1)
        day=date(year,1,1)
        while day.year==year:
            if day.weekday()<5 and day not in closed:sessions.add(str(day))
            day+=timedelta(days=1)
    from app.services.trading_calendar import load_calendar
    sessions|={str(d) for d in load_calendar().sessions if date(2021,1,1)<=d<=date(2023,12,31)}
    save(OUT/'industry/calendar-2016-2023.json',{'sessions':sorted(sessions),'at':now(),'holiday_intervals':HOLIDAYS,
        'sources':records,'calendar_only_no_price_or_label':True,'later_corrections_not_available_before_publication':True})

if __name__=='__main__':run()
