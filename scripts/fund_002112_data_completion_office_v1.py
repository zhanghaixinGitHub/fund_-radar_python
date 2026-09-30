"""用随桌面提供的 Python 只读提取政策附件，保留单元格坐标、空值和公式。"""
import hashlib
import io
import json
import struct
import sys
import time
import zipfile
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'.local-runs/fund-exposure-002112/data-completion/20260930-v1'
sys.path.insert(0,str(OUT/'legacy-deps'))

def legacy_word(raw):
    """按微软 MS-DOC 的 FIB、Clx、PlcPcd 顺序读文本；不执行宏或外链。"""
    import olefile
    with olefile.OleFileIO(io.BytesIO(raw)) as ole:
        word=ole.openstream('WordDocument').read()
        flags=struct.unpack_from('<H',word,10)[0]
        if flags&0x100:raise ValueError('ENCRYPTED_WORD_NOT_SUPPORTED')
        table=ole.openstream('1Table' if flags&0x200 else '0Table').read()
        fc,size=struct.unpack_from('<II',word,0x1a2)
        clx=table[fc:fc+size];pos=0
        while pos<len(clx) and clx[pos]==1:pos+=3+struct.unpack_from('<H',clx,pos+1)[0]
        if pos>=len(clx) or clx[pos]!=2:raise ValueError('WORD_PIECE_TABLE_MISSING')
        size=struct.unpack_from('<I',clx,pos+1)[0];plc=clx[pos+5:pos+5+size]
        if len(plc)!=size or (size-4)%12:raise ValueError('WORD_PIECE_TABLE_INVALID')
        n=(size-4)//12;cp=struct.unpack_from('<'+str(n+1)+'I',plc,0);parts=[]
        if any(b<=a for a,b in zip(cp,cp[1:])):raise ValueError('WORD_CHARACTER_ORDER_INVALID')
        for i in range(n):
            encoded=struct.unpack_from('<I',plc,4*(n+1)+8*i+2)[0]
            compressed=bool(encoded&0x40000000);offset=encoded&0x3fffffff
            length=cp[i+1]-cp[i]
            if compressed:offset//=2
            piece=word[offset:offset+length*(1 if compressed else 2)]
            if len(piece)!=length*(1 if compressed else 2):raise ValueError('WORD_PIECE_TRUNCATED')
            parts.append(piece.decode('cp1252' if compressed else 'utf-16-le',errors='strict'))
        return ''.join(parts).replace('\r','\n').replace('\x07','\t')

def parse(raw,mime):
    """只解析文件自身数据；表格留坐标，公式不计算、不读取外部引用。"""
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:names=set(z.namelist())
        if 'word/document.xml' in names:
            from docx import Document
            d=Document(io.BytesIO(raw))
            return {'format':'DOCX','paragraphs':[p.text for p in d.paragraphs],
                'tables':[[[c.text for c in row.cells] for row in t.rows] for t in d.tables],
                'metadata':{'created':str(d.core_properties.created),'modified':str(d.core_properties.modified)},'layout_verified':False}
        if 'xl/workbook.xml' in names:
            from openpyxl import load_workbook
            w=load_workbook(io.BytesIO(raw),read_only=True,data_only=False,keep_links=False)
            out={'format':'XLSX','sheets':[],'metadata':{'created':str(w.properties.created),'modified':str(w.properties.modified)}}
            for s in w:
                if s.max_row*s.max_column>2_000_000:raise ValueError('WORKSHEET_BOUNDED_SIZE_LIMIT')
                out['sheets'].append({'name':s.title,'rows':s.max_row,'columns':s.max_column,
                    'cells':[{'coordinate':c.coordinate,'value':c.value,'type':c.data_type} for row in s for c in row if c.value is not None]})
            w.close();return out
    if raw.startswith(bytes.fromhex('d0cf11e0a1b11ae1')):
        if 'excel' in mime:
            import xlrd
            w=xlrd.open_workbook(file_contents=raw)
            return {'format':'XLS','sheets':[{'name':s.name,'rows':s.nrows,'columns':s.ncols,'cells':[
                {'row':i+1,'column':j+1,'value':s.cell_value(i,j),'type':s.cell_type(i,j)}
                for i in range(s.nrows) for j in range(s.ncols) if s.cell_type(i,j) not in (0,6)]} for s in w.sheets()],
                'formula_semantics_verified':False}
        return {'format':'DOC','text':legacy_word(raw),'layout_verified':False,'parser_spec':'https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-doc/01d5d8c4-cf9c-4ef9-80fd-439e763cfe01'}
    raise ValueError('UNRECOGNIZED_OFFICE_OR_ERROR_RESPONSE')

def run():
    while True:
        for p in (OUT/'receipts').glob('*.json'):
            r=json.loads(p.read_text(encoding='utf-8'));mime=r.get('headers',{}).get('content-type','')
            if r.get('group')!='attachments' or not r.get('ok') or not any(x in mime for x in ('word','excel','spreadsheet')):continue
            dest=OUT/'office-attachments'/p.name
            if dest.exists():continue
            raw=Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
            d={'source':r,'semantic_verified':False,'training_eligible':False}
            try:d.update(parse(raw,mime),parsed=True)
            except Exception as e:d.update(parsed=False,error=type(e).__name__+':'+str(e))
            dest.parent.mkdir(exist_ok=True)
            with dest.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,default=str)
            print(json.dumps({'office':p.stem,'parsed':d['parsed'],'format':d.get('format') or d.get('error')}),flush=True)
        if (OUT/'public-recovery-complete.json').exists():break
        time.sleep(5)

if __name__=='__main__':run()
