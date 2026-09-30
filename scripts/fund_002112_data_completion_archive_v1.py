"""按真实文件格式处理附件和压缩包，不执行宏，不信任服务器 MIME 或包内路径。"""
import io
import json
import hashlib
import zipfile
from pathlib import Path
from scripts.fund_002112_data_completion_office_v1 import OUT,parse

def read(p): return json.loads(Path(p).read_text('utf-8'))
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,default=str)

def run():
    for p in (OUT/'receipts').glob('*.json'):
        r=read(p)
        if not r.get('ok') or r.get('group')!='attachments':continue
        dest=OUT/'attachment-format-audit'/p.name
        if dest.exists():continue
        raw=Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
        out={'receipt':r,'bytes_verified':True,'parsed':False,'gaps':[]}
        try:
            if zipfile.is_zipfile(io.BytesIO(raw)):
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    infos=[x for x in z.infolist() if not x.is_dir()];names={x.filename for x in infos}
                    if 'word/document.xml' in names or 'xl/workbook.xml' in names:
                        office=OUT/'office-attachments'/p.name
                        if not office.exists():save(office,{'source':r,'parsed':True,**parse(raw,''),'training_eligible':False})
                        out.update(format='OFFICE_ZIP',parsed=True,parsed_path=str(office))
                    else:
                        assert len(infos)<=10000 and sum(x.file_size for x in infos)<=512*1024**2,'ZIP_SIZE_LIMIT'
                        members=[]
                        for info in infos:
                            assert not info.flag_bits&1,'ENCRYPTED_ZIP_MEMBER'
                            data=z.read(info);digest=hashlib.sha256(data).hexdigest();key=hashlib.sha256((r['url']+'#archive-member='+info.filename).encode()).hexdigest()
                            path=OUT/'archive-members'/digest
                            if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                            mime='application/pdf' if data.startswith(b'%PDF') else 'application/vnd.ms-excel' if info.filename.lower().endswith(('.xls','.xlsx')) else 'application/msword'
                            receipt={'url':r['url']+'#archive-member='+info.filename,'group':'attachments','ok':True,'status':200,'headers':{'content-type':mime},'path':str(path),'sha256':digest,'bytes':len(data),'archive_parent':r['sha256'],'member_original_name':info.filename,'network_request':False}
                            rp=OUT/'asset-receipts'/(key+'.json')
                            if not rp.exists():save(rp,receipt)
                            member={'name':info.filename,'crc32':info.CRC,'bytes':len(data),'sha256':digest,'receipt_path':str(rp)}
                            if mime!='application/pdf':
                                op=OUT/'office-attachments'/(key+'.json')
                                if not op.exists():
                                    try:save(op,{'source':receipt,'parsed':True,**parse(data,mime),'training_eligible':False})
                                    except Exception as e:save(op,{'source':receipt,'parsed':False,'error':type(e).__name__+':'+str(e)})
                                member['parse_path']=str(op)
                            members.append(member)
                        out.update(format='ZIP_ARCHIVE',members=members,all_members_crc_verified=True,parsed=True,member_semantics_verified=False)
            elif raw.startswith(b'%PDF'):out.update(format='PDF',parsed=True,parsed_path=str(OUT/'policy-assets'/p.name))
            elif raw.startswith(bytes.fromhex('d0cf11e0a1b11ae1')):
                import olefile
                with olefile.OleFileIO(io.BytesIO(raw)) as ole:mime='application/vnd.ms-excel' if ole.exists('Workbook') or ole.exists('Book') else 'application/msword'
                op=OUT/'office-attachments'/p.name
                if not op.exists():save(op,{'source':r,'parsed':True,**parse(raw,mime),'training_eligible':False})
                out.update(format='OLE_OFFICE',parsed=True,parsed_path=str(op))
            else:out['gaps'].append('UNSUPPORTED_MAGIC_OR_HTML_RESPONSE')
        except Exception as e:out['gaps'].append(type(e).__name__+':'+str(e))
        save(dest,out)
        print(json.dumps({'attachment':p.stem,'format':out.get('format'),'members':len(out.get('members',[])),'gaps':out['gaps']}),flush=True)

if __name__=='__main__':run()
