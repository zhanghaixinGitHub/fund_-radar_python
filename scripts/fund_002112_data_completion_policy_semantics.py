"""冻结政策/新闻的字面事实、版本引用和历史持仓关联；不推断市场方向。"""
import re,hashlib,sys
from pathlib import Path
from collections import Counter,defaultdict
from datetime import date,timedelta
from urllib.parse import urlparse
from scripts.fund_002112_data_completion_v1 import OUT,read,save,sha,now

NUM=re.compile(r'(?:国办发|国发|医保(?:办发|办函|发|函)|国卫(?:医函|医发|办医函|办医发)|财社|人社部发|药监综药管函)[〔\[（(](20\d{2})[〕\]）)]\s*\d{1,4}\s*号')
DAY=r'20\d{2}年\d{1,2}月\d{1,2}日'
def norm(s):return re.sub(r'\s+','',s or '').replace('（','(').replace('）',')')
def title(s):return re.sub(r'^【[^】]+】','',s).strip()
def classify(t,host):
 if any(s in t for s in ['征求意见','公示','公开征求']):return 'DRAFT_OR_PUBLIC_CONSULTATION'
 if any(s in t for s in ['解读','答记者问','一图读懂','权威解答','问答']):return 'INTERPRETATION_NOT_ORIGINAL_POLICY'
 if host.endswith('.gov.cn') and any(t.endswith(s) for s in ['通知','办法','意见','公告','通告','决定','细则','条例']):return 'OFFICIAL_POLICY_OR_NOTICE'
 if '视频' in t or '直播' in t:return 'VIDEO_OR_LIVE_REPORT'
 return 'PUBLIC_NEWS_OR_SERVICE_INFORMATION'

def asset_segments(p,seen=None):
 """保留段落/页/单元格位置；表格空单元格绝不转换为 0。"""
 p=Path(p);seen=seen or set()
 if str(p) in seen:return []
 seen.add(str(p));d=read(p);segments=[]
 if d.get('format')=='ZIP_ARCHIVE':
  for m in d.get('members',[]):
   key=Path(m['receipt_path']).stem
   q=next((OUT/f/f'{key}.json' for f in ['policy-assets-v2','office-attachments','policy-assets'] if (OUT/f/f'{key}.json').exists() and read(OUT/f/f'{key}.json').get('parsed')),None)
   if q:segments+=asset_segments(q,seen)
  return segments
 for i,text in enumerate(d.get('pages',[]),1):segments.append({'source':str(p),'locator':{'page':i},'text':text,'ocr':bool(d.get('ocr_pages'))})
 for i,text in enumerate(d.get('paragraphs',[]),1):segments.append({'source':str(p),'locator':{'paragraph':i},'text':text,'ocr':False})
 if d.get('text'):segments.append({'source':str(p),'locator':{'legacy_document_text':True},'text':d['text'],'ocr':False})
 for i,table in enumerate(d.get('tables',[]),1):
  for j,row in enumerate(table,1):segments.append({'source':str(p),'locator':{'table':i,'row':j},'text':' | '.join(str(x) if x is not None else '[原件空白]' for x in row),'ocr':False})
 # XLS/XLSX 的完整单元格坐标和公式原文留在原解析件；不在此计算公式。
 return segments

def run():
 version=sys.argv[1] if len(sys.argv)>1 else '';suffix='-'+version if version else ''
 sourceversion=sys.argv[2] if len(sys.argv)>2 else 'v7'
 visual=read(OUT/'policy-clause-visual-review.json')['rows']
 def out(name):return OUT/(name+suffix+'.json')
 rows=[];indexes=defaultdict(list);title_index=defaultdict(list);aliases=defaultdict(list);unmatched=[]
 for p in sorted((OUT/('public-audit-'+sourceversion)).glob('*.json')):
  d=read(p);t=title(d['aliases'][0]['title']);host=urlparse(d['url']).hostname or '';kind=classify(t,host)
  segments=[{'source':str(p),'locator':{'paragraph':i},'text':s,'ocr':False} for i,s in enumerate(d.get('text','').splitlines(),1) if s.strip()]
  assets=[]
  for a in d.get('attachments',[])+d.get('images',[]):
   if not a.get('required_body_asset',True):continue
   if a.get('parsed_path'):
    assets.append({'url':a['url'],'source':a['parsed_path'],'sha256':a['parsed_sha256'],'binding':'EXPLICIT_SOURCE_BODY_LINK','source_version_time_not_independently_proven':True})
    segments+=asset_segments(a['parsed_path'])
  own=[];cited=[];clauses=[];issues=[];body=norm(d.get('text',''))
  for i,s in enumerate(segments):
   q=norm(s['text']);loc={k:v for k,v in s.items() if k!='text'}
   for m in NUM.finditer(q):
    item={'number':m[0],'anchor':loc,'quote':q[max(0,m.start()-90):m.end()+150]}
    # 文号单独一行，且在正文前 12 段，才认定为本通知文号；提及文号只是引用。
    if kind=='OFFICIAL_POLICY_OR_NOTICE' and s['source']==str(p) and i<12 and q==m[0]:own.append(item)
    else:cited.append(item)
   for sentence in re.split(r'(?<=[。；;])',q):
    if not sentence:continue
    effect=re.search(r'自('+DAY+r'|印发之日|发布之日|公布之日)起?(?:实施|施行|执行|生效)',sentence)
    repeal=bool(re.search(r'(?:同时|即行|予以|一并)废止|废止[《「]',sentence))
    revision=bool(re.search(r'对[《「].{2,120}?[》」].{0,40}(?:修改|修订)|修订后的',sentence))
    if effect or repeal or revision:
     operative=kind=='OFFICIAL_POLICY_OR_NOTICE' and not any(x in sentence for x in ['征求意见','拟自','拟于','建议'])
     clauses.append({'anchor':loc,'quote':sentence,'type':'EFFECTIVE_DATE' if effect else 'EXPLICIT_REPEAL_CLAUSE' if repeal else 'EXPLICIT_REVISION_CLAUSE',
       'effective_date_literal':effect[1] if effect else None,'named_documents':re.findall(r'《([^》]{2,160})》',sentence),
       'operative_text_at_source':operative,'ocr_numeric_requires_original_review':s['ocr'],
       'unconditional_full_legal_effect_not_inferred':True,'announcement_of_future_effect_not_market_impact':True})
  # 成文日期与刊发日期分开保存，签署日期不替代公开可用日期。
  # 只纠正已经回看过对应原图的法律条款，原 OCR 字符和其余数字继续保留。
  for c in clauses:
   verified=next((v for v in visual if v['asset_key']==Path(c['anchor']['source']).stem),None)
   if verified and c['ocr_numeric_requires_original_review']:
    assert sha(verified['image'])==verified['image_sha256']
    literal=verified['visually_verified_quote']
    if c['type']=='EFFECTIVE_DATE' and '自' in literal or c['type']=='EXPLICIT_REPEAL_CLAUSE' and '废止' in literal:
     c['original_ocr_quote']=c['quote'];c['quote']=literal
     c['visual_review']=verified;c['ocr_numeric_requires_original_review']=False
     effect=re.search(r'自('+DAY+r'|印发之日|发布之日|公布之日)起?(?:实施|施行|执行|生效)',literal)
     c['effective_date_literal']=effect[1] if effect else None
     c['named_documents']=re.findall(r'《([^》]{2,160})》',literal)
     c['verified_document_numbers']=[m[0] for m in NUM.finditer(norm(literal))]
  signatures=[]
  for s in segments:
   if s['source']!=str(p):continue
   if re.fullmatch(DAY,norm(s['text'])):signatures.append({'date_literal':norm(s['text']),'locator':s['locator'],'role':'DOCUMENT_DATE_NOT_ASSUMED_PUBLICATION_DATE'})
  if not d.get('source_body_identity_time_verified'):issues.append('SOURCE_BODY_IDENTITY_OR_PUBLISHED_DAY_NOT_CLOSED')
  if any(g['reason']=='BODY_UPDATED_AFTER_ORIGINAL_PUBLICATION' for g in d['gaps']):issues.append('LATER_EDIT_RETAINED_AND_ASOF_DELAYED')
  if kind=='OFFICIAL_POLICY_OR_NOTICE' and not own:issues.append('NO_UNAMBIGUOUS_OWN_DOCUMENT_NUMBER')
  if kind=='OFFICIAL_POLICY_OR_NOTICE' and not any(c['type']=='EFFECTIVE_DATE' and c['operative_text_at_source'] for c in clauses):issues.append('NO_EXPLICIT_COMMENCEMENT_CLAUSE_NOT_IMPUTED')
  if any(c['ocr_numeric_requires_original_review'] for c in clauses):issues.append('OCR_LEGAL_DATE_OR_NUMBER_NEEDS_ORIGINAL_PAGE_CHECK')
  if any(g['reason']=='ATTACHMENT_OR_IMAGE_NOT_PARSED' for g in d['gaps']):issues.append('EXPLICIT_BODY_ASSET_UNAVAILABLE')
  r={'id':p.stem,'at':now(),'title':t,'url':d['url'],'kind':kind,'source':str(p),'source_sha256':sha(p),
    'aliases':d['aliases'],'published_date':d.get('published_date'),'available_at':d.get('conservative_available_at'),
    'source_body_identity_time_verified':d.get('source_body_identity_time_verified',False),'body_version_updated_at':d.get('body_version_updated_at'),
    'publication_evidence_role':d.get('source_display_date_field','PUBLISHED_DAY'),
    'first_publication_not_separately_exposed':d.get('first_publication_not_separately_exposed',False),
    'document_date_literals':signatures,'own_document_numbers':own,'explicit_number_references':cited,'literal_effect_revision_repeal_clauses':clauses,
    'explicit_linked_assets':assets,'all_source_gaps':d['gaps'],'field_gaps':issues,'first_seen_archive_proven':False,
    'whole_policy_legal_effect_verified':False,'training_eligible':False,'market_direction_inferred':False}
  save(OUT/('public-semantic-closure'+suffix)/p.name,r);rows.append(r)
  for n in own:indexes[n['number']].append(r)
  if r['source_body_identity_time_verified']:title_index[norm(t)].append(r)
  if d.get('text_sha256') and len(body)>100:aliases[d['text_sha256']].append(r['id'])
 # 只向发布不晚于引用文书的材料建立关系。后来修订不反写历史窗口。
 edges=[]
 for r in rows:
  for ref in r['explicit_number_references']:
   found=[x for x in indexes.get(ref['number'],[]) if x['id']!=r['id'] and x.get('available_at') and r.get('available_at') and x['available_at']<=r['available_at']]
   item={'from_id':r['id'],'number':ref['number'],'anchor':ref['anchor'],'quote':ref['quote'],'to_ids':[x['id'] for x in found],
    'relation':'EXPLICIT_CITATION_NOT_AUTOMATIC_AMENDMENT','no_backfill':True}
   (edges if found else unmatched).append(item)
  for cl in r['literal_effect_revision_repeal_clauses']:
   for named in cl['named_documents']:
    found=[x for x in title_index.get(norm(named),[]) if x['id']!=r['id'] and x.get('available_at') and r.get('available_at') and x['available_at']<=r['available_at']]
    if found:edges.append({'from_id':r['id'],'to_ids':[x['id'] for x in found],'relation':cl['type'],'named_document':named,'anchor':cl['anchor'],
      'operative_clause':cl['operative_text_at_source'],'no_backfill':True})
 save(out('public-explicit-reference-graph'),{'at':now(),'edges':edges,'unmatched_references':unmatched,
  'body_exact_duplicate_groups':[v for v in aliases.values() if len(v)>1],'versions_not_merged_just_for_same_title':True})
 # 从已核验的历史发行人名称中查找字面提及；名称与出处都必须早于本材料刊发日。
 profiles=read(OUT/'historical-issuer-name-profiles.json');matches=defaultdict(list)
 for r in rows:
  if not r['source_body_identity_time_verified'] or not r.get('available_at'):continue
  body=norm(read(r['source']).get('text',''))
  for code,prlist in profiles.items():
   for pr in prlist:
    if pr['published_date']>r['published_date']:continue
    hit=next((n['name'] for n in pr['names'] if len(norm(n['name']))>=6 and norm(n['name']) in body),None)
    if hit:
     at=body.index(norm(hit));matches[code].append({'document_id':r['id'],'available_at':r['available_at'],'published_date':r['published_date'],
       'name':hit,'quote':body[max(0,at-90):at+len(norm(hit))+150],'name_source':pr['source'],'name_raw_sha256':pr['raw_sha256'],
       'relation':'EXPLICIT_LEGAL_ENTITY_MENTION_ONLY','financial_effect_verified':False});break
 relations=[]
 for target in read(OUT/'historical-industry-relations-final.json')['rows']:
  links=[];lo=str(date.fromisoformat(target['target'])-timedelta(days=30))
  for h in target['holdings']:
   for m in matches.get(h['stock_code'].split('.')[0],[]):
    if lo<=m['published_date'] and m['available_at']<=target['as_of']:
     links.append({**m,'stock_code':h['stock_code'],'holding_name':h['holding_name'],'disclosed_nav_weight_pct':h['disclosed_nav_weight_pct'],
       'holding_report_sha256':target['holding_report_sha256'],'holding_report_end':target['holding_report_end'],
       'partial_holdings_disclosure':not target['holding_disclosure_full']})
  relations.append({'target':target['target'],'as_of':target['as_of'],'links':links,'no_link_means_no_verified_relation_not_zero_impact':True})
 save(out('public-historical-holding-relations'),{'at':now(),'rows':relations,'window_days':30,
  'links':sum(len(r['links']) for r in relations),'company_mention_documents':len({m['document_id'] for ms in matches.values() for m in ms}),
  'topic_words_never_used_as_proof_of_company_benefit':True,'all_links_available_before_target':all(m['available_at']<=r['as_of'] for r in relations for m in r['links'])})
 save(out('public-semantic-closure-result'),{'at':now(),'frozen_scope':1475,'processed':len(rows),'document_kinds':dict(Counter(r['kind'] for r in rows)),
  'source_body_identity_time_verified':sum(r['source_body_identity_time_verified'] for r in rows),'own_number_documents':sum(bool(r['own_document_numbers']) for r in rows),
  'literal_effect_revision_repeal_clauses':sum(len(r['literal_effect_revision_repeal_clauses']) for r in rows),'explicit_reference_edges':len(edges),
  'unmatched_references':len(unmatched),'exact_body_alias_groups':sum(len(v)>1 for v in aliases.values()),'gap_counts':dict(Counter(g for r in rows for g in r['field_gaps'])),
  'historical_holding_links':sum(len(r['links']) for r in relations),'new_fits':0})

if __name__=='__main__':run()
