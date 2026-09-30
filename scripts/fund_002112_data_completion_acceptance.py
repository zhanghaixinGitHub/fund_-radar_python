"""完整冻结范围的分层验收与输入索引；不把字段提取、身份通过写成训练通过。"""
import json,sys
from pathlib import Path
from collections import Counter,defaultdict
from scripts.fund_002112_data_completion_v1 import OUT,ROOT,OLD,MANIFEST,PY,WEB,read,save,sha,now

def run():
 suffix='-'+sys.argv[1] if len(sys.argv)>1 else ''
 def output(name):return OUT/(name+suffix+'.json')
 before=read(OUT/'protection-before.json');after=read(OUT/'protection-after.json')
 changed={r['path'] for r in after['changed']}|set(after['missing']);verified={}
 def verify(path,expected=None):
  p=Path(path);key=str(p)
  if key not in verified:
   if key in before['files'] and key not in changed:actual=before['files'][key];mode='FULL_FINAL_PROTECTION_READBACK'
   else:actual=sha(p);mode='CURRENT_BYTE_READBACK'
   verified[key]={'path':key,'sha256':actual,'verification':mode}
  if expected is not None:assert verified[key]['sha256'].lower()==expected.lower(),key
  return verified[key]
 def evidence(p):return verify(p)
 def receipt_original(receipt):
  if receipt.get('path'):return verify(receipt['path'],receipt['sha256'])
  candidates=[root/receipt['file'] for root in [ROOT,ROOT/'materials-live',ROOT/'peer-materials',PY/'data/fund-materials']]
  for p in candidates:
   if p.exists() and verify(p)['sha256']==receipt['sha256']:return verify(p)
  raise ValueError('CATALOG_RECEIPT_ORIGINAL_NOT_FOUND:'+receipt['file'])
 manifest=read(MANIFEST);original_rows=read(OLD/'admission-audit-v3/20260929-172820.json')['rows']
 assert len(original_rows)==6167
 manual=read(OUT/'forecast-visual-review.json')['manual_cell_verification']
 rows=[];failures=[]
 for old in original_rows:
  id=old['document_id'];p=OUT/'historical-semantic-closure-v2'/f'{id}.json';r=read(p)
  verify(r['source'],r['source_sha256']);d=read(r['source']);receipt=d['receipt']
  raw=receipt.get('path') or str(ROOT/receipt['file']);verify(raw,r['raw_sha256'])
  row={'id':id,'company':r['company'],'category':r['category'],'title':r['title'],'kind':r['document_kind'],
   'original':verify(raw),'source_record':evidence(r['source']),'source_url':receipt.get('url'),
   'published_date':r['published_date'],'available_at':r['declared_available_at'],'identity_verified':r['identity_verified'],
   'identity_evidence':[evidence(x) for x in r['identity_evidence']],
   'old_stop':r['old_collection_stop'],'old_version_issues':r['old_revision_issues'],'original_field_reviews':r['old_field_reviews'],
   'literal_evidence':evidence(r['literal_facts_source']),'unresolved':[],
   'historical_first_seen_archive_proven':False,'whole_event_version_chain_verified':False,'training_eligible':False}
  if not r['identity_verified']:row['unresolved'].append('SOURCE_IDENTITY_CONFLICT_PRESERVED')
  if r['old_revision_issues']:row['unresolved'].append('ORIGINAL_LATE_VERSION_CONFLICT_NOT_CLEARED')
  fp=OUT/'financial-table-facts-v5'/p.name
  if fp.exists():
   f=read(fp);row['financial_evidence']=evidence(fp);row['financial_fields']=len(f['facts'])
   if f['facts']:
    lp=next(x for x in [OUT/'financial-grid-v5'/p.name,OUT/'financial-layout-v5'/p.name] if x.exists() and read(x).get('all_extracted_rows_layout_verified'))
    verify(read(lp)['fact_source'],read(lp)['fact_source_sha256']);row['financial_layout_evidence']=evidence(lp)
    row['financial_explicit_currency_fields']=sum(x['currency'] is not None for x in f['facts'])
    if any(x['currency'] is None for x in f['facts']):row['unresolved'].append('FINANCIAL_CURRENCY_NOT_EXPLICIT_IN_VERIFIED_TABLE')
   else:row['unresolved'].append('NO_VERIFIED_PARENT_PROFIT_TABLE_FOR_THIS_DOCUMENT_PURPOSE')
   row['financial_parser_gaps_preserved']=f['gaps']
  forecast=OUT/'forecast-final-v6'/p.name
  if forecast.exists():
   f=read(forecast);row['forecast_evidence']=evidence(forecast);row['forecast_literal_evidence']=evidence(f['literal_source'])
   checked=[]
   for fact in f['facts']:
    v=manual.get(id)
    manual_match=bool(v and v['page']==fact['page'] and v['current_raw']==fact['current']['raw'] and v['prior_raw']==fact['prior']['raw'])
    passed=fact['independent_layout_verified'] or manual_match
    if not passed:row['unresolved'].append('FORECAST_NUMERIC_CELL_REPLAY_FAILED')
    checked.append({'page':fact['page'],'layout_verified':passed,'manual_visual_override':manual_match,
      'period':fact['accounting_period'],'period_verified':fact['period_verified'],'effective_period_basis':fact['effective_period_basis'],
      'currency_explicit_on_page':fact['currency_explicit_on_page'],'stage':fact['stage'],
      'metric':fact['metric'],'reference_role':fact['current_column_reference_role']})
   row['forecast_cell_checks']=checked
   row['forecast_literal_count']=f['literal_fact_count']
   manual_source=OUT/'forecast-manual-completion'/p.name
   if manual_source.exists():
    supplement=read(manual_source);assert supplement['fact_scope_and_literal_verified']
    verify(supplement['raw_path'],supplement['raw_sha256']);row['manual_fact_scope_evidence']=evidence(manual_source)
    row['manual_fact_metric']=supplement['fact']['metric']
    if supplement.get('visual_evidence'):
     verify(supplement['visual_evidence']['path'],supplement['visual_evidence']['sha256'])
   elif not checked and not f['literal_fact_count']:row['unresolved'].append('NO_UNAMBIGUOUS_PARENT_PROFIT_NUMERIC_CLAUSE')
   if any(not x['period_verified'] for x in checked):row['unresolved'].append('FORECAST_PER_TABLE_PERIOD_REMAINS_UNRESOLVED')
   row['unresolved'].append('FORECAST_RESTATEMENT_COMPARATIVE_AND_CURRENCY_CHAIN_NOT_FULLY_CLOSED')
  bp=OUT/'buyback-execution-fields'/p.name
  if bp.exists():
   b=read(bp);row['buyback_evidence']=evidence(bp);row['buyback_actual_clause_count']=len(b['actual_facts'])
   row['buyback_complete_literal_count']=sum(f['literal_fields_complete'] for f in b['actual_facts'])
   row['buyback_not_implemented_count']=len(b['explicit_not_implemented'])
   row['unresolved'].append('REPURCHASE_UNIQUE_EVENT_AND_REVISION_CHAIN_NOT_FULLY_CLOSED')
  cp=OUT/'contract-scope-review'/p.name
  if cp.exists():
   c=read(cp);assert c['stage_and_scope_verified'];row['contract_evidence']=evidence(cp);row['contract_stage']=c['event_stage']
   row['unresolved'].append('CONTRACT_PAYMENT_REVENUE_AND_ALL_PARTY_ALLOCATIONS_NOT_INFERRED')
  rows.append(row)
 assert {r['id'] for r in rows}=={r['document_id'] for r in original_rows}
 save(output('historical-acceptance-final'),{'at':now(),'rows':rows,'all_frozen_rows_preserved':True,'no_new_fits':True})
 # 46/4/55 分别验收，不把重叠文书视为互斥事件。
 missing=[]
 for item in manifest['historical_missing_originals']:
  p=OUT/'historical-missing-audit-v4'/(item['document_id']+'.json');r=read(p);assert r['identity_verified'] and r['original_sha256_verified']
  for match in r['raw_catalog_matches']:
   receipt_original(match['receipt'])
  missing.append({'id':item['document_id'],'evidence':evidence(p),'original_and_identity_verified':True,'whole_semantics_not_implied':True})
 current=[]
 for item in manifest['current_announcements_pending']:
  id=str(item.get('document_id') or item.get('id'));p=OUT/'current-audit-v2'/(id+'.json');r=read(p);d=read(r['source_file'])
  verify(r['source_file'],r['source_sha256']);verify(d['raw_path'],r['raw_sha256'])
  for match in r['raw_catalog_matches']:
   receipt_original(match['receipt'])
  current.append({'id':id,'evidence':evidence(p),'raw':verify(d['raw_path']),
   'historical_training_eligible':d['historical_training_eligible'],'current_identity_verified':r['catalog_binding_verified'] and r['source_publication_verified'] and r['current_document_nature_verified']})
 assert len(missing)==46 and len(current)==55 and not any(r['historical_training_eligible'] for r in current)
 assert all(r['current_identity_verified'] for r in current)
 # 公共材料保留失败，不拿下载、图片 OCR 或目录日期冒充法律效力与首次发布。
 public=[];asset_by_url={}
 for p in (OUT/'public-audit-v10').glob('*.json'):
  r=read(p);receipt=r['receipt']
  if receipt.get('path') and receipt.get('sha256'):verify(receipt['path'],receipt['sha256'])
  if r.get('rendered_source'):
   verify(r['rendered_source'],r['rendered_source_sha256']);api=read(r['rendered_source'])['receipt'];verify(api['path'],api['sha256'])
  for a in r.get('attachments',[])+r.get('images',[]):
   ar=a.get('receipt',{});rawok=bool(ar.get('path') and ar.get('sha256'))
   if rawok:verify(ar['path'],ar['sha256'])
   if a.get('parsed_path'):verify(a['parsed_path'],a['parsed_sha256'])
   asset_by_url[a['url']]={'url':a['url'],'required':a['required_body_asset'],'original_verified':rawok,'parsed':bool(a['parsed_path']),
      'receipt_status':ar.get('status'),'reason':ar.get('reason'),'receipt':ar,'parser':a.get('parsed_path')}
  public.append({'id':p.stem,'title':r['aliases'][0]['title'],'url':r['url'],'source':evidence(p),
   'identity_body_display_date_verified':r.get('source_body_identity_time_verified',False),'published_date':r.get('published_date'),
   'available_at':r.get('conservative_available_at'),'first_seen_archive_proven':False,'training_eligible':False,'gaps':r['gaps'],
   'semantic_source':evidence(OUT/'public-semantic-closure-v4'/p.name)})
 assert len(public)==1475
 save(output('public-acceptance-final'),{'at':now(),'rows':public,'assets':list(asset_by_url.values()),'no_topic_word_inferred_fund_benefit':True})
 # 分开记录本轮请求、旧消耗、解压成员与未记在 HTTP 请求账的浏览器/搜索动作。
 request_sets={};requests=[]
 for folder in sorted(OUT.rglob('requests')):
  if not folder.is_dir():continue
  files=list(folder.glob('*.json'));request_sets[str(folder.relative_to(OUT))]=len(files)
  for p in files:requests.append({'file':str(p),'sha256':sha(p)})
 main=Counter(read(p)['group'] for p in (OUT/'requests').glob('*.json'))
 budget={'at':now(),'old_body_consumed':3523,'old_body_limit':3523,'current_http_attempt_records':len(requests),'request_sets':request_sets,
  'main_group_counts':dict(main),'requests':requests,'archive_members_are_not_network_requests':True,
  'browser_and_web_search_used_for_public_source_discovery_not_in_http_attempt_count':True,
  'other_running_application_requests_not_claimed_as_this_run':True,'new_fits':0,'cumulative_fits':after['cumulative_fits'],
  'remaining_shared_fits_not_reallocated':manifest['remaining_shared_fits_as_of_source']}
 assert main['historical']==45 and main['alternatives']<=120 and main['body_media']<=len(read(OUT/'public-body-media-scope-v5.json')['urls'])
 save(output('request-ledger-final'),budget)
 sources=['version-evidence-final.json','version-copy-text-comparison.json','industry/coverage-validated-v2.json','historical-industry-relations-final.json',
  'historical-industry-relations-result-final.json','historical-fund-relations.json','independent-sector-inputs.json','independent-sector-input-acceptance.json',
  'public-historical-holding-relations-v4.json','public-explicit-reference-graph-v4.json','historical-explicit-reference-graph-v2.json',
  'forecast-visual-review.json','forecast-manual-completion-result.json','financial-unresolved-detail.json',
  'policy-clause-visual-review.json','historical-cover-visual-review-v2.json','public-browser-blocks.json']
 for name in sources:
  p=OUT/name
  if p.exists():verify(p)
 version=read(OUT/'version-evidence-final.json');fr=read(OUT/'forecast-context-result-v6.json')
 summary={'at':now(),'status':'PARTIAL_WITH_EXPLICIT_UNRESOLVED_EVIDENCE','all_data_completed':False,'missing_originals':missing,'current_announcements':current,
  'historical_scope':len(rows),'historical_identity_verified':sum(r['identity_verified'] for r in rows),
  'identity_conflicts':[{'id':r['id'],'title':r['title'],'source':r['source_record']} for r in rows if not r['identity_verified']],
  'version_evidence':version,'financial':read(OUT/'financial-table-result-v5.json'),'forecast':fr,'forecast_manual_numeric_override_count':1,
  'forecast_additional_six_document_review':read(OUT/'forecast-manual-completion-result.json'),
  'forecast_literal':read(OUT/'forecast-literal-result-v3.json'),'buyback':read(OUT/'buyback-execution-result.json'),
  'contracts':read(OUT/'contract-scope-review-result.json'),'public':read(OUT/'public-audit-result-v10.json'),
  'public_semantics':read(OUT/'public-semantic-closure-result-v4.json'),'industry':read(OUT/'historical-industry-relations-result-final.json'),
  'sector_inputs':read(OUT/'independent-sector-input-acceptance.json'),'unresolved_labels_overlap':dict(Counter(g for r in rows for g in r['unresolved'])),
  'protected_immutable_research_changed':[r for r in after['research_changes'] if any(part in Path(r['path']).parts for part in ['closure','information-research','fit-ledger'])],
  'concurrent_mutable_runtime_cache_changes':after['research_changes'],'database':read(OUT/'runtime-final.json'),
  'new_fits':0,'new_registration':False,'model_replacement':False,'personal_account_conditions':'UNKNOWN',
  'no_production_deployment':True,'no_commit_or_push':True,'network_ledger':evidence(output('request-ledger-final')),
  'inputs':{'historical':evidence(output('historical-acceptance-final')),'public':evidence(output('public-acceptance-final'))}}
 assert not summary['protected_immutable_research_changed']
 save(output('acceptance-final'),summary)
 save(output('verified-files-final'),{'at':now(),'files':list(verified.values()),'count':len(verified),'all_expected_hashes_matched':True,
  'old_live_catalog_indexes_may_refresh_original_raw_catalog_receipts_verified_independently':True})
 print(json.dumps({'documents':len(rows),'public':len(public),'verified_files':len(verified),'http_attempt_records':len(requests),'status':summary['status']},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
