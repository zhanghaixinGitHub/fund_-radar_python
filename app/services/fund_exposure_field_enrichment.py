"""补齐官方文档中默认不返回的财务字段；与默认字段快照分别留证，不重复统计业绩。"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read, save
from app.services.fund_exposure_quotes import safe_error
from app.services.fund_exposure_supplement import (
    SUPPLEMENT,
    SupplementProvider,
    financial_rows,
    main_business,
    plan,
    verified_bytes,
)
from app.services.fund_materials_store import LIVE, source_path, versioned_save


def update_financials(codes, cutoff, *, check_id=None, progress=None):
    """更新当前关联公司十类资料；同日成功项复用，更正留旧版，空返回独立留证。

    财务公告滚动重查 90 日，主营构成按报告期重查 400 日；每月首次检查补核更早更正。
    新关联公司首次取已有权限下的 2020 年以来资料，不改变训练历史范围。
    """
    from app.services.fund_exposure_supplement import FINANCIAL_APIS, OPERATING_APIS

    check_id = check_id or cutoff
    provider = SupplementProvider(cache_period=check_id)
    probes = read(SUPPLEMENT / "field-permission-probes.json")["probes"]
    fields = {
        p["api"]: ",".join(p["requested_fields"])
        for p in probes
        if set(p["requested_fields"]) <= set(p["returned_fields"])
    }
    apis = FINANCIAL_APIS + OPERATING_APIS
    result = {"created": 0, "updated": 0, "skipped": 0, "empty": 0, "errors": []}
    total = len(codes) * len(apis)
    for index, code in enumerate(codes):
        for offset, api in enumerate(apis):
            try:
                relative = f"supplement/enriched-financials/{code}/{api}.json"
                output = LIVE / relative
                base_path = source_path(SUPPLEMENT.parent, relative)
                if not base_path.exists():
                    base_path = source_path(SUPPLEMENT.parent, f"supplement/financials/{code}/{api}.json")
                base = read(base_path) if base_path.exists() else {"rows": []}
                if base.get("check_id") == check_id:
                    result["skipped"] += 1
                    continue
                end = datetime.strptime(cutoff, "%Y%m%d")
                monthly = base.get("full_checked_month") != cutoff[:6]
                # 已核验基线月份视为做过完整核对，不在接入当日重取全部历史。
                baseline_month = plan()["end_date"][:6]
                full_check = not base_path.exists() or (monthly and cutoff[:6] != baseline_month)
                start = (
                    "20200101"
                    if full_check
                    else (end - timedelta(days=400 if api == "fina_mainbz" else 90)).strftime("%Y%m%d")
                )
                params = {"ts_code": code}
                if api not in {"dividend", "disclosure_date"}:
                    params.update(start_date=start, end_date=cutoff)
                if api == "fina_mainbz":
                    value, receipts = main_business(provider, code, start, cutoff, fields.get(api, ""))
                else:
                    value, receipt = provider.query(api, params, fields.get(api, ""))
                    receipts = [receipt]
                if fields.get(api) and not set(fields[api].split(",")) <= set(value["data"]["fields"]):
                    raise ValueError("EXPOSURE_FULL_FIELDS_NOT_RETURNED")
                rows, future = financial_rows(value, code, cutoff, row_limit=None if api == "fina_mainbz" else 1000)

                # 当前视图替换同一业务记录，旧行和原始响应保存在 versions 中，避免冲突版本叠加。
                def identity(row):
                    keys = (
                        "ts_code",
                        "end_date",
                        "report_type",
                        "comp_type",
                        "bz_item",
                        "curr_type",
                        "bz_code",
                        "type",
                        "ann_date",
                        "f_ann_date",
                    )
                    return digest({k: row.get(k) for k in keys})

                def publication(row):
                    return (str(row.get("f_ann_date") or row.get("ann_date") or ""), str(row.get("update_flag") or ""))

                # 公告日属于来源版本身份，不能将同期间不同公告日的多份记录压成一行。
                # 基线只读补入，可恢复中断轮里还没进入当前覆盖层的历史版本。
                baseline_path = SUPPLEMENT / "enriched-financials" / code / (api + ".json")
                if not baseline_path.exists():
                    baseline_path = SUPPLEMENT / "financials" / code / (api + ".json")
                original_rows = read(baseline_path)["rows"] if baseline_path.exists() else []
                merged = {identity(r): r for r in sorted(original_rows + base["rows"], key=publication)}
                before = dict(merged)
                incoming_keys = set()
                for row in sorted(rows, key=publication):
                    key = identity(row)
                    incoming_keys.add(key)
                    if key not in merged or publication(row) >= publication(merged[key]):
                        merged[key] = row
                for key in incoming_keys:
                    if key not in before:
                        result["created"] += 1
                    elif digest(before[key]) != digest(merged[key]):
                        result["updated"] += 1
                    else:
                        result["skipped"] += 1
                if not rows:
                    result["empty"] += 1
                versioned_save(
                    output,
                    {
                        "api": api,
                        "stock_code": code,
                        "rows": list(merged.values()),
                        "future_rows": future,
                        "receipt": receipts[0],
                        "all_receipts": receipts,
                        "checked_through": cutoff,
                        "check_id": check_id,
                        "full_checked_month": cutoff[:6]
                        if full_check
                        else base.get("full_checked_month", baseline_month),
                        "query_status": "AVAILABLE" if rows else "SOURCE_RETURNED_EMPTY",
                        "first_seen_at": now().isoformat(),
                        "training_eligible": False,
                    },
                )
            except Exception as exc:
                result["errors"].append({"code": code, "api": api, "reason": safe_error(exc)})
            finally:
                if progress:
                    progress(index * len(apis) + offset + 1, total, code, "正在更新公司财务与经营资料")
    return result


def acquire_full_fields():
    """按已实测成功的字段清单补充 469 家；结果是同一类资料的宽字段版本，不是新增季度。"""
    scope, provider = plan(), SupplementProvider()
    probes = read(SUPPLEMENT / "field-permission-probes.json")["probes"]
    apis = {p["api"]: p["requested_fields"] for p in probes if set(p["requested_fields"]) <= set(p["returned_fields"])}
    if len(apis) != 7:
        raise ValueError("EXPOSURE_FULL_FIELD_PROBE_INCOMPLETE")
    for item in probes:
        verified_bytes(item["receipt"])
    plan_path = SUPPLEMENT / "field-enrichment-plan.json"
    if not plan_path.exists():
        save(
            plan_path,
            {
                "at": now().isoformat(),
                "apis": apis,
                "codes": scope["codes"],
                "new_purchase_cny": 0,
                "default_snapshot_preserved": True,
                "training_eligible": False,
                "joint_provider_rate_limit": True,
            },
        )

    def one(code):
        counts, errors = {}, []
        for api, fields in apis.items():
            output = SUPPLEMENT / "enriched-financials" / code / (api + ".json")
            try:
                if output.exists():
                    value = read(output)
                    for receipt in value["all_receipts"]:
                        verified_bytes(receipt)
                    counts[api] = len(value["rows"])
                    continue
                default_path = SUPPLEMENT / "financials" / code / (api + ".json")
                if not default_path.exists():
                    raise ValueError("EXPOSURE_DEFAULT_SNAPSHOT_NOT_READY")
                base = read(default_path)
                params = base["receipt"]["params"]
                if api == "fina_mainbz":
                    value, receipts = main_business(
                        provider, code, scope["start_date"], scope["end_date"], ",".join(fields)
                    )
                else:
                    value, receipt = provider.query(api, params, ",".join(fields))
                    receipts = [receipt]
                if not set(fields) <= set(value["data"]["fields"]):
                    raise ValueError("EXPOSURE_FULL_FIELDS_NOT_RETURNED")
                rows, future = financial_rows(
                    value, code, scope["end_date"], row_limit=None if api == "fina_mainbz" else 1000
                )
                save(
                    output,
                    {
                        "api": api,
                        "stock_code": code,
                        "rows": rows,
                        "future_rows": future,
                        "receipt": receipts[0],
                        "all_receipts": receipts,
                        "fields": fields,
                        "first_seen_at": max(r["received_at"] for r in receipts),
                        "publication_date_missing": api == "fina_mainbz",
                        "training_eligible": False,
                        "status": "AVAILABLE" if rows else "SOURCE_RETURNED_EMPTY",
                        "default_snapshot": default_path.relative_to(SUPPLEMENT).as_posix(),
                        "meaning": "FULL_FIELD_VIEW_OF_SAME_FINANCIAL_FACTS_NOT_ADDITIONAL_PERIODS",
                    },
                )
                counts[api] = len(rows)
            except Exception as exc:
                errors.append({"code": code, "api": api, "reason": safe_error(exc)})
        return counts, errors

    result = {
        "started_at": now().isoformat(),
        "companies": len(scope["codes"]),
        "apis": list(apis),
        "completed_companies": 0,
        "files": 0,
        "errors": [],
        "rows_are_alternative_views": True,
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(one, code) for code in scope["codes"]]
        for done, future in enumerate(as_completed(futures), 1):
            counts, errors = future.result()
            result["files"] += len(counts)
            result["completed_companies"] = done
            result["errors"].extend(errors)
            result["new_requests"] = provider.count
            save(SUPPLEMENT / "field-enrichment-result.json", result, replace=True)
            if done % 20 == 0 or done == len(futures):
                print(
                    json.dumps({"full_field_companies": done, "total": len(futures), "errors": len(result["errors"])}),
                    flush=True,
                )
    result["finished_at"] = now().isoformat()
    result["status"] = "COMPLETE" if not result["errors"] else "PARTIAL"
    save(SUPPLEMENT / "field-enrichment-result.json", result, replace=True)
    return result
