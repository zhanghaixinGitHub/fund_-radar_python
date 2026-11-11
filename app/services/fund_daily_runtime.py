"""每日续办的真实服务适配；配置凭据只在内存，正式归档和只读回查使用既有边界。"""

import json
import subprocess
import sys
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.services.direction_1d_protocol import digest, window
from app.services.fund_daily_prediction import TERMINAL
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.fund_workflow_materials import registered_materials, workflow_root

PROJECT = Path(__file__).resolve().parents[2]


class LocalDailyBackend:
    """同一运行复用服务作业，逐只异常留痕，批准与方向始终来自实际聊天审阅。"""

    def __init__(self, directory, core=Path(r"D:\Project-FullStack\workSpace01\02-java")):
        self.directory, self.core = Path(directory), core
        self.active_block_file = None
        self.settings = get_settings()
        self.client = httpx.Client(
            timeout=httpx.Timeout(25, connect=5),
            trust_env=False,
            headers={"X-Service-Token": self.settings.ai_service_token.get_secret_value()},
        )
        self.base = "http://127.0.0.1:8000/internal/v1/funds/sync-jobs"

    def close(self):
        self.client.close()
        if hasattr(self, "collector"):
            self.collector.close()

    def ratings(self, terminal):
        """资料终态后执行持仓月评；公告和持仓检查没有真实回执时保持待办。"""
        # 再取一次本人持仓，避免同步期间已清仓的基金仍被新发布。
        from app.commands.fund_prediction_followthrough import owner_scope
        from app.services.fund_market_preparation import market_scope
        from app.services.fund_market_workflow import MarketSyncWorkflow

        codes = set(owner_scope(self.settings, self.owner))
        catalog = market_scope()
        if not codes <= {row["fund_code"] for row in catalog}:
            raise ValueError("RATING_HOLDING_NOT_IN_ACTIVE_CATALOG")
        if not codes:
            return {"status": "SUCCEEDED", "scope": [], "reason": "EMPTY_HOLDINGS", "readback": []}
        workflow = MarketSyncWorkflow(scope_provider=lambda: [r for r in catalog if r["fund_code"] in codes])
        try:
            for child in terminal["children"]:
                items = (child.get("result_summary") or {}).get("items", [])
                if child["job_type"] == "FUND_NEWS":
                    workflow.news_results = {r["fund_code"]: r for r in items if r["fund_code"] in codes}
                elif child["job_type"] == "FUND_MATERIALS":
                    workflow.material_results = {r["fund_code"]: r for r in items if r["fund_code"] in codes}
            from app.services.fund_market_review import pending_events

            # 同步中心的精简摘要不含事项正文；必须读取原公告台账，不能把字段缺失当作无事项。
            for code in codes:
                path = workflow_root() / "news/funds" / (code + ".json")
                news = read(path) if path.exists() else {}
                item = workflow.news_results.setdefault(code, {"status": "FAILED"})
                # 原子任务失败后可能已由本轮逐篇核验补办成功；以今日真实台账为准，
                # 旧子任务的失败回执保留，但不能继续遮蔽已完成的补办。
                item["status"] = (
                    "SUCCEEDED"
                    if news.get("status") == "SUCCEEDED"
                    and news.get("catalog_complete")
                    and news.get("coverage", {}).get("end", "") >= str(now().date())
                    else "FAILED"
                )
                item["candidate_events"] = pending_events(news.get("candidate_events", []))
            result = workflow.ratings(lambda *args: None)
            # 正式查询独立读回；不得把更新器返回值当作列表已显示。
            result["scope"] = sorted(codes)
            result["readback"] = [
                self.get(f"http://127.0.0.1:8000/internal/v1/funds/{code}/rating") for code in sorted(codes)
            ]
            return result
        finally:
            workflow.close()

    def get(self, url):
        response = self.client.get(url)
        response.raise_for_status()
        return response.json()

    def context(self):
        from app.commands.fund_prediction_followthrough import owner_scope

        self.owner = json.loads((PROJECT / ".local-runs/automation-owner/account.json").read_text(encoding="utf-8"))[
            "owner_user_id"
        ]
        for url in (
            "http://127.0.0.1:8000/internal/v1/health",
            self.settings.core_service_base_url + "/actuator/health",
        ):
            self.get(url)
        scope = owner_scope(self.settings, self.owner)
        self.scope = scope
        # 一个运行分页读取一次正式档案；新保存后只刷新受影响基金，避免逐周期N+1。
        from app.commands.fund_prediction_followthrough import saved_records

        self.window = window(now())
        self.records = saved_records(self.owner, scope, self.window["target_nav_date"], self.core / ".env")
        return now().isoformat(), self.owner, scope, self.window

    def guard(self, output, blocked_file):
        self.blocked_file = blocked_file
        self.active_block_file = self.current_blocks(blocked_file) if blocked_file else None
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-m",
            "app.commands.fund_prediction_followthrough",
            "--output",
            str(output),
            "--core-env",
            str(self.core / ".env"),
        ]
        if self.active_block_file:
            command += ["--blocked-file", str(self.active_block_file)]
        # 收尾会逐份核对本人多个周期的原结果、正文和完整输入快照。现场52份
        # 结果已因120秒旧上限被提前中断；仍有界等待五分钟，不跳过任何读回，
        # 超限继续保留失败，绝不将旧守卫或已导入记录当作完成。
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", timeout=300)
        if result.returncode not in {0, 2} or not output.exists():
            raise RuntimeError("FOLLOWTHROUGH_EXECUTION_FAILED:" + result.stderr[-500:])
        return read(output)

    def current_blocks(self, path):
        """外部失败只绑定首次登记时的真实资料版本；资料变更后自动重新打开行动。

        原回执和失败原件始终保留；本方法只生成过滤视图，不改写历史外部阻断。
        字节、日期、基金、尝试时间等证据仍由完成守卫独立验证。
        """
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        ledger_path = self.directory / "external-input-bindings.json"
        bindings = read(ledger_path) if ledger_path.exists() else {}
        active, changed = [], []
        for row in rows:
            code = row["fund_code"]
            canonical = workflow_root() / "materials" / (code + ".json")
            handoff = workflow_root() / "comprehensive-inputs" / (code + ".json")
            material = read(canonical) if canonical.exists() else {}
            approval = read(handoff) if handoff.exists() else {}
            from app.services.fund_prediction_snapshot_compat import materials_input_identity

            approved = read(Path(approval["materials_path"])) if approval.get("materials_path") else None
            if approved is not None and digest(approved) != approval.get("materials_hash"):
                raise ValueError("EXTERNAL_BLOCK_INPUT_MATERIALS_CHANGED")
            context_hash = digest(
                {
                    "source": material.get("source_hash"),
                    "materials": materials_input_identity(approved) if approved is not None else None,
                    "workflow": approval.get("workflow_source_hash"),
                }
            )
            key = digest(row)
            # 本地验收早期版本按完整审计包摘要绑定；只有旧摘要与当前原包逐字一致
            # 才迁入明确的输入身份版本，不能借升级重新批准已变化的外部失败。
            legacy_hash = digest(
                {
                    "source": material.get("source_hash"),
                    "materials": approval.get("materials_hash"),
                    "workflow": approval.get("workflow_source_hash"),
                }
            )
            if (
                key in bindings
                and not bindings[key].get("identity_version")
                and bindings[key]["input_hash"] == legacy_hash
            ):
                bindings[key] = {"fund_code": code, "input_hash": context_hash, "identity_version": 2}
            if key not in bindings:
                bindings[key] = {"fund_code": code, "input_hash": context_hash, "identity_version": 2}
            if bindings[key]["input_hash"] == context_hash:
                active.append(row)
            else:
                changed.append({"fund_code": code, "horizon": row["horizon"], "reason": "INPUT_CHANGED_RETRY_REQUIRED"})
        versioned_save(ledger_path, bindings)
        versioned_save(self.directory / "reopened-external-actions.json", changed)
        filtered = self.directory / "external-block-views" / (digest(active) + ".json")
        filtered.parent.mkdir(parents=True, exist_ok=True)
        if not filtered.exists():
            with filtered.open("x", encoding="utf-8") as stream:
                json.dump(active, stream, ensure_ascii=False)
        return filtered

    def latest_batch(self):
        return self.get(self.base + "/all/latest")

    def start_or_reuse(self, previous):
        # 运行版本必须认识 materialsOnly，探针失败不会实际启动任务。
        probe = self.client.post(self.base + "/all", json={"materialsOnly": "__invalid_contract_probe__"})
        if probe.status_code != 422 or not any(
            x.get("loc") == ["body", "materialsOnly"] for x in probe.json().get("detail", [])
        ):
            raise ValueError("RUNNING_SERVICE_MATERIALS_ONLY_CONTRACT_MISSING")
        if previous and previous["status"] not in TERMINAL:
            return {"job_id": previous["job_id"], "reused": True}
        from app.services.fund_exposure_common import ROOT

        state = read(ROOT.parent / "sync-center/jobs.json")
        active = [j for j in state["jobs"] if j["status"] not in TERMINAL]
        if active:
            original = next((j for j in active if j["status"] == "RUNNING"), active[0])
            return {"job_id": original["job_id"], "reused": True}
        response = self.client.post(self.base + "/all", json={"materialsOnly": True})
        if response.status_code == 409:
            # 与其他正常入口竞态且明确未接受时，恢复实际原作业等待，不重试POST。
            state = read(ROOT.parent / "sync-center/jobs.json")
            active = [j for j in state["jobs"] if j["status"] not in TERMINAL]
            if active:
                original = next((j for j in active if j["status"] == "RUNNING"), active[0])
                return {"job_id": original["job_id"], "reused": True}
        response.raise_for_status()
        return {"job_id": response.json()["job_id"], "reused": False}

    def recover_submission(self, intent):
        latest = self.latest_batch()
        previous_id = (intent.get("previous") or {}).get("job_id")
        if latest and latest["job_id"] != previous_id:
            return {"job_id": latest["job_id"], "recovered_uncertain_submission": True}
        if latest and latest["status"] not in TERMINAL:
            return {"job_id": latest["job_id"], "reused": True}
        raise ValueError("UNCERTAIN_SUBMISSION_REQUIRES_ORIGINAL_JOB_LOOKUP")

    def batch_status(self, batch):
        job = self.get(self.base + "/" + batch["job_id"])
        # 子任务ID来自实际状态树，不按固定12阶段或旧jobId猜测。
        from app.services.fund_exposure_common import ROOT

        state = read(ROOT.parent / "sync-center/jobs.json")
        ids = state["children"].get(job["job_id"], [])
        options = state.get("prediction_scopes", {}).get(job["job_id"], {})
        if not options.get("materials_only"):
            if batch.get("reused"):
                return {
                    "job": job,
                    "children": [],
                    "terminal": False,
                    "existing_job_finished": job["status"] in TERMINAL,
                    "waiting_original_job": True,
                }
            raise ValueError("ORIGINAL_JOB_NOT_MATERIALS_ONLY")
        children = [self.get(self.base + "/" + child) for child in ids]
        terminal = job["status"] in TERMINAL and bool(children) and all(c["status"] in TERMINAL for c in children)
        result = {
            "job": job,
            "children": children,
            "terminal": terminal,
            "prediction_outcomes": (job.get("result_summary") or {}).get("predictionOutcomes"),
        }
        outcome = result["prediction_outcomes"]
        required = {"total", "checked", "completed", "created", "pending", "failed", "remaining"}
        if outcome:
            child = next((c for c in children if c["job_id"] == outcome.get("jobId")), None)
            if (
                not child
                or child["job_type"] != "PREDICTION_OUTCOMES"
                or child["status"] != outcome.get("status")
                or not required <= set(outcome.get("summary") or {})
            ):
                result.update(prediction_outcomes=None, outcome_validation_error="OUTCOME_SUBTASK_RECEIPT_INCOMPLETE")
            elif terminal and outcome.get("summary", {}).get("pending"):
                # 常驻进程不为加载修复而重启。每日新进程复用原快照日历续核等待项，
                # 原jobId和七项计数保持原回执，续办结果单独保存并明确报告。
                from app.services.fund_prediction_outcome_recovery import recover

                try:
                    result["prediction_outcomes_recovery"] = recover(
                        self.directory / "prediction-outcomes-recovery.json", outcome, child
                    )
                except Exception as error:
                    # 旧结果核对与新的独立预测隔离，保留具体失败，绝不默认零条成功。
                    result["prediction_outcomes_recovery"] = {
                        "status": "FAILED",
                        "failure": type(error).__name__ + ":" + str(error)[:400],
                    }
                    versioned_save(
                        self.directory / "prediction-outcomes-recovery-failure.json",
                        result["prediction_outcomes_recovery"],
                    )
        for child in children:
            if child.get("error_code"):
                path = self.directory / "first-failures" / (child["job_id"] + ".json")
                if not path.exists():
                    versioned_save(path, child)
        return result

    def recover_materials(self, code, output, current_window):
        from app.services.fund_prediction_recovery import recover

        # 同一输入版本已复核且守卫无资料行动时，无须重复采集或改写批准时间。
        report = read(self.directory / "followthrough.json")
        fund = next((f for f in report["funds"] if f["fund_code"] == code), None)
        if fund and not fund["actions"]:
            return {"status": "CURRENT_MATERIALS_VERIFIED"}
        if not hasattr(self, "collector"):
            from app.services.market_company_materials import MarketCompanyCollector

            self.collector = MarketCompanyCollector(workflow_root())
        return recover(code, output, current_window, collector=self.collector)

    def approve(self, code, review):
        from app.services.fund_market_review import approve_comprehensive, record_prediction_event_review
        from app.services.fund_news_identity import record_applicability_review

        # 所有文字证据由当前聊天提交，逐字匹配仍由原入口负责；不能从关键词自动批准。
        identities = review.get("notice_identity_reviews", [])
        for row in identities:
            record_applicability_review(
                code,
                row["entry"],
                row["source"],
                row["catalog_source"],
                row["scope"],
                row["note"],
                row["quotes"],
                root=workflow_root() / "news",
                identity_proof=row.get("identity_proof"),
                manager_anchor=row.get("manager_anchor"),
                product_name=row.get("product_name"),
            )
        if identities:
            from app.services.fund_market_news import refresh_material_news_link
            from app.services.fund_market_preparation import MarketFundPreparation

            preparation = MarketFundPreparation()
            try:
                result = preparation.sync_news(code)
            finally:
                preparation.close()
            if result["status"] != "SUCCEEDED":
                raise ValueError("NOTICE_RECOVERY_STILL_INCOMPLETE")
            refresh_material_news_link(code)
        for row in review.get("prediction_event_reviews", []):
            record_prediction_event_review(code, row["article_id"], row["source_hash"], row["note"], row["quotes"])
        if review.get("materials_path"):
            # 新原件已取得时，显式选择的旧资料路径也必须包含该原件。
            # 防止“恢复器发现新半年报、用途审阅却批准旧季报包”的衔接遗漏；
            # 这里只核本基金本轮原文摘要，不自动批准、不替用户形成方向。
            folder = self.directory / "funds" / code / "companies"
            discoveries = [read(p) for p in folder.glob("*.JP/jp-financials/latest.json")]
            required = {r["raw_sha256"] for r in discoveries if r.get("status") == "ORIGINAL_PARSED_READING_REQUIRED"}
            if required:
                selected = read(Path(review["materials_path"]))
                available = {d.get("raw_sha256") for d in selected.get("documents", [])}
                if not required <= available:
                    raise ValueError("CURRENT_JP_FINANCIAL_ORIGINAL_NOT_IN_REVIEWED_MATERIALS")
            return approve_comprehensive(
                code,
                review["note"],
                materials_path=Path(review["materials_path"]),
                resolutions=review.get("resolutions", []),
            )
        return {"fund_code": code, "notice_reviews_applied": len(identities)}

    def periods(self, code, current_window):
        from app.commands.fund_prediction_followthrough import horizon_dates
        from app.services.fund_information_snapshot import prepare
        from app.services.fund_materials import snapshot
        from app.services.fund_prediction_followthrough import HORIZONS, recovery_actions, verify_outputs

        handoff = registered_materials(code, at=now())
        material = read(Path(handoff["materials_path"])) if handoff else None
        from app.services.fund_prediction_snapshot_compat import materials_input_identity

        if not hasattr(self, "materials_input_hashes"):
            self.materials_input_hashes = {}
        self.materials_input_hashes[code] = materials_input_identity(material) if material else None
        news = read(workflow_root() / "news/funds" / (code + ".json"))
        reports = [r for r in (snapshot(code) or {}).get("reports", []) if r["publishedDate"] <= str(now().date())]
        latest = max(reports, key=lambda r: (r["endDate"], r["publishedDate"])) if reports else None
        actions = recovery_actions(
            code,
            latest,
            material,
            news,
            handoff,
            current_window["base_nav_date"],
            str(now().date()),
            target_reports={
                target: max(
                    (snapshot(target) or {}).get("reports", []),
                    key=lambda r: (r["endDate"], r["publishedDate"]),
                    default=None,
                )
                for target in (material or {}).get("exposure_chain", [code])[1:]
            },
        )
        dates = {}
        if handoff and handoff.get("reviewed") and not handoff.get("gaps"):
            dates, calendar_actions, _ = horizon_dates(prepare(now(), code, handoff["materials_path"]), code, now())
            actions += calendar_actions
        exceptions = (
            json.loads(Path(self.active_block_file).read_text(encoding="utf-8")) if self.active_block_file else []
        )
        checked = verify_outputs(code, current_window, dates, [], actions, exceptions)
        if not hasattr(self, "period_dates"):
            self.period_dates = {}
        self.period_dates[code] = dates
        return [
            {
                "horizon": h,
                "status": "EXTERNAL_BLOCKED"
                if row["status"] == "EXTERNAL_BLOCKED"
                else "ACTION_REQUIRED"
                if any(not a.get("horizon") or a["horizon"] == h for a in actions)
                else "READY",
                "reason": row.get("reason"),
                "actions": [a for a in actions if not a.get("horizon") or a["horizon"] == h],
            }
            for h, row in zip(HORIZONS, checked["horizons"], strict=True)
        ]

    def read_back(self, code, horizon, current_window):

        records = self.records
        expected = getattr(self, "period_dates", {}).get(code, {}).get(horizon) or {}
        end_date = current_window["target_nav_date"] if horizon == "T1" else expected.get("endDate")
        return bool(end_date) and any(
            r["fund_code"] == code
            and r["horizon"] == horizon
            and r["base_date"] == current_window["base_nav_date"]
            and r["end_date"] == end_date
            and r["owner_verified"]
            and r.get("snapshot_verified") is True
            and r.get("materials_input_hash") == self.materials_input_hashes.get(code)
            for r in records
        )

    def freeze(self, code, horizon, folder):
        from app.services import comprehensive_multi_prediction as multi
        from app.services.fund_codex_analysis import freeze_packet

        handoff = registered_materials(code, at=now())
        if not handoff or not handoff.get("reviewed") or handoff.get("gaps"):
            raise ValueError("CURRENT_MATERIALS_REVIEW_REQUIRED")
        path = folder / "input.json"
        if path.exists():
            packet = read(path)
            data = packet["input"] if horizon == "T1" else packet["input"]["data"]
            if data["fund_code"] != code or data["window"]["target_nav_date"] != self.window["target_nav_date"]:
                raise ValueError("FROZEN_IDENTITY_OR_WINDOW_CHANGED")
            if data.get("materials_path") == handoff["materials_path"]:
                return {"path": str(path), "directory": str(folder), "input_hash": packet["input_hash"]}
            # 新资料追加新冻结版本；旧原文、旧输入与旧失败审阅保持原样。
            folder = folder / "versions" / handoff["materials_hash"]
            path = folder / "input.json"
            if path.exists():
                packet = read(path)
                return {"path": str(path), "directory": str(folder), "input_hash": packet["input_hash"]}
        result = (
            freeze_packet(folder, code, Path(handoff["materials_path"]))
            if horizon == "T1"
            else multi.freeze_packet(folder, code, horizon, Path(handoff["materials_path"]))
        )
        return {"path": str(path), "directory": str(folder), **result}

    def import_and_archive(self, code, horizon, folder, owner, current_window):
        from app.commands.fund_prediction_followthrough import owner_scope
        from app.services import comprehensive_multi_prediction as multi
        from app.services.fund_codex_analysis import import_review
        from app.services.fund_daily_owner_archive import archive_multi

        if code not in owner_scope(self.settings, owner):
            raise ValueError("OWNER_REMOVED_BEFORE_SAVE")
        receipt_path = folder / "import-receipt.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
        else:
            receipt = (
                import_review(folder / "input.json", folder / "review.json")
                if horizon == "T1"
                else multi.import_review(folder / "input.json", folder / "review.json")
            )
            versioned_save(receipt_path, receipt)
        if horizon == "T1":
            archived = self.archive_one_day(code, owner, current_window["target_nav_date"], folder)
        else:
            archived = archive_multi(
                self.core, self.settings, owner, code, receipt["prediction_id"], current_window["target_nav_date"]
            )
        versioned_save(folder / "owner-archive.json", archived)
        from app.commands.fund_prediction_followthrough import saved_records

        self.records = [r for r in self.records if r["fund_code"] != code] + saved_records(
            owner, [code], current_window["target_nav_date"], self.core / ".env"
        )
        return receipt

    def archive_one_day(self, code, owner, target, folder):
        """沿本人既有幂等入口归档；时钟新鲜度拒绝只重试一次，不放宽校验。

        本轮实际出现过 CLOCK_SKEW：繁忙时状态响应到达Java已超出其原容差。
        重新请求取得服务端当前状态即可恢复。其他400、权限和身份拒绝不重试，
        首个拒绝单独保留，重试仍绑定同一本人、基金和目标，不创建另一预测。
        """
        for attempt in range(2):
            response = self.client.post(
                self.settings.core_service_base_url + "/internal/v1/direction-1d/sync/" + code,
                json={"targetNavDate": target, "ownerUserId": owner},
            )
            try:
                body = response.json()
            except ValueError:
                body = None
            clock_rejected = (
                response.status_code == 400
                and isinstance(body, dict)
                and any(body.get(key) == "CLOCK_SKEW" for key in ("code", "message", "reason", "error"))
            )
            if clock_rejected and attempt == 0:
                # 不保存未经筛选的完整服务响应，避免异常字段携带凭据或个人信息。
                versioned_save(
                    folder / "owner-archive-first-failure.json",
                    {
                        "at": now().isoformat(),
                        "fund_code": code,
                        "target_date": target,
                        "owner_user_id": owner,
                        "status": 400,
                        "reason": "CLOCK_SKEW",
                    },
                )
                continue
            response.raise_for_status()
            return body
        raise RuntimeError("OWNER_ARCHIVE_RETRY_EXHAUSTED")
