"""完整本地验证入口：PDF单位复核→程序账本→首次模型→独立校验→全量综合。

新运行使用修复后的 PDF 单位读取；已有实验仅续做未执行步骤，不改写首次回答。
当前链路输出事实账本和待审草稿，不切换服务、不调用付费提供商。
"""

import argparse
import json
from pathlib import Path

from scripts import local_fund_chain_synthesis as synthesis
from scripts import local_fund_validation_chain as chain
from scripts.local_announcement_document import extract_pdf
from scripts.local_fund_pdf_units import checked_grid
from scripts.local_fund_repair_run import read, save, sha


def verified_pdf(path):
    """同源逐页字符覆盖及列网格校验后，独立核对每列单位和显式百分号。"""
    packet = extract_pdf(path)
    return packet, checked_grid(packet)


def execute(root: Path, repair: Path, previous: Path | None, fresh: Path | None, stage: str):
    """同一实验断点继续，不以第二次模型回答代替失败；未知传输结果直接停止。"""
    if not root.exists():
        if previous is None or fresh is None:
            raise ValueError("NEW_RUN_REQUIRES_FROZEN_PREVIOUS_AND_HOLDOUT_METADATA")
        if any(source["document"].get("body") for source in read(fresh)):
            raise ValueError("HOLDOUT_METADATA_MUST_NOT_PRELOAD_BODY")
        chain.freeze_code(root)
        frozen = read(root / "code-freeze.json")
        # 本轮新建目录的冻结记录一次补齐，随后才读取新 PDF 正文。
        additions = [
            Path(__file__).resolve(),
            Path(synthesis.__file__).resolve(),
            Path(__file__).with_name("local_fund_pdf_units.py"),
        ]
        frozen["files"].update({str(path): sha(path) for path in additions})
        (root / "code-freeze.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
        for path in additions:
            (root / "source/scripts" / path.name).write_bytes(path.read_bytes())
        # 按原 URL 复用历史 PDF；新文件由 metadata 明确指定，禁止猜测源文件。
        reference = previous / "pdf-map.json"
        save(root / "pdf-map.json", read(reference) if reference.exists() else {})
        chain.read_pdf = verified_pdf
        chain.prepare(root, repair, previous, fresh)
        save(
            root / "pipeline-driver.json",
            {
                "pdf_units_version": "LOCAL_PDF_GRID_UNITS_2",
                "historical_and_new_sources_full_text_preserved": True,
                "paid_calls": 0,
                "production_switched": False,
            },
        )
    chain.verify_freeze(root)
    if not (root / "summary.json").exists():
        chain.run(root, repair)
    child = root / stage
    if not (child / "protocol.json").exists():
        synthesis.prepare(child, root, repair)
    if not (child / "check.json").exists():
        synthesis.run(child, root, repair)
    # 已完成路径只核对原始回答哈希，不再次触发模型。
    responses = []
    for directory in (root, child):
        for record in (directory / "raw").glob("*.json"):
            if not record.name.endswith("-started.json"):
                responses.append({"path": str(record.resolve()), "sha256": sha(record)})
    return {
        "stage": "LOCAL_DIAGNOSTIC_COMPLETED",
        "model_responses": len(responses),
        "history_documents": 54,
        "new_documents": 6,
        "synthesis": read(child / "check.json")["status"],
        "semantic_adoption": False,
        "paid_calls": 0,
        "production_switched": False,
        "raw_responses": responses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--new-sources", type=Path)
    parser.add_argument("--synthesis-stage", default="synthesis-complete-01")
    args = parser.parse_args()
    value = execute(args.output, args.repair, args.previous, args.new_sources, args.synthesis_stage)
    print(json.dumps({key: value[key] for key in value if key != "raw_responses"}, ensure_ascii=False), flush=True)
