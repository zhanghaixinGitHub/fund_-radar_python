"""本机 Docker PostgreSQL 备份及隔离恢复演练，不改原库路由、不删除原库或既有文件。"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.prediction_models import model_directory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="fund-radar-postgres-1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.container):
        raise ValueError("容器名不正确")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def command(program, *parts, stdin=None, stdout=subprocess.PIPE):
        # 容器内只引用既有数据库管理员名；不打印或传出密码，参数逐项传递。
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                args.container,
                "sh",
                "-c",
                'backup_program=$1; shift; exec "$backup_program" -U "$POSTGRES_USER" "$@"',
                "backup",
                program,
                *parts,
            ],
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise RuntimeError(f"{program}失败：" + result.stderr.decode("utf-8", errors="replace")[:800])
        return result.stdout.decode("utf-8").strip() if stdout == subprocess.PIPE else None

    def sql(database, query):
        return command("psql", "-d", database, "-At", "-v", "ON_ERROR_STOP=1", "-c", query)

    checks = {
        "fund_core": ("portfolio_decision_report", "report_id"),
        "fund_ai": ("fund_prediction_record", "prediction_id"),
    }
    report = {"checkedAt": datetime.now(UTC).isoformat(), "mode": "REAL_LOCAL_BACKUP_ISOLATED_RESTORE", "databases": {}}
    originals = {}
    # 先备份引用方，再备份公共预测方；原文为追加式，引用所需旧记录不会被删除。
    for database, (table, id_column) in checks.items():
        originals[database] = sql(
            database, f"SELECT {id_column}||':'||content_hash FROM {table} ORDER BY {id_column} LIMIT 5"
        )
        path = output / (database + ".dump")
        with path.open("wb") as stream:
            command("pg_dump", "-d", database, "-Fc", stdout=stream)
        report["databases"][database] = {
            "dumpBytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    model_hashes = {}
    with zipfile.ZipFile(output / "models.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in model_directory().glob("*.json"):
            model_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, path.name)
        policy = Path(__file__).resolve().parents[1] / "app/data/prediction_policy_v1.json"
        archive.write(policy, "prediction_policy_v1.json")
    restored = output / "restored-models"
    restored.mkdir()
    with zipfile.ZipFile(output / "models.zip") as archive:
        for name in archive.namelist():
            target = (restored / name).resolve()
            if target.parent != restored.resolve():
                raise ValueError("恢复文件必须位于本次演练目录")
            target.write_bytes(archive.read(name))
    assert all(
        hashlib.sha256((restored / name).read_bytes()).hexdigest() == value for name, value in model_hashes.items()
    )
    report["models"] = {"count": len(model_hashes), "restoredHashesMatch": True, "files": model_hashes}

    for database, (table, id_column) in checks.items():
        temporary = "prediction_restore_" + uuid4().hex
        # 名称由本脚本生成且与两个原库名严格不同；仅清理本次成功创建的空目标库。
        assert re.fullmatch(r"prediction_restore_[0-9a-f]{32}", temporary)
        sql("postgres", f"CREATE DATABASE {temporary}")
        try:
            with (output / (database + ".dump")).open("rb") as stream:
                command("pg_restore", "-d", temporary, "--no-owner", "--no-privileges", "--exit-on-error", stdin=stream)
            for row in originals[database].splitlines():
                identity, digest = row.split(":", 1)
                assert re.fullmatch(r"[0-9a-f-]{36}", identity)
                assert sql(temporary, f"SELECT content_hash FROM {table} WHERE {id_column}='{identity}'") == digest
            count = int(sql(temporary, f"SELECT count(*) FROM {table}"))
            report["databases"][database].update(restoredRecords=count, originalHashesMatch=True)
        finally:
            sql("postgres", f"DROP DATABASE {temporary}")
        print(f"{database}: backup/restored/hash checked", flush=True)
    report["temporaryDatabasesRemoved"] = True
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"model files restored: {len(model_hashes)}", flush=True)


if __name__ == "__main__":
    main()
