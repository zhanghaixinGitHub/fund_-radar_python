"""保留已安装依赖，修复复制后的Windows虚拟环境及所有console_scripts入口。"""

import json
import os
import shutil
import subprocess
import sys
import venv
from datetime import datetime
from pathlib import Path


def install_runtime_paths(environment):
    """本项目虚拟环境启动即限定D盘临时目录及缓存，覆盖IDE/计划任务绕过dev.ps1的情况。"""
    environment = Path(environment)
    values = {
        "TEMP": "tmp", "TMP": "tmp", "PIP_CACHE_DIR": "pip", "XDG_CACHE_HOME": "xdg",
        "MPLCONFIGDIR": "matplotlib", "JOBLIB_TEMP_FOLDER": "joblib", "RUFF_CACHE_DIR": "ruff",
        "PYTHONPYCACHEPREFIX": "pycache",
    }
    # pth只执行一行；先配置环境再导入tempfile，避免Windows默认目录回落C盘。
    code = ("import os,sys; _fund_cache=os.path.join(os.path.dirname(sys.prefix),'.local-runs','dev-cache'); "
            f"os.environ.update({{k:os.path.join(_fund_cache,v) for k,v in {values!r}.items()}}); "
            "os.makedirs(os.environ['TEMP'],exist_ok=True); sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']; "
            "import tempfile; tempfile.tempdir=os.environ['TEMP']\n")
    hook = environment / "Lib/site-packages/fund_radar_runtime.pth"
    hook.write_text(code, encoding="utf-8")
    return hook


def main():
    """用基础解释器运行；先备份启动文件，再重建环境包装和可执行入口，不联网。"""
    root = Path(__file__).resolve().parents[1]
    environment = root / ".venv"
    interpreter = environment / "Scripts/python.exe"
    if Path(sys.prefix).resolve() == environment.resolve():
        raise SystemExit("请使用pyvenv.cfg中记录的基础Python运行此脚本，避免改写正在使用的python.exe。")
    probe = (
        "import importlib.metadata,json; "
        "print(json.dumps([(x.name,x.value) for x in importlib.metadata.entry_points(group='console_scripts')]))"
    )
    entries = [tuple(item) for item in json.loads(
        subprocess.check_output([str(interpreter), "-B", "-c", probe], text=True)
    )]
    backup = root / ".local-runs/workspace-relocation" / datetime.now().strftime("venv-%Y%m%d-%H%M%S")
    backup.mkdir(parents=True, exist_ok=False)
    scripts = environment / "Scripts"
    shutil.copy2(environment / "pyvenv.cfg", backup / "pyvenv.cfg")
    shutil.copytree(scripts, backup / "Scripts")
    # clear=False保留site-packages，不删除依赖；基础解释器与原环境版本相同。
    venv.EnvBuilder(with_pip=False, clear=False).create(environment)
    sys.path.insert(0, str(environment / "Lib/site-packages"))
    from pip._vendor.distlib.scripts import ScriptMaker

    maker = ScriptMaker(None, str(scripts))
    maker.executable = str(interpreter)
    maker.clobber = True
    maker.variants = {""}
    entries.extend((name, "pip._internal.cli.main:main") for name in ("pip3", "pip3.13"))
    for name, value in sorted(set(entries)):
        # 包元数据只能产生Scripts内的文件，禁止把不合法入口写到环境外。
        if Path(name).name != name or any(c in name for c in ("/", "\\", os.sep)):
            raise ValueError("VENV_ENTRYPOINT_NAME_INVALID")
        maker.make(f"{name} = {value}")
    install_runtime_paths(environment)
    print(json.dumps({"environment": str(environment), "entrypoints": len(set(entries)), "backup": str(backup)}))


if __name__ == "__main__":
    main()
