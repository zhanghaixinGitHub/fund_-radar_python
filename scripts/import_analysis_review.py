"""本机导入已留档的事后综合回看；默认预览，--commit 才追加独立回看记录。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.analysis_reviews import import_archive  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fund", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(import_archive(args.directory, args.fund, args.target, commit=args.commit), ensure_ascii=False))
