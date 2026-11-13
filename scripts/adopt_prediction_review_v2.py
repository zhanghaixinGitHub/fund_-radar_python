"""预览或正式采用已核实的第二版分析；只有--adopt追加正式发布记录。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.prediction_review_v2_publication import adopt_release  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--adopt", action="store_true")
    args = parser.parse_args()
    print(json.dumps(adopt_release(args.ledger, adopt=args.adopt), ensure_ascii=False))
