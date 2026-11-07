"""已证实截断的参照不能再次发布；纠正包和新来源不受该历史退役门槛阻挡。"""
import pytest
from app.services.fund_rating_reviewed_snapshot import BASIS, publish_reviewed

RETIRED = "1f8728dadf879cc5900bee7c7dd1e3053a9b75de727c1bd1946ddd5b0abb65c0"


class NoDatabaseMutation:
    """探针在第一次数据库读取处结束；退役检查必须在它之前拒绝旧包。"""

    def __init__(self):
        self.reads = 0

    def get(self, *_):
        self.reads += 1
        raise RuntimeError("DATABASE_BOUNDARY_REACHED")


def dossier(selection_sha=None):
    entry = {"fund_code": "014156", "research_archive": {"sha256": "a" * 64}}
    if selection_sha is not None:
        entry["enhanced_performance_calculation"] = {
            "inputs": {"cohort": {"selection_source": {"sha256": selection_sha}}}
        }
    return {"basis": BASIS, "entries": [entry]}


def test_known_truncated_package_rejected_before_database_or_source_writes(tmp_path):
    session = NoDatabaseMutation()
    with pytest.raises(ValueError, match="RATING_PUBLICATION_COHORT_TRUNCATED_REBUILD_REQUIRED"):
        publish_reviewed(session, dossier(RETIRED), "原件验收后发布", root=tmp_path)
    assert session.reads == 0 and list(tmp_path.iterdir()) == []


def test_correction_keeps_retired_source_as_history_without_reactivating_it(tmp_path):
    value = dossier()
    value["entries"][0]["extra_archives"] = [{"sha256": RETIRED, "kind": "json"}]
    value["entries"][0]["withdrawn_reference_component"] = {"selection_sha256": RETIRED}
    session = NoDatabaseMutation()
    with pytest.raises(RuntimeError, match="DATABASE_BOUNDARY_REACHED"):
        publish_reviewed(session, value, "追加纠正并保留历史", root=tmp_path)
    assert session.reads == 1


def test_new_cohort_source_is_not_blocked_by_retired_source_guard(tmp_path):
    session = NoDatabaseMutation()
    with pytest.raises(RuntimeError, match="DATABASE_BOUNDARY_REACHED"):
        publish_reviewed(session, dossier("b" * 64), "采用另行核准的新来源", root=tmp_path)
    assert session.reads == 1
