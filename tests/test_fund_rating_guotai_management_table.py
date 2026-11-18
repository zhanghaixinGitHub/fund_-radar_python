import pytest
from app.services.fund_rating_management_inputs import _guotai_completed_table_remediation

ACTUAL_LAYOUT = (
    "受到调查或处罚等措施的主体 国泰基金管理有限公司\n"
    "受到的具体措施类型 责令改正\n"
    "管理人采取整改措施的情况（如提 公司已采取完善管控机制、加强管理等整改措施，现已完成整\n"
    "出整改意见） 改，并经中国证监会上海监管局验收通过。"
)


def test_actual_guotai_interleaved_label_keeps_completed_remediation():
    assert _guotai_completed_table_remediation("国泰基金管理有限公司", ACTUAL_LAYOUT)


@pytest.mark.parametrize("before,after", [
    ("现已完成整", "计划完成整"),
    ("验收通过", "已申请验收"),
    ("出整改意见）", "任意插入文字）"),
    ("受到调查或处罚等措施的主体 国泰基金管理有限公司", "受到调查或处罚等措施的主体 其他基金管理有限公司"),
])
def test_future_unverified_or_other_subject_is_not_completed(before, after):
    assert not _guotai_completed_table_remediation("国泰基金管理有限公司", ACTUAL_LAYOUT.replace(before, after))


def test_same_phrase_cannot_be_reused_for_another_manager():
    assert not _guotai_completed_table_remediation("其他基金管理有限公司", ACTUAL_LAYOUT)
