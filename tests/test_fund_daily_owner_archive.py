"""归档只能复用实际项目类路径，Maven参数文件不应被当作路径本身。"""
import json
from pathlib import Path

import pytest
from app.services.fund_daily_owner_archive import runtime_classpath


def test_maven_single_classpath_argfile_and_direct_ide_classpath(tmp_path):
    core = Path('C:/ideaProject/workSpace12')
    cp = r'C:\ideaProject\workSpace12\target\classes;C:\maven repository\spring.jar'
    path = tmp_path / 'spring-boot.argfile'
    path.write_text(json.dumps(cp), encoding='utf-8')
    assert runtime_classpath(['java', '-cp', '@'+str(path), 'com.fundradar.core.FundCoreApplication'], core) == cp
    assert runtime_classpath(['java', '-classpath', cp], core) == cp


@pytest.mark.parametrize('value', [
    [], 'C:/other/target/classes;C:/cache/lib.jar', 'C:/ideaProject/workSpace12\n-cp other',
    'C:/ideaProject/workSpace12-old/target/classes',
])
def test_wrong_project_or_multiple_values_are_rejected(tmp_path, value):
    path = tmp_path / 'args'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='JAVA_RUNTIME_CLASSPATH'):
        runtime_classpath(['java','-cp','@'+str(path)], Path('C:/ideaProject/workSpace12'))


def test_malformed_and_missing_classpath_are_rejected(tmp_path):
    path = tmp_path / 'args'
    path.write_text('-cp C:/ideaProject/workSpace12 other',encoding='utf-8')
    with pytest.raises(ValueError, match='ARGFILE_INVALID'):
        runtime_classpath(['java','-cp','@'+str(path)],Path('C:/ideaProject/workSpace12'))
    with pytest.raises(ValueError, match='CLASSPATH_MISMATCH'):
        runtime_classpath(['java','-cp'],Path('C:/ideaProject/workSpace12'))
