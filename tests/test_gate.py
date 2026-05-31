import pytest

from quant.validate.gate import assert_not_consumed, is_consumed, mark_consumed


def test_unconsumed_by_default(tmp_path):
    state = tmp_path / "holdout_state.json"
    assert is_consumed("momentum", state) is False


def test_mark_then_consumed(tmp_path):
    state = tmp_path / "sub" / "holdout_state.json"  # 父目录自动建
    mark_consumed("momentum", state)
    assert is_consumed("momentum", state) is True
    # 其他因子不受影响
    assert is_consumed("ma_bias", state) is False


def test_assert_raises_after_consume(tmp_path):
    state = tmp_path / "holdout_state.json"
    assert_not_consumed("momentum", state)  # 未消耗 → 不抛
    mark_consumed("momentum", state)
    with pytest.raises(RuntimeError, match="holdout 已消耗"):
        assert_not_consumed("momentum", state)
