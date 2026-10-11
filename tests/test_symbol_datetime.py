import pandas as pd
import pytest
from unittest.mock import patch

from arcticdb_mcp.tools import symbol_tools
from arcticdb_mcp.tools.symbol_tools import _infer_datetime_index


def test_infer_datetime_index_valid_column_becomes_datetime_index() -> None:
    df = pd.DataFrame(
        {"price": [2, 1], "date": ["2024-01-02", "2024-01-01"]}
    )

    result = _infer_datetime_index(df)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "date"
    assert "date" not in result.columns
    assert result.index.is_monotonic_increasing


def test_infer_datetime_index_invalid_value_raises_value_error() -> None:
    df = pd.DataFrame(
        {"date": ["2024-01-01", "not-a-date"], "price": [1, 2]}
    )

    with pytest.raises(ValueError) as excinfo:
        _infer_datetime_index(df)

    assert "date" in str(excinfo.value)


def test_infer_datetime_index_none_value_raises_value_error() -> None:
    df = pd.DataFrame({"date": ["2024-01-01", None], "price": [1, 2]})

    with pytest.raises(ValueError) as excinfo:
        _infer_datetime_index(df)

    assert "date" in str(excinfo.value)


def test_infer_datetime_index_ignores_non_datetime_candidates() -> None:
    df = pd.DataFrame({"price": ["garbage", "more-garbage"], "volume": [1, 2]})

    result = _infer_datetime_index(df)

    assert result is df
    assert isinstance(result.index, pd.RangeIndex)


def test_infer_datetime_index_empty_frame_is_unchanged() -> None:
    df = pd.DataFrame()

    result = _infer_datetime_index(df)

    assert result is df
    assert isinstance(result.index, pd.RangeIndex)


def test_write_symbol_propagates_value_error() -> None:
    with patch.object(symbol_tools, "get_ac") as ac:
        lib = ac.return_value.__getitem__.return_value

        with pytest.raises(ValueError) as excinfo:
            symbol_tools.write_symbol(
                "f", "S", [{"date": "2024-01-01", "price": 1}, {"date": "bad", "price": 2}]
            )

        assert "date" in str(excinfo.value)
        lib.write.assert_not_called()


def test_write_symbol_valid_data_writes() -> None:
    with patch.object(symbol_tools, "get_ac") as ac:
        lib = ac.return_value.__getitem__.return_value

        message = symbol_tools.write_symbol("f", "S", [{"date": "2024-01-01", "price": 1}])

        assert message == "Written 1 rows to 'S'."
        assert lib.write.call_count == 1


def test_append_symbol_propagates_value_error() -> None:
    with patch.object(symbol_tools, "get_ac") as ac:
        lib = ac.return_value.__getitem__.return_value

        with pytest.raises(ValueError) as excinfo:
            symbol_tools.append_symbol(
                "f", "S", [{"date": "2024-01-01", "price": 1}, {"date": None, "price": 2}]
            )

        assert "date" in str(excinfo.value)
        lib.append.assert_not_called()