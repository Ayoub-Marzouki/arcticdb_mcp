import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from arcticdb import ErrorCode, ErrorCategory
from arcticdb_ext.version_store import VersionRequestType

from arcticdb_mcp.utils.serialization import (
    normalize_value,
    serialize_batch_entry,
    serialize_data_error,
)


def test_dataframe_values_are_json_serializable() -> None:
    frame = pd.DataFrame(
        {"price": np.array([10], dtype=np.int64), "ratio": [float("nan")]},
        index=pd.DatetimeIndex(["2024-01-01"], name="date"),
    )

    records = normalize_value(frame)

    assert records == [{"date": "2024-01-01 00:00:00", "price": 10, "ratio": None}]
    json.dumps(records, allow_nan=False)


def test_series_values_are_normalized_recursively() -> None:
    series = pd.Series([pd.Timestamp("2024-01-01"), np.int64(2)])

    assert normalize_value(series) == {
        "0": "2024-01-01 00:00:00",
        "1": 2,
    }


def _data_error(**overrides):
    fields = {
        "symbol": "MISSING",
        "error_code": ErrorCode.E_NO_SUCH_VERSION,
        "error_category": ErrorCategory.MISSING_DATA,
        "exception_string": "E_NO_SUCH_VERSION: symbol not found",
        "version_request_type": VersionRequestType.LATEST,
        "version_request_data": {"symbol": "MISSING"},
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_serialize_data_error_enum_error_code_is_strict_json() -> None:
    entry = serialize_data_error(_data_error())

    assert entry["error"] is True
    assert entry["symbol"] == "MISSING"
    assert entry["error_code"] == 3000
    assert isinstance(entry["error_code"], int)
    assert entry["error_category"] == "ErrorCategory.MISSING_DATA"
    assert entry["exception_string"] == "E_NO_SUCH_VERSION: symbol not found"
    assert entry["version_request_type"] == "VersionRequestType.LATEST"
    assert entry["version_request_data"] == {"symbol": "MISSING"}
    json.dumps(entry, allow_nan=False)


def test_serialize_data_error_none_error_code_serializes() -> None:
    entry = serialize_data_error(_data_error(error_code=None))

    assert entry["error_code"] is None
    assert entry["error"] is True
    assert entry["symbol"] == "MISSING"
    json.dumps(entry, allow_nan=False)


def test_serialize_batch_entry_data_error_serializes() -> None:
    entry = serialize_batch_entry(_data_error())

    assert entry["error"] is True
    assert entry["error_code"] == 3000
    json.dumps(entry, allow_nan=False)
