import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from arcticdb import ErrorCode, ErrorCategory

from arcticdb_mcp.tools import batch_tools, query_tools, snapshot_tools, symbol_tools
from arcticdb_mcp.utils.serialization import normalize_value


class ReadConsistencyTests(unittest.TestCase):
    def test_all_paths_preserve_collisions_and_produce_strict_json(self):
        frame = pd.DataFrame({'date': ['source'], 'price': [float('nan')], 'missing': [pd.NA]},
                             index=pd.DatetimeIndex(['2026-01-01'], name='date'))
        original = frame.copy(deep=True)
        expected = normalize_value(frame)
        self.assertEqual(expected[0]['date'], 'source')
        self.assertEqual(expected[0]['date_index'], '2026-01-01 00:00:00')
        self.assertIsNone(expected[0]['price'])
        self.assertIsNone(expected[0]['missing'])
        for module, name, kwargs in [
            (symbol_tools, 'read_symbol', {}),
            (symbol_tools, 'head_symbol', {}),
            (symbol_tools, 'tail_symbol', {}),
            (snapshot_tools, 'read_symbol_from_snapshot', {'snapshot_name': 'before'}),
            (query_tools, 'query_date_range', {'start': '2026-01-01', 'end': '2026-01-02'}),
        ]:
            with self.subTest(name=name), patch.object(module, 'get_ac') as ac:
                lib = ac.return_value.__getitem__.return_value
                for method in ('read', 'head', 'tail'):
                    getattr(lib, method).return_value = SimpleNamespace(data=frame)
                result = getattr(module, name)('finance', 'prices', **kwargs)
                self.assertEqual(result, expected)
                json.dumps(result, allow_nan=False)
        pd.testing.assert_frame_equal(frame, original)

    def test_multiindex_and_missing_values(self):
        frame = pd.DataFrame({'level_0': [pd.NaT]}, index=pd.MultiIndex.from_tuples([(1, 2)]))
        result = normalize_value(frame)
        self.assertEqual(result, [{'level_0_index': 1, 'level_1': 2, 'level_0': None}])
        json.dumps(result, allow_nan=False)

    def test_read_batch_missing_symbols_returns_error_entries(self):
        frame = pd.DataFrame({'price': [1.0, 2.0]})
        errors = [
            SimpleNamespace(
                symbol='MISSING_A',
                error_code=ErrorCode.E_NO_SUCH_VERSION,
                error_category=ErrorCategory.MISSING_DATA,
                exception_string='E_NO_SUCH_VERSION: not found A',
                version_request_type='LATEST',
                version_request_data=None,
            ),
            SimpleNamespace(
                symbol='MISSING_B',
                error_code=ErrorCode.E_NO_SUCH_VERSION,
                error_category=ErrorCategory.MISSING_DATA,
                exception_string='E_NO_SUCH_VERSION: not found B',
                version_request_type='LATEST',
                version_request_data=None,
            ),
        ]
        with patch.object(batch_tools, 'get_ac') as ac:
            lib = ac.return_value.__getitem__.return_value
            lib.read_batch.return_value = [
                SimpleNamespace(symbol='T', library='finance', version=0,
                                timestamp='1', metadata=None, data=frame),
                *errors,
            ]
            result = batch_tools.read_batch('finance', ['T', 'MISSING_A', 'MISSING_B'])
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0]['symbol'], 'T')
            self.assertEqual(result[0]['data'], [{'index': 0, 'price': 1.0},
                                                 {'index': 1, 'price': 2.0}])
            self.assertEqual(result[1]['error'], True)
            self.assertEqual(result[1]['error_code'], 3000)
            self.assertEqual(result[1]['symbol'], 'MISSING_A')
            self.assertEqual(result[2]['error'], True)
            self.assertEqual(result[2]['error_code'], 3000)
            self.assertEqual(result[2]['symbol'], 'MISSING_B')
            json.dumps(result, allow_nan=False)
