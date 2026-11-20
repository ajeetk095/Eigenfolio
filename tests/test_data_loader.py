"""Data download normalization tests; no network access is required."""
import pandas as pd

from src.data_loader import _extract_close


def test_extract_close_price_ticker_layout():
    columns = pd.MultiIndex.from_product([['Close', 'Volume'], ['AAPL', 'MSFT']])
    raw = pd.DataFrame([[10, 20, 100, 200], [11, 21, 110, 210]],
                       index=pd.date_range('2025-01-01', periods=2), columns=columns)
    result = _extract_close(raw, ['AAPL', 'MSFT'])
    assert result.to_dict('list') == {'AAPL': [10, 11], 'MSFT': [20, 21]}


def test_extract_close_ticker_price_layout():
    columns = pd.MultiIndex.from_tuples([
        ('AAPL', 'Close'), ('AAPL', 'Volume'), ('MSFT', 'Close'), ('MSFT', 'Volume')])
    raw = pd.DataFrame([[10, 100, 20, 200]],
                       index=pd.date_range('2025-01-01', periods=1), columns=columns)
    result = _extract_close(raw, ['AAPL', 'MSFT'])
    assert result.iloc[0].to_dict() == {'AAPL': 10, 'MSFT': 20}


def test_extract_close_single_ticker_plain_columns():
    raw = pd.DataFrame({'Close': [10, 11]}, index=pd.date_range('2025-01-01', periods=2))
    result = _extract_close(raw, ['AAPL'])
    assert result['AAPL'].tolist() == [10, 11]
