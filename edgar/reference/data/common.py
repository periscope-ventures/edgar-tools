import threading
from importlib import resources

import pandas as pd
import pyarrow.parquet as pq

__all__ = ['read_parquet_from_package', 'read_pyarrow_from_package', 'read_csv_from_package']

# Package parquet files are decoded once and memoized. A plain ``@lru_cache`` is NOT enough for a
# threaded reader: it gates the RESULT, not the function BODY, so on a cold cache N worker threads
# all miss and run the pandas/pyarrow decode of the SAME file at the same instant. pyarrow's
# dictionary-decode path is not thread-safe, and that concurrent cold decode aborts the process
# natively (``parquet::ParquetStatusException`` "Index not in dictionary bounds" -> the runtime's
# ExitError). A double-checked lock around an explicit dict memo fixes both problems: the warm path
# is a lock-free ``dict.get`` (atomic under the GIL) and the cold decode runs exactly once,
# single-threaded, per file. (This also drops the old ``maxsize=1`` cache, which made the two
# reference tables — ``ct.pq`` and ``company_tickers.parquet`` — evict and re-decode each other.)
_parquet_cache: dict = {}
_pyarrow_cache: dict = {}
_decode_lock = threading.Lock()


def read_parquet_from_package(parquet_filename: str) -> pd.DataFrame:
    cached = _parquet_cache.get(parquet_filename)
    if cached is not None:
        return cached

    with _decode_lock:
        cached = _parquet_cache.get(parquet_filename)
        if cached is not None:
            return cached

        package_name = 'edgar.reference.data'
        ref = resources.files(package_name).joinpath(parquet_filename)
        with resources.as_file(ref) as parquet_path:
            df = pd.read_parquet(parquet_path)

        _parquet_cache[parquet_filename] = df
        return df


def read_pyarrow_from_package(parquet_filename: str):
    cached = _pyarrow_cache.get(parquet_filename)
    if cached is not None:
        return cached

    with _decode_lock:
        cached = _pyarrow_cache.get(parquet_filename)
        if cached is not None:
            return cached

        package_name = 'edgar.reference.data'
        ref = resources.files(package_name).joinpath(parquet_filename)
        with resources.as_file(ref) as parquet_path:
            # Read a pyarrow table from a parquet file
            table = pq.read_table(parquet_path)

        _pyarrow_cache[parquet_filename] = table
        return table


def read_csv_from_package(csv_filename: str, **pandas_kwargs):
    package_name = 'edgar.reference.data'

    ref = resources.files(package_name).joinpath(csv_filename)
    with resources.as_file(ref) as csv_path:
        df = pd.read_csv(csv_path, **pandas_kwargs)

    return df
