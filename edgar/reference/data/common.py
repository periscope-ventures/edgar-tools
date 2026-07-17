import threading
from importlib import resources

import pandas as pd
import pyarrow.parquet as pq

__all__ = ['read_parquet_from_package', 'read_pyarrow_from_package', 'read_csv_from_package']

# Package parquet files are RLE_DICTIONARY-encoded and decoded once, then memoized. Two hazards,
# both of which abort the process natively (``parquet::ParquetStatusException`` "Index not in
# dictionary bounds" -> the runtime's ExitError) because Arrow does NOT validate that dictionary
# indices are in bounds (ARROW-1658) — an out-of-bounds index read during a racing decode is
# unguarded:
#
#  1. CROSS-thread: a plain ``@lru_cache`` gates the RESULT, not the BODY, so on a cold cache N
#     worker threads all decode the SAME file at once. The double-checked lock below fixes this —
#     the warm path is a lock-free ``dict.get`` (atomic under the GIL) and the cold decode runs
#     exactly once, per file. (This also drops the old ``maxsize=1`` cache, which made the two
#     reference tables — ``ct.pq`` and ``company_tickers.parquet`` — evict and re-decode each other.)
#  2. INTRA-read: pandas' ``read_parquet`` defaults to pyarrow ``use_threads=True``, fanning the
#     per-column dictionary decode across pyarrow's own C++ thread pool. We read with
#     ``use_threads=False`` so a single decode is serial too — no parallel dictionary decode from
#     either direction.
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
            # use_threads=False: serial decode (no pyarrow thread-pool parallelism). Then
            # MATERIALIZE every column to plain numpy: ``to_pandas()`` returns pyarrow-backed
            # columns (ArrowStringArray) regardless of the pandas ``future.infer_string`` option,
            # and their string/dictionary compute enters pyarrow C++ (GIL released). This frame is
            # cached and SHARED — ``find_cik`` filters it on every ticker lookup — so concurrent
            # filters would run pyarrow kernels on the same array at once and abort the process
            # (``parquet::ParquetStatusException`` "Index not in dictionary bounds" -> ExitError).
            # numpy-backed columns make concurrent reads of the cached frame pure-Python/GIL-safe.
            arrow_df = pq.read_table(parquet_path, use_threads=False).to_pandas()
            df = pd.DataFrame({c: arrow_df[c].to_numpy() for c in arrow_df.columns})

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
            # Read a pyarrow table from a parquet file (serial decode; see read_parquet_from_package).
            table = pq.read_table(parquet_path, use_threads=False)

        _pyarrow_cache[parquet_filename] = table
        return table


def read_csv_from_package(csv_filename: str, **pandas_kwargs):
    package_name = 'edgar.reference.data'

    ref = resources.files(package_name).joinpath(csv_filename)
    with resources.as_file(ref) as csv_path:
        df = pd.read_csv(csv_path, **pandas_kwargs)

    return df
