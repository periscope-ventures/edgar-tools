"""Pre-warm the process-global caches the read path builds lazily, in the current thread.

The reference ticker/CUSIP tables ship as package parquet and are decoded on first use (see
:mod:`edgar.reference.data.common`). Decoding pyarrow on a cold cache from several threads at once
is not thread-safe, so a service that serves reads from a worker-thread pool should warm these
ONCE at startup — before any concurrent request — so every read then hits an already-built cache
and no two threads ever cold-decode the same parquet. ``read_parquet_from_package`` also guards the
decode with a lock, so warming is an optimization (it moves the one-time decode to single-threaded
startup and removes even the first-request lock contention), not the correctness mechanism.

The standardization mappings (JSON) are warmed too, so the first statement/ratio build does no lazy
file IO. All inputs are bundled with the package, so this does no network IO. Idempotent and cheap
once warm; call it after configuring identity/storage and before serving.
"""
from __future__ import annotations


def warm_caches() -> None:
    """Populate the reference-data and standardization-mapping caches, single-threaded."""
    from edgar.entity.mappings_loader import (
        load_canonical_structures,
        load_concept_linkages,
        load_learned_mappings,
        load_virtual_trees,
    )
    from edgar.reference.tickers import cusip_ticker_mapping, get_company_tickers

    # Reference parquet — the only pyarrow decode on the read path. Warming both tables here means
    # no request ever triggers a cold, concurrent parquet decode.
    get_company_tickers()   # company_tickers.parquet
    cusip_ticker_mapping()  # ct.pq

    # Standardization mappings (JSON) the statement/ratio engine reads on the first build.
    load_learned_mappings()
    load_canonical_structures()
    load_virtual_trees()
    load_concept_linkages()


__all__ = ["warm_caches"]
