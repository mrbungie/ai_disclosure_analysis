"""
scripts/common/pit.py — point-in-time conventions shared by gold and analytics.

Every event row carries the date its information became public
(`available_date`: filing date, call date, trading date). Constructs that
accumulate information (posture, archetypes) are materialized as snapshots on
one fixed calendar: the first day of each quarter (SNAPSHOT_MONTHS). A snapshot
dated `as_of_date` only uses rows with `available_date < as_of_date`, always
the latest ones available.

Joins between an event and a snapshot are as-of joins on the event's date:
the event receives the most recent snapshot whose `as_of_date` is on or before
the event date (the snapshot itself only contains information published
strictly before `as_of_date`, so an event on that same day is point-in-time).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

SNAPSHOT_MONTHS = (1, 4, 7, 10)


def snapshot_dates(start: dt.date | str, end: dt.date | str) -> list[pd.Timestamp]:
    """Quarter-start snapshot dates in [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    first = pd.Timestamp(year=start.year, month=1, day=1)
    dates = pd.date_range(first, end, freq="QS-JAN")
    return [d for d in dates if start <= d <= end and d.month in SNAPSHOT_MONTHS]


def asof_join(events: pd.DataFrame, snapshots: pd.DataFrame, *, event_date: str,
              snapshot_date: str = "as_of_date", by: str | list[str] = "ticker",
              suffixes: tuple[str, str] = ("", "_snapshot"), strict: bool = False) -> pd.DataFrame:
    """Attach to each event the latest snapshot with snapshot_date <= event_date
    (< with `strict`; backward as-of join within `by`). Row order and count of
    `events` are kept."""
    left = events.reset_index(drop=False).rename(columns={"index": "_event_row"})
    left[event_date] = pd.to_datetime(left[event_date]).astype("datetime64[ns]")
    right = snapshots.copy()
    right[snapshot_date] = pd.to_datetime(right[snapshot_date]).astype("datetime64[ns]")
    merged = pd.merge_asof(left.sort_values(event_date), right.sort_values(snapshot_date),
                           left_on=event_date, right_on=snapshot_date, by=by,
                           direction="backward", allow_exact_matches=not strict, suffixes=suffixes)
    return merged.sort_values("_event_row").drop(columns="_event_row").reset_index(drop=True)
