"""Normal-window sampling with explicit fault guard bands."""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .common import DaySpan, FaultCase, NormalWindow, make_case_id


Interval = Tuple[int, int]


def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    sorted_intervals = sorted(intervals)
    if not sorted_intervals:
        return []

    merged = [sorted_intervals[0]]
    for start_ts, end_ts in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start_ts <= last_end:
            merged[-1] = (last_start, max(last_end, end_ts))
            continue
        merged.append((start_ts, end_ts))
    return merged


def subtract_intervals(day_span: Interval, blocked_spans: Sequence[Interval]) -> List[Interval]:
    allowed = []
    cursor = day_span[0]
    day_end = day_span[1]

    for blocked_start, blocked_end in blocked_spans:
        blocked_start = max(blocked_start, day_span[0])
        blocked_end = min(blocked_end, day_end)
        if blocked_end <= cursor:
            continue
        if blocked_start > cursor:
            allowed.append((cursor, blocked_start))
        cursor = max(cursor, blocked_end)
    if cursor < day_end:
        allowed.append((cursor, day_end))
    return allowed


def sample_normal_windows(
    dataset: str,
    day_spans: Sequence[DaySpan],
    fault_cases: Sequence[FaultCase],
    window_size_seconds: int,
    guard_band_seconds: int,
    target_windows_by_day: Mapping[str, int],
) -> Tuple[List[NormalWindow], Dict[str, object]]:
    """Sample non-overlapping safe normal windows from admissible day spans."""

    faults_by_day = defaultdict(list)
    for case in fault_cases:
        faults_by_day[case.day].append(case)

    windows = []
    safe_spans_by_day = {}
    admitted_counts = Counter()
    requested_counts = Counter(target_windows_by_day)

    for span in day_spans:
        blocked = []
        for case in faults_by_day.get(span.day, []):
            blocked.append(
                (
                    max(span.start_ts, case.start_ts - guard_band_seconds),
                    min(span.end_ts, case.end_ts + guard_band_seconds),
                )
            )
        merged_blocked = merge_intervals(blocked)
        safe_spans = subtract_intervals((span.start_ts, span.end_ts), merged_blocked)
        safe_spans_by_day[span.day] = safe_spans

        requested = target_windows_by_day.get(span.day, 0)
        sampled = 0
        for safe_start, safe_end in safe_spans:
            cursor = safe_start
            while cursor + window_size_seconds <= safe_end and sampled < requested:
                window_id = make_case_id(dataset, "normal", span.day, sampled)
                windows.append(
                    NormalWindow(
                        dataset=dataset,
                        window_id=window_id,
                        day=span.day,
                        start_ts=cursor,
                        end_ts=cursor + window_size_seconds,
                        source="safe_gap",
                        metadata={
                            "guard_band_seconds": guard_band_seconds,
                            "safe_span_start_ts": safe_start,
                            "safe_span_end_ts": safe_end,
                            "sampling_order": sampled,
                        },
                    )
                )
                sampled += 1
                admitted_counts[span.day] += 1
                cursor += window_size_seconds
            if sampled >= requested:
                break

    diagnostics = {
        "window_size_seconds": window_size_seconds,
        "guard_band_seconds": guard_band_seconds,
        "requested_counts_by_day": dict(requested_counts),
        "admitted_counts_by_day": dict(admitted_counts),
        "shortfall_by_day": {
            day: max(0, requested_counts[day] - admitted_counts.get(day, 0))
            for day in requested_counts
        },
        "safe_spans_by_day": {
            day: [{"start_ts": start_ts, "end_ts": end_ts} for start_ts, end_ts in spans]
            for day, spans in safe_spans_by_day.items()
        },
    }
    return windows, diagnostics
