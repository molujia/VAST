"""Leakage-safe supervision-mode validation shared by lightweight tests."""


_VALID_SUPERVISION_MODES = {
    "auto",
    "semi_supervised",
    "oracle_budget",
    "oracle_full",
}


def validate_supervision_request(supervision_mode: str, query_strategy: str) -> str:
    """Validate a request and return its explicit effective supervision mode."""

    mode = str(supervision_mode or "auto").strip().lower()
    strategy = str(query_strategy or "").strip().lower()
    if mode not in _VALID_SUPERVISION_MODES:
        raise ValueError("unsupported supervision_mode: %s" % supervision_mode)
    if mode == "semi_supervised" and strategy == "oracle_label_coverage":
        raise ValueError(
            "oracle_label_coverage is label-aware and cannot be used with semi_supervised mode"
        )
    if mode == "auto":
        if strategy == "oracle_label_coverage":
            return "oracle_budget"
        return "semi_supervised"
    return mode
