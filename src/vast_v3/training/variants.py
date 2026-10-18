"""The fixed A0--A6 objective matrix (not dataset-specific model branches)."""

NAMES = ('pn', 'source', 'local_recovery', 'process_recovery', 'invariance', 'U_soft')
VARIANTS = {
    'A0': frozenset(('pn',)),
    'A1': frozenset(('pn', 'local_recovery')),
    'A2': frozenset(('pn', 'local_recovery', 'process_recovery', 'invariance')),
    'A3': frozenset(NAMES),
    'A4': frozenset(NAMES),
    'A5': frozenset(NAMES),
    'A6': frozenset(NAMES) - {'U_soft'},
}


def variant_components(variant):
    if variant not in VARIANTS:
        raise ValueError('variant must be one of A0--A6')
    return VARIANTS[variant]
