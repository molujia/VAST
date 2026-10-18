"""Relation query acceleration must preserve the native observation contract."""
from dataclasses import asdict, fields, replace
import math
import pickle
import random

import pytest

from vast_v3.data.schema import Entity, ObservableCase, Relation
from vast_v3.data.serialization import deserialize_observable, serialize_observable


def _naive(relation, start, end):
    if end <= start:
        return False
    return any(start <= a < end if a == b else start < b and end > a
               for a, b in (relation.active_intervals or ((relation.start, relation.end),)))


def _relation(intervals=(), start=-10., end=20.):
    return Relation('ab', 'a', 'b', 'call', start, end,
                    active_intervals=intervals)


def _case(relation):
    return ObservableCase('case', (Entity('a', 'service'), Entity('b', 'service')),
                          -10., 0., 20., relations=(relation,))


class _CountedEndpoint(float):
    """Count actual endpoint work without a timing-dependent speed assertion."""
    comparisons = 0

    def __lt__(self, other):
        type(self).comparisons += 1
        return super().__lt__(other)

    def __le__(self, other):
        type(self).comparisons += 1
        return super().__le__(other)

    def __gt__(self, other):
        type(self).comparisons += 1
        return super().__gt__(other)

    def __ge__(self, other):
        type(self).comparisons += 1
        return super().__ge__(other)

    def __eq__(self, other):
        type(self).comparisons += 1
        return super().__eq__(other)


@pytest.mark.parametrize('kind', ('points', 'continuous', 'mixed'))
def test_query_endpoint_work_is_logarithmic_not_a_full_interval_scan(kind):
    size = 4096
    intervals = tuple((_CountedEndpoint(3 * i),
                       _CountedEndpoint(3 * i + (kind == 'continuous' or (kind == 'mixed' and i % 2))))
                      for i in reversed(range(size)))
    relation = _relation(intervals, end=float(3 * size))
    # A gap inside the relation's hull cannot be answered by a hull shortcut.
    start, end = float(3 * (size // 2) + 1), float(3 * (size // 2) + 2)
    _CountedEndpoint.comparisons = 0
    assert not relation.is_active(start, end)
    assert _CountedEndpoint.comparisons <= 4 * math.ceil(math.log2(size)) + 8


@pytest.mark.parametrize(('intervals', 'start', 'end', 'expected'), [
    (((1., 3.),), 0., 1., False),
    (((1., 3.),), 1., 2., True),
    (((1., 3.),), 2., 3., True),
    (((1., 3.),), 3., 4., False),
    (((3., 3.),), 2., 3., False),
    (((3., 3.),), 3., 4., True),
    (((3., 3.),), 3., 3., False),
    (((1., 3.),), 2., 2., False),
    (((1., 3.), (3., 3.)), 3., 4., True),
    (((1., 3.), (4., 6.)), 3., 4., False),
    (((1., 12.), (2., 3.), (5., 6.)), 8., 9., True),
    (((5., 6.), (2., 3.), (1., 12.)), 12., 13., False),
    (((3., 3.), (1., 2.), (3., 3.), (1., 2.)), 2., 3., False),
    (((3., 3.), (1., 2.), (3., 3.), (1., 2.)), 3., 4., True),
])
def test_exact_half_open_endpoints_and_overlapping_support(intervals, start, end, expected):
    relation = _relation(intervals)
    assert relation.is_active(start, end) is expected
    assert relation.is_active(end, start) is False


@pytest.mark.parametrize(('start', 'end'), ((-2., 2.), (3., 3.)))
def test_empty_subintervals_retain_hull_or_point_fallback(start, end):
    relation = _relation(start=start, end=end)
    for left in (-3., -2., 0., 2., 3., 4.):
        for right in (-3., -2., 0., 2., 3., 4.):
            assert relation.is_active(left, right) is _naive(relation, left, right)


@pytest.mark.parametrize('intervals', ((), ((1., 2.),), ((3., 3.),), ((1., 2.), (3., 3.))))
def test_nonfinite_queries_match_existing_comparison_semantics(intervals):
    relation = _relation(intervals)
    bounds = (-math.inf, -1., 1., 2., 3., 4., math.inf, math.nan)
    for start in bounds:
        for end in bounds:
            assert relation.is_active(start, end) is _naive(relation, start, end)


@pytest.mark.parametrize('intervals', ((), ((1., 3.),), ((3., 3.),)))
def test_numeric_scalar_queries_still_return_builtin_bool(intervals):
    np = pytest.importorskip('numpy')
    relation = _relation(intervals)
    for start, end in ((1., 2.), (3., 4.), (0., 1.), (2., 2.)):
        start, end = np.float64(start), np.float64(end)
        assert relation.is_active(start, end) is _naive(relation, start, end)


@pytest.mark.parametrize('seed', range(8))
def test_random_small_graph_queries_match_naive_oracle(seed):
    rng = random.Random(seed)
    nodes = tuple(Entity(str(i), 'service') for i in range(5))
    for graph in range(12):
        relations = []
        for index in range(rng.randint(1, 8)):
            intervals = []
            for _ in range(rng.randint(0, 20)):
                a = rng.randint(-32, 32) / 4
                b = a if rng.randrange(3) else rng.randint(int(4 * a), 40) / 4
                intervals.append((a, b))
                if rng.randrange(4) == 0:
                    intervals.append((a, b))
            rng.shuffle(intervals)
            relations.append(Relation(str(index), str(rng.randrange(5)), str(rng.randrange(5)),
                                      'call', -10., 10., active_intervals=tuple(intervals)))
        case = ObservableCase(str(graph), nodes, -10., 0., 10., relations=tuple(relations))
        for relation in case.relations:
            endpoints = {value for pair in relation.active_intervals for value in pair}
            queries = [(a, b) for a in endpoints for b in endpoints]
            queries += [(rng.randint(-48, 48) / 4, rng.randint(-48, 48) / 4) for _ in range(30)]
            for start, end in queries:
                assert relation.is_active(start, end) is _naive(relation, start, end)


def test_original_intervals_public_fields_equality_hash_and_repr_are_unchanged():
    intervals = ((3., 3.), (1., 12.), (2., 3.), (3., 3.))
    relation = _relation(intervals)
    expected = dict(id='ab', source='a', target='b', kind='call', start=-10., end=20.,
                    owner=None, response=False, active_intervals=intervals)
    assert relation.active_intervals is intervals
    assert tuple(field.name for field in fields(relation)) == tuple(expected)
    assert asdict(relation) == expected
    assert repr(relation) == 'Relation(' + ', '.join(f'{key}={value!r}' for key, value in expected.items()) + ')'
    assert hash(relation) == hash(tuple(expected.values()))
    assert relation == Relation(**expected)
    relation.is_active(4., 5.)
    assert asdict(relation) == expected
    assert hash(relation) == hash(tuple(expected.values()))


def test_dataclass_replace_rebuilds_support_without_mutating_the_original():
    original = _relation(((1., 2.), (4., 4.)))
    changed = replace(original, active_intervals=((8., 8.), (6., 7.)))
    renamed = replace(original, id='renamed', response=True, owner='ab')
    fallback = replace(original, start=2., end=3., active_intervals=())
    for relation in (original, changed, renamed, fallback):
        for start in range(-1, 11):
            assert relation.is_active(start, start + 1) is _naive(relation, start, start + 1)
    assert original.active_intervals == ((1., 2.), (4., 4.))


def test_mutable_outer_interval_list_keeps_live_query_behavior():
    intervals = [(1., 2.)]
    relation = _relation(intervals)
    assert relation.is_active(1., 2.)
    intervals.append((8., 9.))
    assert relation.is_active(8., 9.)
    intervals[:] = [(4., 4.)]
    assert relation.is_active(4., 5.)
    assert not relation.is_active(1., 2.)
    assert relation.active_intervals is intervals


@pytest.mark.parametrize('container', (tuple, list))
def test_mutable_nested_intervals_remain_consistent_with_pickle_and_replace(container):
    intervals = container(([1., 2.], [3., 3.]))
    relation = _relation(intervals)
    assert relation.is_active(1., 2.)
    intervals[0][:] = [8., 9.]
    intervals[1][:] = [6., 6.]
    copied = pickle.loads(pickle.dumps(relation))
    replaced = replace(relation)
    for start in range(-1, 11):
        for end in range(-1, 11):
            expected = _naive(relation, start, end)
            assert relation.is_active(start, end) is expected
            assert copied.is_active(start, end) is expected
            assert replaced.is_active(start, end) is expected
    assert relation.active_intervals is intervals


def test_case_time_shift_preserves_queries_and_original_interval_audit_order():
    original = _case(_relation(((8., 8.), (1., 12.), (2., 3.), (8., 8.))))
    for offset in (-100., 100.25):
        shifted = original.shift_time(offset)
        assert shifted.relations[0].active_intervals == tuple(
            (a + offset, b + offset) for a, b in original.relations[0].active_intervals)
        for start in range(-12, 23):
            for end in range(-12, 23):
                assert shifted.relations[0].is_active(start + offset, end + offset) is \
                    original.relations[0].is_active(start, end)


def test_json_serialization_has_only_existing_public_fields_and_round_trips():
    original = _case(_relation(((3., 3.), (1., 12.), (2., 3.), (3., 3.))))
    payload = serialize_observable(original)
    assert deserialize_observable(payload) == original
    assert deserialize_observable(payload).relations[0].active_intervals == original.relations[0].active_intervals
    assert all(not key.startswith('_') for key in asdict(original)['relations'][0])
    assert serialize_observable(deserialize_observable(payload)) == payload


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_excludes_derived_state_and_rebuilds_equivalent_support(protocol):
    original = _relation(((8., 8.), (1., 12.), (2., 3.), (8., 8.)))
    # Before indexing, pickle state was exactly the public dataclass attributes.
    assert original.__getstate__() == asdict(original)
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert restored == original
    assert asdict(restored) == asdict(original)
    for start in range(-12, 23):
        for end in range(-12, 23):
            assert restored.is_active(start, end) is _naive(original, start, end)


@pytest.mark.parametrize('intervals', (((2., 1.),), ((-11., 2.),), ((1., 21.),),
                                    ((math.nan, 2.),), ((1., math.inf),)))
def test_invalid_subintervals_still_fail_validation(intervals):
    with pytest.raises(ValueError, match='invalid active relation subinterval'):
        _relation(intervals)
