"""Random acquisition consumes public IDs and access counts, never annotations."""
import inspect
import random

import pytest

from vast_v3.acquisition.baselines import RandomCaseSelector
from vast_v3.data.schema import Entity, LabelStore, ObservableCase


def test_random_selector_has_total_access_budget_interface():
    selector = RandomCaseSelector(('a', 'b', 'c'), priority=('c', 'a', 'b'))
    method = getattr(selector, 'select_budgeted', None)
    assert callable(method), 'random control needs an access-counted total-budget API'
    assert method(('a',), total_budget=2, batch_size=5) == ('c',)
    assert method(('a', 'c'), total_budget=2, batch_size=5) == ()


def test_random_total_budget_counts_failed_root_access_not_only_revealed_labels():
    case = ObservableCase('a', (Entity('x', 'service'), Entity('y', 'service')), 0., 1., 2.)
    ledger = LabelStore({'a': ('x', 'y')}, budget=1, max_roots=1)
    with pytest.raises(ValueError, match='prior'):
        ledger.query(case)
    assert ledger.spent == 1 and not ledger.queried
    selector = RandomCaseSelector(('a', 'b'))
    method = getattr(selector, 'select_budgeted', None)
    assert callable(method), 'budget must use the full access ledger, including failed answers'
    assert method(ledger.state_dict()['accessed'], total_budget=1) == ()
    assert ledger.spent == 1 and not ledger.queried


@pytest.mark.parametrize('budget', [-1, 1.5, True, False])
def test_random_total_budget_requires_nonnegative_integer_not_boolean(budget):
    selector = RandomCaseSelector(('a', 'b'))
    method = getattr(selector, 'select_budgeted', None)
    assert callable(method), 'random control needs an access-counted total-budget API'
    with pytest.raises(ValueError, match='budget'):
        method((), total_budget=budget)


def test_random_total_budget_rejects_already_overspent_access_ledger():
    selector = RandomCaseSelector(('a', 'b', 'c'))
    method = getattr(selector, 'select_budgeted', None)
    assert callable(method), 'random control needs an access-counted total-budget API'
    with pytest.raises(ValueError, match='exceeds'):
        method(('a', 'b'), total_budget=1)


@pytest.mark.parametrize('accessed', [('a', 'a'), ('outside',)])
def test_random_total_budget_rejects_duplicate_or_unknown_access_ids(accessed):
    selector = RandomCaseSelector(('a', 'b'))
    method = getattr(selector, 'select_budgeted', None)
    assert callable(method), 'random control needs an access-counted total-budget API'
    with pytest.raises(ValueError, match='partition'):
        method(accessed, total_budget=3)


@pytest.mark.parametrize(('batch', 'remaining'), [(True, 2), (2, True), (False, 2),
                                                  (-1, 2), (2, -1), (1.5, 2), (2, 1.5)])
def test_random_batch_and_remaining_budget_are_strict_nonnegative_integers(batch, remaining):
    with pytest.raises(ValueError, match='integer'):
        RandomCaseSelector(('a', 'b')).select(batch_size=batch, remaining_budget=remaining)


@pytest.mark.parametrize('case_ids', [('a', ''), ('a', 'a'), (1, 2), ('a', None)])
def test_random_pool_requires_unique_nonempty_public_string_ids(case_ids):
    with pytest.raises(ValueError, match='IDs'):
        RandomCaseSelector(case_ids)


@pytest.mark.parametrize(('labeled', 'unlabeled'), [
    (('a', 'a'), None), (('outside',), None), (('a',), ('a', 'b')),
    ((), ('a',)), ((), ('a', 'a', 'b')), ((), ('a', 'b', 'outside')),
])
def test_random_rejects_incomplete_overlapping_or_invalid_partitions(labeled, unlabeled):
    with pytest.raises(ValueError, match='partition'):
        RandomCaseSelector(('a', 'b')).select(labeled, unlabeled)


def test_random_seeded_priority_matches_uniform_permutation_without_label_filtering():
    ids = ('d', 'a', 'c', 'b')
    expected = sorted(ids)
    random.Random(42).shuffle(expected)
    first = RandomCaseSelector(iter(ids), seed=42)
    reordered = RandomCaseSelector(tuple(reversed(ids)), seed=42)
    assert first.priority == reordered.priority == tuple(expected)
    assert first.select(batch_size=99, remaining_budget=99) == tuple(expected)


def test_random_restored_priority_is_used_without_reading_labels_or_global_rng():
    before = random.getstate()
    priority = ('b', 'c', 'a')
    selector = RandomCaseSelector(('a', 'b', 'c'), seed=9, priority=priority)
    assert selector.select(('b',), ('a', 'c'), batch_size=2, remaining_budget=1) == ('c',)
    assert selector.priority == priority
    assert random.getstate() == before
    # Public API permits IDs/counts only: no annotation store, root/type fields,
    # model scores, observation features, or test labels can be supplied.
    assert set(inspect.signature(RandomCaseSelector).parameters) == {'case_ids', 'seed', 'priority'}
    assert set(inspect.signature(selector.select).parameters) == {
        'labeled_ids', 'unlabeled_ids', 'batch_size', 'remaining_budget'}
    with pytest.raises(TypeError):
        RandomCaseSelector(('a', 'b'), root_labels={'a': ('root',)})


@pytest.mark.parametrize('priority', [('a',), ('a', 'a'), ('a', 'outside')])
def test_random_restore_rejects_priority_not_permutation_of_complete_pool(priority):
    with pytest.raises(ValueError, match='priority'):
        RandomCaseSelector(('a', 'b'), priority=priority)


def test_random_limits_selection_by_batch_remaining_budget_and_unaccessed_capacity():
    selector = RandomCaseSelector(('a', 'b', 'c'), priority=('a', 'b', 'c'))
    assert selector.select(batch_size=1, remaining_budget=2) == ('a',)
    assert selector.select(('a',), batch_size=9, remaining_budget=9) == ('b', 'c')
    assert selector.select(batch_size=0, remaining_budget=2) == ()
    assert selector.select(batch_size=2, remaining_budget=0) == ()
    assert RandomCaseSelector(()).select() == ()


def test_random_total_budget_rounds_do_not_repeat_accessed_cases_or_mutate_inputs():
    selector = RandomCaseSelector(('a', 'b', 'c', 'd'), priority=('d', 'b', 'c', 'a'))
    method = getattr(selector, 'select_budgeted', None)
    assert callable(method), 'random control needs an access-counted total-budget API'
    accessed = []
    first = method(accessed, total_budget=3, batch_size=2)
    assert accessed == [] and first == ('d', 'b')
    accessed.extend(first)
    second = method(accessed, total_budget=3, batch_size=2)
    assert accessed == ['d', 'b'] and second == ('c',)
    assert method(first+second, total_budget=3) == ()
