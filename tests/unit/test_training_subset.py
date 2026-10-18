"""Training view exposes exactly the frozen training IDs, without data reads."""
from collections.abc import Mapping
import importlib
from types import SimpleNamespace

import pytest

from vast_v3.data.manifests import FrozenSplit


def api():
    assert importlib.util.find_spec('vast_v3.data.subsets'), 'training-only Mapping is not implemented'
    return importlib.import_module('vast_v3.data.subsets').TrainingCases


class GuardedRepository(Mapping):
    def __init__(self):
        self.manifest = SimpleNamespace(split=FrozenSplit('aiops2022_pre', ('b', 'a'), ('test',), 'a'*64))
        self.reads = []; self.audits = []
    def __iter__(self):
        return iter(('test', 'a', 'b'))
    def __len__(self):
        return 3
    def __getitem__(self, key):
        self.reads.append(key)
        if key == 'test':
            pytest.fail('test observation accessed through training view')
        return SimpleNamespace(case_id=key)
    def case_audit(self, key):
        self.audits.append(key)
        return {'case_id': key}


def test_training_view_exposes_complete_frozen_order_without_materializing_cases():
    repo = GuardedRepository(); view = api()(repo)
    assert isinstance(view, Mapping)
    assert tuple(view) == ('b', 'a') and len(view) == 2
    assert set(view.keys()) == {'a', 'b'}
    assert 'test' not in view and 'a' in view and repo.reads == []
    assert view['a'].case_id == 'a' and repo.reads == ['a']


@pytest.mark.parametrize('key', ['test', 'unknown'])
def test_test_and_unknown_ids_fail_before_any_underlying_observation_or_audit_access(key):
    repo = GuardedRepository(); view = api()(repo)
    with pytest.raises(KeyError):
        view[key]
    with pytest.raises(KeyError):
        view.case_audit(key)
    assert view.get(key) is None and key not in view
    assert repo.reads == repo.audits == []


def test_training_items_and_audits_stay_within_train_ids():
    repo = GuardedRepository(); view = api()(repo)
    assert [key for key, value in view.items()] == ['b', 'a']
    assert repo.reads == ['b', 'a']
    assert view.case_audit('b') == {'case_id': 'b'}
    assert repo.audits == ['b']


def test_training_view_rejects_underlying_case_identity_mismatch():
    class Wrong(GuardedRepository):
        def __getitem__(self, key):
            return SimpleNamespace(case_id='test')
    with pytest.raises(ValueError, match='identity'):
        api()(Wrong())['a']
