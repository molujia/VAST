"""Bounded prepared views preserve the fixed keyed epoch masks exactly."""
from dataclasses import replace
import gc
import random
import weakref

import pytest

from test_ranker import setup
from vast_v3.training.objectives import EpochViews


def test_default_epoch_view_cache_does_not_retain_the_entire_pool():
    case, model = setup()
    views = EpochViews(model.frontend.preprocessing, seed=42)
    first = views.prepare(replace(case, case_id='first'), epoch=0)
    retained = weakref.ref(first)
    del first
    views.prepare(replace(case, case_id='second'), epoch=0)
    views.prepare(replace(case, case_id='third'), epoch=0)
    gc.collect()
    assert len(views._cache) <= 2
    assert retained() is None, 'evicted full masked cases must not stay resident'


def test_eviction_rebuilds_identical_masks_targets_without_consuming_global_rng():
    case, model = setup()
    views = EpochViews(model.frontend.preprocessing, seed=42, max_cached_cases=1)
    state = random.getstate()
    expected = views.prepare(case, epoch=0)
    views.prepare(replace(case, case_id='evict'), epoch=0)
    actual = views.prepare(case, epoch=0)
    assert actual == expected and actual is not expected
    assert random.getstate() == state
    assert views.prepare(case, epoch=0) is actual
    assert views.prepare(case, epoch=1) is not actual
    assert len(views._cache) == 1


def test_zero_capacity_recomputes_without_retaining_views():
    case, model = setup()
    views = EpochViews(model.frontend.preprocessing, max_cached_cases=0)
    first = views.prepare(case, epoch=0)
    second = views.prepare(case, epoch=0)
    assert first == second and first is not second
    assert not views._cache


@pytest.mark.parametrize('capacity', [-1, True, 1.5, None])
def test_invalid_epoch_cache_capacity_is_rejected(capacity):
    case, model = setup()
    with pytest.raises(ValueError, match='capacity|cached'):
        EpochViews(model.frontend.preprocessing, max_cached_cases=capacity)
