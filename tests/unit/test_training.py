import importlib.util
import torch
from test_ranker import setup


def api():
    assert importlib.util.find_spec('vast_v3.training.objectives') is not None, 'joint objective missing'
    from vast_v3.training.objectives import TrainingObjectives, EpochViews
    return TrainingObjectives,EpochViews


def test_one_mask_per_epoch_targets_are_actually_deleted():
    c,m=setup(); _,Views=api(); views=Views(m.frontend.preprocessing,seed=42)
    a=views.prepare(c,epoch=0); b=views.prepare(c,epoch=0)
    assert a is b
    assert 0 < len(a.blocks) < len(a.original_blocks)
    assert a.targets
    for target in a.targets:
        for series in a.case.numeric:
            if series.owner==target.owner and series.channel==target.channel:
                assert all(not seen for t,seen in zip(series.times,series.observed) if t==target.time)


def test_joint_objective_has_labelled_and_unlabelled_gradients():
    c,m=setup(); Obj,_=api(); obj=Obj(m,max_roots=2)
    components=obj.case_losses(c,epoch=0,roots={'a'},seed=42)
    assert components['pn'] is not None and components['source'] is not None
    assert components['U_soft'] is None
    sum(x for x in components.values() if x is not None).backward()
    assert m.rank[-1].weight.grad.abs().sum()>0
    obj.zero_grad()
    components=obj.case_losses(c,epoch=1,roots=None,seed=42)
    assert components['pn'] is None and components['source'] is None
    assert components['U_soft'] is not None
    components['U_soft'].backward()
    assert m.rank[-1].weight.grad.abs().sum()>0


def test_empty_observations_skip_recovery_not_pn():
    from dataclasses import replace
    c,m=setup(); Obj,_=api(); obj=Obj(m,max_roots=2)
    components=obj.case_losses(replace(c,numeric=(),relations=()),epoch=0,roots={'a'})
    assert components['pn'] is not None
    assert all(components[k] is None for k in ('source','U_soft','local_recovery','process_recovery','invariance'))


def test_U_library_membership_cached_but_scores_recomputed():
    c,m=setup(); Obj,_=api(); obj=Obj(m,max_roots=2)
    a=obj.case_losses(c,epoch=0,roots=None,seed=42)
    membership=obj.library_membership(c.case_id,0)
    with torch.no_grad():
        m.rank[-1].bias.add_(1.)
    b=obj.case_losses(c,epoch=0,roots=None,seed=42)
    assert obj.library_membership(c.case_id,0)==membership
    assert not torch.allclose(a['U_soft'],b['U_soft'])
