import importlib.util
import torch
from vast_v3.data.schema import Entity, NumericSeries, Relation, ObservableCase
from vast_v3.data.preprocessing import Preprocessor
from vast_v3.encoders.frontends import NativeFrontend


def setup():
    assert importlib.util.find_spec('vast_v3.models.ranker') is not None, 'shared ranker missing'
    from vast_v3.models.ranker import VASTModel
    c=ObservableCase('x',(Entity('a','service'),Entity('b','service'),Entity('empty','service')),
                     0.,3.,6.,(NumericSeries('metric','a','cpu',(0.,1.,2.,3.,4.),(0.,1.,0.,3.,2.)),
                     NumericSeries('metric','b','cpu',(0.,1.,2.,3.,4.),(0.,1.,0.,1.,3.))),(),
                     (Relation('ab','a','b','call',0.,6.),))
    pre=Preprocessor.fit([c],permitted_case_ids={'x'})
    front=NativeFrontend(pre,('service',),('call',),('cpu',),text_encoder=lambda ts:torch.zeros(len(ts),384))
    return c,VASTModel(front)


def test_ranker_scores_all_candidates_without_truth_and_backpropagates():
    c,m=setup(); out=m(c,seed=42)
    assert out.candidate_ids == ('a','b','empty')
    assert out.scores.shape == (3,)
    assert torch.isfinite(out.scores).all()
    out.scores.sum().backward()
    assert m.frontend.metric_cnn[0].weight.grad.abs().sum() > 0
    assert m.scope.token_mlp[0].weight.grad is not None


def test_process_recovery_is_time_anchored_not_ranking_readout():
    c,m=setup(); out=m(c,seed=42)
    assert out.fragments
    a=m.process_at(out,0,2.)
    b=m.process_at(out,0,4.)
    assert a is not None and b is not None
    assert not torch.allclose(a,b)
    assert m.process_at(out,2,4.) is None


def test_no_events_keep_full_catalog_and_empty_process():
    from dataclasses import replace
    c,m=setup(); out=m(replace(c,numeric=(),relations=()),seed=42)
    assert not out.fragments
    assert out.process.abs().sum() == 0
    assert out.scores.shape == (3,)


def test_entity_permutation_transports_metadata_and_scores():
    from dataclasses import replace
    c,m=setup(); a=m(c,seed=42)
    b=m(replace(c,entities=tuple(reversed(c.entities))),seed=42,sampling_marks=a.sampling_marks)
    av=dict(zip(a.candidate_ids,a.scores)); bv=dict(zip(b.candidate_ids,b.scores))
    for key in av:
        assert torch.allclose(av[key],bv[key],atol=1e-5)
