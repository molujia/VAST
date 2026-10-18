import importlib.util
import torch
from vast_v3.data.schema import Entity, NumericSeries, LogRecord, Relation, ObservableCase
from vast_v3.data.preprocessing import Preprocessor


def fixture():
    entities = (Entity('a','service'),Entity('b','service'),Entity('c','service'))
    series = (NumericSeries('metric','a','cpu',(0.,1.,2.,3.,4.,5.),(0.,1.,2.,2.,6.,2.)),
              NumericSeries('trace','ab','latency',(0.,1.,2.,3.,4.,5.),(1.,2.,3.,8.,1.,2.)),
              NumericSeries('trace','cb','latency',(0.,1.,2.,3.,4.,5.),(2.,1.,3.,1.,9.,1.)))
    case = ObservableCase('x',entities,0.,3.,6.,series,(LogRecord('a',3.,'a failed',2),),
                          (Relation('ab','a','b','call',0.,6.),Relation('cb','c','b','call',0.,6.)))
    return case, Preprocessor.fit([case],permitted_case_ids={'x'})


def model(pre):
    assert importlib.util.find_spec('vast_v3.encoders.frontends') is not None, 'native encoder is not implemented'
    from vast_v3.encoders.frontends import NativeFrontend
    # A frozen deterministic embedding fixture, never a production text-model substitute.
    return NativeFrontend(pre,('service',),('call',),('cpu','latency'),
                          text_encoder=lambda texts: torch.ones(len(texts),384))


def test_native_frontend_retains_owner_table_and_shapes():
    c,p = fixture(); m=model(p); f=m(c)
    assert f.h.shape == f.local.shape == (3,6,64)
    assert f.edge_states.shape == (2,6,64)
    assert f.edge_owner_ids == ('ab','cb')
    assert not torch.equal(f.edge_states[0],f.edge_states[1])
    assert torch.isfinite(f.h).all()
    assert m.metric_cnn is not m.trace_cnn


def test_masked_trace_target_has_context_but_is_not_observed_event():
    c,p=fixture(); m=model(p)
    masked,_=c.mask_blocks({('edge','ab',3)},delta=p.delta)
    f=m(masked)
    assert not f.edge_mask[0,3]
    assert f.edge_context_mask[0,3]
    assert f.edge_states[0,3].abs().sum() == 0
    assert f.edge_recovery_states[0,3].abs().sum() > 0
    f.edge_recovery_states[0,3].square().mean().backward()
    assert m.trace_cnn[0].weight.grad.abs().sum() > 0


def test_no_numeric_context_gives_zero_and_not_healthy_observation():
    c,p=fixture(); m=model(p)
    empty=ObservableCase('missing',c.entities,0.,3.,6.)
    f=m(empty)
    assert f.local.abs().sum() == 0
    assert f.h.abs().sum() == 0
    assert not f.availability.any()
    assert f.edge_states.shape == (0,6,64)


def test_uniform_time_shift_including_origin_is_exact():
    c,p=fixture(); m=model(p)
    a,b=m(c),m(c.shift_time(100.))
    assert torch.allclose(a.h,b.h,atol=1e-6)
    assert torch.allclose(a.edge_states,b.edge_states,atol=1e-6)


def test_frontend_does_not_read_withheld_value():
    from dataclasses import replace
    c,p=fixture(); m=model(p)
    changed=replace(c,numeric=(replace(c.numeric[0],values=(0.,1.,2.,999.,6.,2.)),)+c.numeric[1:])
    a,_=c.mask_blocks({('node','a',3)},delta=p.delta)
    b,_=changed.mask_blocks({('node','a',3)},delta=p.delta)
    assert torch.equal(m(a).h,m(b).h)


def test_native_sequence_order_reaches_embedding_before_bin_pool():
    from dataclasses import replace
    c,p=fixture(); p=replace(p,delta=3.)
    # Native observations swap inside the same bin: pre-bin averaging would lose it.
    other=replace(c,numeric=(replace(c.numeric[0],values=(2.,1.,0.,2.,6.,2.)),)+c.numeric[1:])
    torch.manual_seed(42); m=model(p)
    assert not torch.allclose(m(c).local,m(other).local)
