"""One shared SCOPE encoder for acquisition, ranking and anchored recovery."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import torch
from torch import Tensor, nn
from ..data.schema import ObservableCase
from ..encoders.frontends import NativeFrontend, FrontendOutput, initialize
from ..encoders.fragments import EventRef, FragmentEdge, Fragment, extract_fragments
from ..encoders.scope import ScopeEncoder


@dataclass
class ModelOutput:
    scores: Tensor
    candidate_ids: tuple[str,...]
    front: FrontendOutput
    fragments: tuple[Fragment,...]
    unmarked: Tensor
    marked: dict[int,tuple[Tensor,Tensor,tuple[int,...]]]
    process: Tensor
    sampling_marks: dict[str,float]
    fragment_candidates: int
    origin: float
    delta: float
    sampling_audit: dict | None = None


class AnchoredAttention(nn.Module):
    def __init__(self, *, timed=False):
        super().__init__(); self.timed=timed
        self.key_z=nn.Linear(64,64,bias=False)
        self.key_role=nn.Linear(32,64,bias=False)
        self.key_time=nn.Linear(3,64,bias=False) if timed else None
        self.logit=nn.Linear(64,1,bias=False)
        self.value=nn.Linear(96+(3 if timed else 0),64)
        initialize(self)

    def forward(self,z,role,times=None):
        key=self.key_z(z)+self.key_role(role)
        inputs=[z,role]
        if self.timed:
            if times is None:
                raise ValueError('recovery attention requires target-relative visible times')
            key=key+self.key_time(times); inputs.append(times)
        weights=torch.softmax(self.logit(torch.tanh(key)).squeeze(-1),0)
        return (weights[:,None]*self.value(torch.cat(inputs,-1))).sum(0)


class VASTModel(nn.Module):
    def __init__(self, frontend: NativeFrontend, *, fragment_sampler='exact-direct-v1', max_enumerated=100000):
        super().__init__()
        if fragment_sampler not in ('exact-direct-v1','enumerate-v1'):
            raise ValueError('unsupported fragment sampler')
        self.fragment_sampler=fragment_sampler
        if fragment_sampler=='exact-direct-v1':
            from ..encoders.fragment_cache import DirectSamplingCache
            self.sampling_cache=DirectSamplingCache()
        else:
            self.sampling_cache=None
        self.frontend=frontend
        self.scope=ScopeEncoder(entity_types=len(frontend.entity_categories),relation_types=len(frontend.relation_categories))
        self.readout=AnchoredAttention()
        self.recovery_readout=AnchoredAttention(timed=True)
        self.rank=nn.Sequential(nn.Linear(128+64+7,64),nn.GELU(),nn.Linear(64,1))
        self.max_enumerated=max_enumerated  # Safety guard for enumerate-v1 only.
        initialize(self.rank)

    def _inventory(self,case,front):
        events=[]; by_entity=defaultdict(list)
        delta=self.frontend.preprocessing.delta
        for v,entity in enumerate(case.entities):
            for b in front.telemetry_mask[v].nonzero().flatten().tolist():
                start=case.reference_start+b*delta
                event=EventRef(v,b,start,min(start+delta,case.incident_end),int(front.entity_types[v]),
                               f'{case.case_id}/{entity.id}/{b}')
                by_entity[v].append(len(events)); events.append(event)
        links=[]
        owner_index={r.id:i for i,r in enumerate(case.relations)}
        for edge_index,(u,v,kind) in enumerate(front.relations):
            relation=case.relations[edge_index]
            for i in by_entity[u]:
                for j in by_entity[v]:
                    a,b=events[i],events[j]
                    if i==j or abs(a.bin-b.bin)>9:
                        continue
                    start,end=min(a.start,b.start),max(a.end,b.end)
                    # Require observed interval support or the explicitly
                    # approved, separately audited invariant structural prior.
                    if relation.is_active(start,end):
                        links.append(FragmentEdge(i,j,kind,owner_index[relation.owner or relation.id],relation.response))
        return events,links

    def forward(self,case: ObservableCase, *, seed=0,sampling_marks=None,sampling_audit=None):
        if self.fragment_sampler=='exact-direct-v1' and sampling_marks:
            raise ValueError('exact-direct-v1 does not accept legacy sampling_marks; use sampling_audit')
        if self.fragment_sampler=='enumerate-v1' and sampling_audit is not None:
            raise ValueError('enumerate-v1 does not accept direct sampling_audit')
        front=self.frontend(case)
        events,links=self._inventory(case,front)
        if self.fragment_sampler=='exact-direct-v1':
            from ..encoders.direct_sampling import extract_fragments_direct
            extracted=extract_fragments_direct(events,links,delta=self.frontend.preprocessing.delta,seed=seed,
                sampling_audit=sampling_audit,owner_keys=dict(enumerate(front.edge_owner_ids)),
                sampling_cache=self.sampling_cache)
            audit=extracted.sampling_audit
        else:
            extracted=extract_fragments(events,links,delta=self.frontend.preprocessing.delta,seed=seed,
                sampling_marks=sampling_marks,max_enumerated=self.max_enumerated,
                owner_keys=dict(enumerate(front.edge_owner_ids)))
            audit=None
        fragments=extracted.fragments
        args=(front.h,front.local,front.edge_states,front.availability,front.coverage,front.edge_mask)
        unmarked=[]; marked_by_entity=defaultdict(list)
        for i,fragment in enumerate(fragments):
            z,_=self.scope(fragment,*args)
            unmarked.append(z)
            for v in sorted({ev.entity for ev in fragment.events}):
                z,role=self.scope(fragment,*args,candidate=v)
                marked_by_entity[v].append((z,role,i))
        zero=front.h.new_zeros(64)
        marked={}; process=[]
        for v in range(len(case.entities)):
            tokens=marked_by_entity.get(v,[])
            if tokens:
                z=torch.stack([x[0] for x in tokens]); role=torch.stack([x[1] for x in tokens])
                marked[v]=(z,role,tuple(x[2] for x in tokens))
                process.append(self.readout(z,role))
            else:
                process.append(zero)
        process=torch.stack(process)
        scores=[]
        for v in front.candidate_indices:
            valid=front.local_mask[v]
            local=front.local[v,valid]
            pooled=torch.cat((local.mean(0),local.max(0).values)) if len(local) else zero.new_zeros(128)
            missingness=torch.cat((front.availability[v].float().mean(0),front.coverage[v].mean(0),
                                  zero.new_tensor([v in marked])))
            scores.append(self.rank(torch.cat((pooled,process[v],missingness))).squeeze())
        return ModelOutput(torch.stack(scores),case.candidate_ids,front,fragments,
                           torch.stack(unmarked) if unmarked else zero.new_zeros(0,64),marked,process,
                           extracted.sampling_marks,extracted.candidate_count,case.reference_start,
                           self.frontend.preprocessing.delta,sampling_audit=audit)

    def process_at(self,output: ModelOutput,entity: int,target_time: float):
        """Recovery-only, target-time readout; None means ineligible, not zero error."""
        if entity not in output.marked:
            return None
        z,role,indices=output.marked[entity]
        relative=[]
        for index in indices:
            times=[((ev.start+ev.end)/2)-target_time for ev in output.fragments[index].events]
            relative.append([sum(times)/len(times),min(times),max(times)])
        return self.recovery_readout(z,role,z.new_tensor(relative))
