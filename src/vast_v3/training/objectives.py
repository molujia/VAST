"""Masked-view preparation and the actual PN / observation / source objectives."""
from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from math import ceil, log1p
import hashlib
import random
import torch
from torch import nn
from ..data.schema import ObservableCase, MaskedTarget
from ..data.preprocessing import sanitize_log
from ..encoders.frontends import initialize
from ..models.ranker import VASTModel
from ..models.rsc import RSCDecoder, labeled_source_library, unlabeled_source_library, source_set_mask
from .losses import class_normalized_bce, hierarchical_recovery_loss, labeled_source_loss, unlabeled_source_loss
from .variants import variant_components


def keyed_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256(repr((seed,*parts)).encode()).digest()[:8],'big')


@dataclass(frozen=True)
class PreparedView:
    case: ObservableCase
    targets: tuple[MaskedTarget,...]
    blocks: frozenset[tuple[str,str,int]]
    original_blocks: frozenset[tuple[str,str,int]]


class EpochViews:
    def __init__(self,preprocessing,*,seed=42,max_cached_cases=2):
        if type(max_cached_cases) is not int or max_cached_cases < 0:
            raise ValueError('max_cached_cases capacity must be a nonnegative integer')
        self.pre=preprocessing; self.seed=seed; self.max_cached_cases=max_cached_cases
        self._epoch=None; self._cache=OrderedDict()

    def prepare(self,case,*,epoch):
        if epoch!=self._epoch:
            self._cache.clear(); self._epoch=epoch
        if case.case_id in self._cache:
            self._cache.move_to_end(case.case_id)
            return self._cache[case.case_id]
        blocks=set(); responses=set()
        types={e.id:e.category for e in case.entities}
        edge_types={r.id:types[r.source]+'->'+types[r.target] for r in case.relations}
        for s in case.numeric:
            category=types[s.owner] if s.modality=='metric' else edge_types[s.owner]
            unit='node' if s.modality=='metric' else 'edge'
            for t,v,m in zip(s.times,s.values,s.observed):
                if not m or not case.reference_start<=t<case.incident_end:
                    continue
                value,supported=self.pre.transform(category,s.channel,v)
                if not supported:
                    continue
                block=(unit,s.owner,case.bin_index(t,self.pre.delta)); blocks.add(block)
                if abs(value)>1:
                    responses.add(block)
        for log in case.logs:
            if case.reference_start<=log.time<case.incident_end:
                blocks.add(('node',log.owner,case.bin_index(log.time,self.pre.delta)))
        selected=set()
        if len(blocks)>=2:
            rng=random.Random(keyed_seed(self.seed,case.case_id,epoch,'mask'))
            count=ceil(.2*len(blocks)); uniform=count//2; response=count-uniform
            selected.update(rng.sample(sorted(blocks),uniform))
            pool=sorted(responses-selected)
            selected.update(rng.sample(pool,min(response,len(pool))))
            selected.update(rng.sample(sorted(blocks-selected),count-len(selected)))
        masked,targets=case.mask_blocks(selected,delta=self.pre.delta)
        result=PreparedView(masked,targets,frozenset(selected),frozenset(blocks))
        # Only immutable observations and targets are cached. The keyed RNG
        # above reconstructs exactly the same mask after eviction/resumption;
        # keeping an entire epoch's full masked pool resident is unnecessary.
        if self.max_cached_cases:
            self._cache[case.case_id]=result
            while len(self._cache)>self.max_cached_cases:
                self._cache.popitem(last=False)
        return result


class TrainingObjectives(nn.Module):
    names=('pn','source','local_recovery','process_recovery','invariance','U_soft')

    def __init__(self, model: VASTModel, *, variant='A3', max_roots=None, seed=42, text_reference_mean=None):
        super().__init__(); self.model=model; self.max_roots=max_roots
        self.variant=variant; self.enabled_components=variant_components(variant)
        self.views=EpochViews(model.frontend.preprocessing,seed=seed)
        self._libraries={}; self._library_epoch=None
        nc=len(model.frontend.channels)
        self.decoder=RSCDecoder(entity_types=len(model.frontend.entity_categories),
                                relation_types=len(model.frontend.relation_categories),metric_channels=nc,trace_channels=nc)
        self.local_heads=nn.ModuleDict({'metric':nn.Linear(64,nc),'trace':nn.Linear(64,nc),
                                       'text':nn.Linear(64,384),'log_frequency':nn.Linear(64,1)})
        self.process_heads=nn.ModuleDict({'metric':nn.Linear(64,nc),'trace':nn.Linear(128,nc),
                                         'text':nn.Linear(64,384),'log_frequency':nn.Linear(64,1)})
        initialize(self.local_heads); initialize(self.process_heads)
        # Stored common modules keep the reference initialization sequence equal
        # across controls. Disabled modules never enter loss/optimizer paths.
        # A4's extra parameters use a separate RNG scope, leaving all matching
        # parameters and the caller's subsequent random draws unchanged.
        if variant=='A4':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(keyed_seed(seed,'A4-visible-bypass'))
                self.bypass_heads=nn.ModuleDict({'metric':nn.Linear(64,nc,bias=False),
                    'trace':nn.Linear(128,nc,bias=False),'text':nn.Linear(64,384,bias=False),
                    'log_frequency':nn.Linear(64,1,bias=False)})
                initialize(self.bypass_heads)
        if text_reference_mean is None:
            text_reference_mean=model.frontend.preprocessing.text_reference_mean
        mean=torch.as_tensor(text_reference_mean,dtype=torch.float32)
        if mean.numel() not in (0,384) or not torch.isfinite(mean).all():
            raise ValueError('fixed text target mean must be absent or finite 384-vector')
        self.register_buffer('text_reference_mean',mean)

    def library_membership(self,case_id,epoch):
        return self._libraries[(case_id,epoch)]

    def _targets(self, original, prepared, output):
        """Bin-local fixed target coordinates; missing coordinates remain NaN."""
        pre=self.model.frontend.preprocessing
        ids={e.id:i for i,e in enumerate(original.entities)}
        types={e.id:e.category for e in original.entities}
        edges={r.id:i for i,r in enumerate(original.relations)}
        edge_types={r.id:types[r.source]+'->'+types[r.target] for r in original.relations}
        channel={name:i for i,name in enumerate(self.model.frontend.channels)}
        values=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        text_targets=[t for t in prepared.targets if t.modality=='log']
        frequency_targets=defaultdict(int)
        for t in text_targets:
            frequency_targets[(t.owner,original.bin_index(t.time,pre.delta))]+=t.frequency
        for (owner,b),frequency in frequency_targets.items():
            residual,supported=pre.transform(types[owner],'__log_frequency__',log1p(frequency))
            if supported:
                values[('log_frequency',owner)][b][0].append(residual)
        embedded=None
        if text_targets and self.text_reference_mean.numel():
            texts=[sanitize_log(str(t.value),tuple(ids)) for t in text_targets]
            embedded=self.model.frontend.text_encoder(texts).detach().to(output.scores)
        for t in prepared.targets:
            if t.modality=='log':
                continue
            if t.channel not in channel:
                continue
            category=types[t.owner] if t.modality=='metric' else edge_types[t.owner]
            value,supported=pre.transform(category,t.channel,float(t.value))
            if supported:
                b=original.bin_index(t.time,pre.delta)
                values[(t.modality,t.owner)][b][channel[t.channel]].append(value)
        # Text targets are independent of the trainable log projection. A
        # missing training-reference center makes this head ineligible.
        if embedded is not None:
            grouped=defaultdict(list)
            for j,t in enumerate(text_targets):
                grouped[(t.owner,original.bin_index(t.time,pre.delta))].append((embedded[j]-self.text_reference_mean,t.frequency))
            for (owner,b),entries in grouped.items():
                target=sum(v*f for v,f in entries)/sum(f for _,f in entries)
                for d,v in enumerate(target.tolist()):
                    values[('text',owner)][b][d].append(v)
        records=[]
        for (modality,owner),by_bin in values.items():
            dim=384 if modality=='text' else 1 if modality=='log_frequency' else len(channel)
            bins=sorted(by_bin)
            target=output.scores.new_full((len(bins),dim),float('nan'))
            for j,b in enumerate(bins):
                for d,observed in by_bin[b].items():
                    target[j,d]=sum(observed)/len(observed)
            records.append({'modality':modality,'unit':owner,'bins':bins,'target':target,
                            'index':edges[owner] if modality=='trace' else ids[owner]})
        return records

    def _recovery(self,output,records,*,process):
        built=[]; front=output.front
        for record in records:
            modality=record['modality']; v=record['index']; predictions=[]; eligibility=[]
            for j,b in enumerate(record['bins']):
                if process:
                    t=output.origin+(b+.5)*output.delta
                    if modality=='trace':
                        u,w,_=front.relations[v]
                        pu=self.model.process_at(output,u,t); pw=self.model.process_at(output,w,t)
                        context=torch.cat((pu,pw)) if pu is not None and pw is not None else None
                    else:
                        context=self.model.process_at(output,v,t)
                elif modality=='trace':
                    context=front.edge_recovery_states[v,b] if front.edge_context_mask[v,b] else None
                elif modality in ('text','log_frequency'):
                    context=front.recovery_local[v,b] if front.log_context_mask[v,b] else None
                else:
                    context=front.recovery_local[v,b] if front.local_context_mask[v,b] else None
                valid=context is not None
                head=self.process_heads[modality] if process else self.local_heads[modality]
                predictions.append(head(context) if valid else record['target'].new_zeros(record['target'].shape[-1]))
                eligibility.append(valid)
            target=record['target']
            if not process and modality=='trace':
                target=target.clone()
                for j,b in enumerate(record['bins']):
                    for d,name in enumerate(self.model.frontend.channels):
                        if not front.trace_channel_context.get((v,b,name),False):
                            target[j,d]=float('nan')
            built.append({'modality':modality,'unit':record['unit'],'prediction':torch.stack(predictions),
                          'target':target,'eligible':torch.tensor(eligibility,device=target.device,dtype=torch.bool)})
        return hierarchical_recovery_loss(built)

    def _source_errors(self,output,records,library):
        if not records or not output.front.telemetry_mask.any():
            return None,None
        front=output.front
        candidate_mask=source_set_mask(library,len(output.candidate_ids),device=output.scores.device)
        sources=torch.zeros(len(library),front.h.shape[0],device=output.scores.device,dtype=torch.bool)
        sources[:,list(front.candidate_indices)]=candidate_mask
        states=self.decoder(front.h,front.telemetry_mask,front.entity_types,front.relations,front.active,sources)
        nodes=self.decoder.predict_nodes(states); edges=self.decoder.predict_edges(states,front.relations)
        if self.variant=='A4':
            # h is recomputed from the masked raw view; no withheld value or
            # target-derived feature enters this deliberately unconstrained arm.
            nodes={name:value+self.bypass_heads[name](front.h).unsqueeze(0) for name,value in nodes.items()}
            owners=[u for u,_,_ in front.relations]; targets=[v for _,v,_ in front.relations]
            paired=torch.cat((front.h[owners],front.h[targets]),-1)
            edges=edges+self.bypass_heads['trace'](paired).unsqueeze(0)
        built=[]
        for record in records:
            prediction=edges if record['modality']=='trace' else nodes[record['modality']]
            built.append({'modality':record['modality'],'unit':record['unit'],
                          'prediction':prediction[:,record['index'],record['bins']], 'target':record['target'],
                          'eligible':torch.ones(len(record['bins']),device=output.scores.device,dtype=torch.bool)})
        return hierarchical_recovery_loss(built),candidate_mask

    def case_losses(self,case,*,epoch,roots=None,seed=42,components=None):
        requested=set(self.names if components is None else components)
        if requested-set(self.names):
            raise ValueError('unknown objective component')
        requested &= self.enabled_components
        result={name:None for name in self.names}
        if not requested:
            return result
        if epoch!=self._library_epoch:
            self._libraries={}; self._library_epoch=epoch
        if roots is not None and requested & {'pn','source'}:
            roots=frozenset(roots)
            if not roots or not roots<=set(case.candidate_ids):
                raise ValueError('complete queried roots must lie in the candidate catalog')
            if self.max_roots is not None and len(roots)>self.max_roots:
                raise ValueError('queried roots contradict K_max')
        if roots is not None and 'pn' in requested:
            full=self.model(case,seed=keyed_seed(seed,case.case_id,epoch,'fragments'))
            flags=torch.tensor([x in roots for x in full.candidate_ids],device=full.scores.device)
            result['pn']=class_normalized_bce(full.scores,flags)
        if not requested-{'pn'}:
            return result
        prepared=self.views.prepare(case,epoch=epoch)
        output=self.model(prepared.case,seed=keyed_seed(seed,case.case_id,epoch,'fragments'))
        records=self._targets(case,prepared,output)
        if 'local_recovery' in requested:
            result['local_recovery']=self._recovery(output,records,process=False)
        if 'process_recovery' in requested:
            result['process_recovery']=self._recovery(output,records,process=True)
        if 'invariance' in requested and len(output.unmarked):
            coupling=({'sampling_audit':output.sampling_audit} if self.model.fragment_sampler=='exact-direct-v1'
                      else {'sampling_marks':output.sampling_marks})
            shifted=self.model(prepared.case.shift_time(32*output.delta),
                               seed=keyed_seed(seed,case.case_id,epoch,'fragments'),**coupling)
            result['invariance']=(output.unmarked-shifted.unmarked).square().mean()
        if roots is not None and 'source' in requested:
            true=frozenset(i for i,v in enumerate(case.candidate_ids) if v in roots)
            library=(true,) if self.variant=='A5' else labeled_source_library(len(case.candidate_ids),true,
                                            K_max=self.max_roots,seed=keyed_seed(seed,case.case_id,epoch,'L'))
            errors,_=self._source_errors(output,records,library)
            if errors is not None:
                result['source']=errors[0] if self.variant=='A5' else labeled_source_loss(errors[0],errors[1:])
        if roots is None and 'U_soft' in requested:
            key=(case.case_id,epoch)
            if key not in self._libraries:
                self._libraries[key]=unlabeled_source_library(output.scores,K_max=self.max_roots,
                                                             seed=keyed_seed(seed,case.case_id,epoch,'U'))
            errors,membership=self._source_errors(output,records,self._libraries[key])
            if errors is not None:
                result['U_soft']=unlabeled_source_loss(output.scores,membership,errors)
        return result
