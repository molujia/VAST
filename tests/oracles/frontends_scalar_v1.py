"""Native-time encoders followed by entity/time association (never the reverse)."""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, isfinite
from collections import defaultdict
from typing import Callable
import torch
from torch import Tensor, nn
from ..data.schema import ObservableCase
from ..data.preprocessing import Preprocessor, sanitize_log
from .log_pooling import pool_encoded_logs


def initialize(module: nn.Module):
    for layer in module.modules():
        if isinstance(layer,(nn.Linear,nn.Conv1d)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer,nn.LayerNorm):
            nn.init.ones_(layer.weight); nn.init.zeros_(layer.bias)
        elif isinstance(layer,nn.Embedding):
            nn.init.xavier_uniform_(layer.weight)


@dataclass
class FrontendOutput:
    h: Tensor
    local: Tensor
    edge_states: Tensor
    availability: Tensor
    coverage: Tensor
    edge_mask: Tensor
    edge_context_mask: Tensor
    edge_coverage: Tensor
    entity_types: Tensor
    relations: tuple[tuple[int,int,int],...]
    active: Tensor
    edge_owner_ids: tuple[str,...]
    telemetry_mask: Tensor
    local_mask: Tensor
    candidate_indices: tuple[int,...]
    # Metric context and per-channel trace context are retained for recovery only.
    metric_context_mask: Tensor
    trace_channel_context: dict[tuple[int,int,str],bool]
    recovery_local: Tensor
    edge_recovery_states: Tensor
    local_context_mask: Tensor
    log_context_mask: Tensor


class NativeFrontend(nn.Module):
    def __init__(self, preprocessing: Preprocessor, entity_categories: tuple[str,...],
                 relation_categories: tuple[str,...], channels: tuple[str,...], *,
                 text_encoder: Callable[[list[str]],Tensor], log_recovery_radius_steps: float = 2.0):
        super().__init__()
        if not entity_categories or not relation_categories or not channels:
            raise ValueError('frozen category, relation and channel vocabularies are required')
        radius = float(log_recovery_radius_steps)
        if not isfinite(radius) or radius < 0:
            raise ValueError('log recovery radius must be finite and nonnegative')
        self.log_recovery_radius_steps = radius
        self.preprocessing = preprocessing
        self.entity_categories = tuple(entity_categories)
        self.relation_categories = tuple(relation_categories)
        self.channels = tuple(channels)
        self._categories = {v:i for i,v in enumerate(entity_categories)}
        self._relations = {v:i for i,v in enumerate(relation_categories)}
        self._channels = {v:i for i,v in enumerate(channels)}
        self.metric_cnn = nn.Sequential(nn.Conv1d(3,64,3,padding=1),nn.GELU(),nn.Conv1d(64,64,3,padding=1),nn.GELU())
        self.trace_cnn = nn.Sequential(nn.Conv1d(3,64,3,padding=1),nn.GELU(),nn.Conv1d(64,64,3,padding=1),nn.GELU())
        self.metric_channel = nn.Embedding(len(channels),64)
        self.trace_channel = nn.Embedding(len(channels),64)
        self.metric_pool = nn.Linear(128,64)
        self.trace_pool = nn.Linear(128,64)
        self.numeric_mix = nn.Linear(64*(1+2*len(relation_categories)),64)
        self.log_mlp = nn.Sequential(nn.Linear(386,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.topology_input = nn.Linear(len(entity_categories)+2*len(relation_categories),64)
        self.topology_self = nn.ModuleList([nn.Linear(64,64) for _ in range(2)])
        self.topology_relation = nn.ModuleList([nn.ModuleList([nn.Linear(64,64,bias=False) for _ in relation_categories]) for _ in range(2)])
        self.local_projection = nn.Linear(128,64)
        self.fusion = nn.Sequential(nn.Linear(3*64+6,64),nn.LayerNorm(64))
        initialize(self)
        # Register only after initializing our own layers; pretrained weights
        # must never be visited by the Xavier initializer.
        self.text_encoder = text_encoder
        if isinstance(text_encoder,nn.Module):
            text_encoder.requires_grad_(False); text_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if isinstance(self.text_encoder,nn.Module):
            self.text_encoder.eval()
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def _numeric(self, case, modality, owner_ids, categories, bins):
        """Convolve native sequences; associate their outputs with bins afterwards.

        A withheld native position may have a contextual state but its observed
        flag stays false. Mean/max is across channel states, not raw samples.
        """
        states = defaultdict(list)
        recovery_states = defaultdict(list)
        observed_counts = defaultdict(int)
        total_counts = defaultdict(int)
        channel_context = {}
        owner_lookup = {owner:i for i,owner in enumerate(owner_ids)}
        cnn = self.metric_cnn if modality == 'metric' else self.trace_cnn
        embeddings = self.metric_channel if modality == 'metric' else self.trace_channel
        projection = self.metric_pool if modality == 'metric' else self.trace_pool
        for series in case.numeric:
            if series.modality != modality or series.channel not in self._channels or not series.times:
                continue
            owner = owner_lookup[series.owner]
            indices=[j for j,t in enumerate(series.times) if case.reference_start<=t<case.incident_end]
            if not indices:
                continue
            times=[series.times[j] for j in indices]
            converted = [self.preprocessing.transform(categories[series.owner],series.channel,series.values[j]) for j in indices]
            visible = [series.observed[j] and ok for j,(_,ok) in zip(indices,converted)]
            values = [v if m else 0. for (v,_),m in zip(converted,visible)]
            elapsed = [0.] + [b-a for a,b in zip(times,times[1:])]
            native = torch.tensor([values,visible,elapsed],device=self.device,dtype=torch.float32).unsqueeze(0)
            hidden = cnn(native)[0].T + embeddings.weight[self._channels[series.channel]]
            # Two kernel-3 layers have native-position receptive radius two.
            context = [any(visible[max(0,j-2):min(len(visible),j+3)]) for j in range(len(visible))]
            grouped = defaultdict(list)
            for j,t in enumerate(times):
                b = case.bin_index(t,self.preprocessing.delta)
                if 0 <= b < bins:
                    total_counts[(owner,b)] += 1
                    observed_counts[(owner,b)] += int(visible[j])
                    grouped[b].append(j)
            for b, indices in grouped.items():
                valid = [j for j in indices if visible[j]]
                contextual = [j for j in indices if context[j]]
                channel_context[(owner,b,series.channel)] = bool(contextual)
                # Observed bins use only observed native positions. A fully
                # withheld bin may expose context solely to a recovery query.
                if valid:
                    states[(owner,b)].append(hidden[valid].mean(0))
                if contextual:
                    recovery_states[(owner,b)].append(hidden[contextual].mean(0))
        n = len(owner_ids)
        rows = []; recovery_rows=[]; masks=[]; contexts=[]; coverage=[]
        zero = next(self.parameters()).new_zeros(64)
        for owner in range(n):
            row=[]; recrow=[]; maskrow=[]; contextrow=[]; covrow=[]
            for b in range(bins):
                channels = states.get((owner,b),[])
                if channels:
                    stacked=torch.stack(channels)
                    row.append(projection(torch.cat((stacked.mean(0),stacked.max(0).values))))
                else:
                    row.append(zero)
                contextual=recovery_states.get((owner,b),[])
                if contextual:
                    stacked=torch.stack(contextual)
                    recrow.append(projection(torch.cat((stacked.mean(0),stacked.max(0).values))))
                else:
                    recrow.append(zero)
                num=observed_counts.get((owner,b),0); den=total_counts.get((owner,b),0)
                maskrow.append(num > 0); contextrow.append(bool(contextual)); covrow.append(num/max(den,1))
            rows.append(torch.stack(row)); recovery_rows.append(torch.stack(recrow)); masks.append(maskrow); contexts.append(contextrow); coverage.append(covrow)
        shape=(n,bins)
        return (torch.stack(rows) if rows else zero.new_zeros(n,bins,64),
                torch.tensor(masks,device=self.device,dtype=torch.bool).reshape(shape),
                torch.tensor(contexts,device=self.device,dtype=torch.bool).reshape(shape),
                torch.tensor(coverage,device=self.device,dtype=torch.float32).reshape(shape),channel_context,
                torch.stack(recovery_rows) if recovery_rows else zero.new_zeros(n,bins,64))

    def forward(self, case: ObservableCase) -> FrontendOutput:
        delta=self.preprocessing.delta
        bins=ceil((case.incident_end-case.reference_start)/delta)
        n=len(case.entities); e=len(case.relations); nr=len(self.relation_categories)
        ids={ent.id:i for i,ent in enumerate(case.entities)}
        categories={ent.id:ent.category for ent in case.entities}
        if any(cat not in self._categories for cat in categories.values()):
            raise ValueError('entity category absent from frozen vocabulary')
        if any(r.kind not in self._relations for r in case.relations):
            raise ValueError('relation category absent from frozen vocabulary')
        types=torch.tensor([self._categories[x.category] for x in case.entities],device=self.device)
        relations=tuple((ids[r.source],ids[r.target],self._relations[r.kind]) for r in case.relations)
        edge_categories={r.id:categories[r.source]+'->'+categories[r.target] for r in case.relations}
        metric,metric_mask,metric_context,metric_cov,_,metric_recovery=self._numeric(case,'metric',tuple(ids),categories,bins)
        edge,edge_mask,edge_context,edge_cov,channel_context,edge_recovery=self._numeric(case,'trace',tuple(r.id for r in case.relations),edge_categories,bins)
        active=torch.tensor([[r.is_active(case.reference_start+b*delta,case.reference_start+(b+1)*delta)
                              for b in range(bins)] for r in case.relations],device=self.device,dtype=torch.bool).reshape(e,bins)
        # Native text encoding occurs before bin association. Frozen embeddings
        # are detached, while the log MLP remains trainable.
        visible_logs=[log for log in case.logs if case.reference_start <= log.time < case.incident_end]
        encoded=metric.new_zeros(0,64)
        if visible_logs:
            texts=[sanitize_log(log.text,tuple(ids)) for log in visible_logs]
            with torch.no_grad():
                embedded=self.text_encoder(texts).detach().to(device=self.device,dtype=torch.float32)
            if embedded.shape != (len(texts),384) or not torch.isfinite(embedded).all():
                raise ValueError('fixed text encoder must return finite [records,384] embeddings')
            extras=torch.tensor([[torch.log1p(torch.tensor(float(log.frequency))).item(),log.time-case.reference_start]
                                 for log in visible_logs],device=self.device)
            encoded=self.log_mlp(torch.cat((embedded,extras),-1))
        log_states,log_mask,log_recovery,log_context=pool_encoded_logs(
            encoded,[ids[log.owner] for log in visible_logs],[log.time for log in visible_logs],
            [log.frequency for log in visible_logs],entity_count=n,bins=bins,
            reference_start=case.reference_start,delta=delta,radius_steps=self.log_recovery_radius_steps)
        zero=metric.new_zeros(64)
        # E^t remains untouched; only a relation-indexed summary is added to
        # the numerical node branch. No anonymous averaging destroys E^t.
        incoming=[[[] for _ in range(nr)] for _ in range(n)]
        outgoing=[[[] for _ in range(nr)] for _ in range(n)]
        for j,(u,v,r) in enumerate(relations):
            incoming[v][r].append(j); outgoing[u][r].append(j)
        numeric=[]; num_valid=[]; num_coverage=[]; topo=[]; topo_valid=[]
        for b in range(bins):
            numeric_rows=[]; numeric_flags=[]; numeric_cov=[]; degrees=[]; valid=[]
            for v in range(n):
                pieces=[metric[v,b] if metric_mask[v,b] else zero]
                seen=[metric_mask[v,b]]; cov=[metric_cov[v,b]]
                degree=[]
                for direction in (incoming,outgoing):
                    for r in range(nr):
                        attached=[j for j in direction[v][r] if active[j,b]]
                        measured=[j for j in attached if edge_mask[j,b]]
                        pieces.append(edge[measured,b].mean(0) if measured else zero)
                        seen.append(bool(measured)); cov.append(edge_cov[measured,b].mean() if measured else metric.new_tensor(0.))
                        degree.append(len(attached))
                is_valid=any(bool(x) for x in seen)
                numeric_rows.append(self.numeric_mix(torch.cat(pieces)) if is_valid else zero)
                numeric_flags.append(is_valid); numeric_cov.append(torch.stack(cov).mean())
                degrees.append(degree); valid.append(any(degree))
            deg=torch.tensor(degrees,device=self.device,dtype=torch.float32)
            top=self.topology_input(torch.cat((torch.nn.functional.one_hot(types,len(self._categories)).float(),torch.log1p(deg)),-1))
            for layer in range(2):
                rows=[]
                for v in range(n):
                    result=self.topology_self[layer](top[v])
                    for r in range(nr):
                        neighbors=[relations[j][0] for j in incoming[v][r] if active[j,b]]
                        if neighbors:
                            result=result+self.topology_relation[layer][r](top[neighbors].mean(0))
                    rows.append(torch.nn.functional.gelu(result) if valid[v] else zero)
                top=torch.stack(rows)
            numeric.append(torch.stack(numeric_rows)); num_valid.append(numeric_flags); num_coverage.append(torch.stack(numeric_cov))
            topo.append(top); topo_valid.append(valid)
        numeric=torch.stack(numeric,1); topology=torch.stack(topo,1)
        num_mask=torch.tensor(num_valid,device=self.device,dtype=torch.bool).T
        top_mask=torch.tensor(topo_valid,device=self.device,dtype=torch.bool).T
        availability=torch.stack((num_mask,log_mask,top_mask),-1)
        coverage=torch.stack((torch.stack(num_coverage,1),log_mask.float(),top_mask.float()),-1)
        fused=self.fusion(torch.cat((numeric,log_states,topology,availability.float(),coverage),-1))
        fused=fused*availability.any(-1,keepdim=True)
        # Contextual metric/log states support local reconstruction but cannot
        # create observed events or enter h/local at unobserved positions.
        local_mask=metric_mask|log_mask
        local=self.local_projection(torch.cat((metric,log_states),-1))*local_mask.unsqueeze(-1)
        local_context=metric_context|log_context
        recovery_local=self.local_projection(torch.cat((metric_recovery,log_recovery),-1))*local_context.unsqueeze(-1)
        telemetry=num_mask|log_mask
        return FrontendOutput(fused,local,edge,availability,coverage,edge_mask,edge_context,edge_cov,
                              types,relations,active,tuple(r.id for r in case.relations),telemetry,local_mask,
                              tuple(i for i,ent in enumerate(case.entities) if ent.candidate),metric_context,channel_context,
                              recovery_local,edge_recovery,local_context,log_context)
