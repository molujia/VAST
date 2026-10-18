"""Budgeted closed-loop training with independent objective streams and resume."""
from __future__ import annotations
from dataclasses import asdict
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import os
import time
import tempfile
import torch
from .schedule import TrainingSchedule, IndependentEligibleStreams
from .checkpoint import save_checkpoint, load_checkpoint
from .query_trace import OrderedQueryTrace
from .objectives import keyed_seed
from ..acquisition.propcover import CaseTokens, PropCover


class Trainer:
    def __init__(self,objectives,cases,labels,schedule: TrainingSchedule,*,checkpoint_dir,identity,seed=42,pool_size=None):
        self.objectives=objectives; self.cases=cases; self.labels=labels; self.schedule=schedule
        self.pool_size=len(cases) if pool_size is None else pool_size
        schedule.update_slots(self.pool_size)
        if self.pool_size<len(cases):
            raise ValueError('pool_size cannot be smaller than available case IDs')
        self.directory=Path(checkpoint_dir); self.directory.mkdir(parents=True,exist_ok=True)
        self.identity=dict(identity); self.seed=seed
        self.query_trace=None
        if any(not isinstance(k,str) or not k for k in self.cases):
            raise ValueError('pool keys must be nonempty case identities')
        self.optimizer=torch.optim.Adam(objectives.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.state={'stage_index':0,'epoch':0,'slot':0,'updates':0,'query_stages':[],'epochs':[],
                    'stream':None,'epoch_totals':{},'epoch_counts':{}}

    def _persist_query(self,state):
        payload={'identity':self.identity,'labels':state}
        if self.query_trace is not None:
            payload['query_plans']=self.query_trace.plan_hashes
        descriptor,name=tempfile.mkstemp(prefix='.query-access-',suffix='.tmp',dir=self.directory)
        temporary=Path(name)
        try:
            with os.fdopen(descriptor,'w',encoding='utf-8') as handle:
                json.dump(payload,handle,sort_keys=True); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary,self.directory/'query-access.json')
        finally:
            if temporary.exists():
                temporary.unlink()
        if self.query_trace is not None:
            self.query_trace.reconcile(state,binding=payload['query_plans'])
            self.query_trace.save()

    def _access(self,cid):
        return self.labels.query(self._case(cid),persist=self._persist_query)

    def _case(self,cid):
        case=self.cases[cid]
        if cid!=case.case_id:
            raise ValueError('pool keys must match case identities')
        return case

    def _validate_label_ledger(self,regime,fixed_query_ids,*,ledger=None):
        """Check paid/reserved IDs without resolving any pending root answer."""
        if ledger is None:
            accessed=self.labels.queried|self.labels.pending
            spent=self.labels.spent
        else:
            # Inspect ID metadata before LabelStore restores any root values.
            accessed=frozenset(ledger['accessed'])|frozenset(ledger['revealed'])
            spent=len(ledger['accessed'])
        limit=self.pool_size if regime in ('all_label','lofo') else min(self.schedule.budget,self.pool_size)
        if len(accessed)!=spent or spent>limit:
            raise ValueError('query ledger exceeds the declared run budget')
        if not accessed<=set(self.cases):
            raise ValueError('query ledger contains cases outside the permitted pool')
        if regime=='fixed_list' and not accessed<=set(fixed_query_ids):
            raise ValueError('query ledger contains paid or reserved roots outside the frozen fixed list')

    def _status(self,*,complete=False):
        return {'complete':complete,'updates':self.state['updates'],'queried_ids':sorted(self.labels.queried),
                'unqueried_ids':sorted(set(self.cases)-self.labels.queried),'epochs':self.state['epochs'],
                'pool_size':self.pool_size,'unqueried_count':self.pool_size-len(self.labels.queried),
                'query_order':self.query_trace.query_order if self.query_trace is not None else None,
                'variant':self.objectives.variant,'preprocessing':self.preprocessing_metadata}

    def _save(self,stages,*,complete=False):
        index=self.state['stage_index']
        stage=stages[index]['stage'] if index<len(stages) else 'finished'
        libraries=[{'case_id':cid,'epoch':epoch,'members':[sorted(s) for s in lib]}
                   for (cid,epoch),lib in self.objectives._libraries.items()]
        save_checkpoint(self.directory/'latest.pt',model=self.objectives,optimizer=self.optimizer,identity=self.identity,
                        progress={'stage':stage,'epoch':self.state['epoch'],'slot':self.state['slot'],
                                  'encoder_version':self.state['updates'],'stage_complete':complete},
                        query_sets={'L':sorted(self.labels.queried),'U':sorted(set(self.cases)-self.labels.queried)},
                        extra_state={'loop':self.state,'libraries':libraries,'library_epoch':self.objectives._library_epoch,
                                     'query_plans':self.query_trace.plan_hashes if self.query_trace is not None else None,
                                     'query_trace':self.query_trace.state_dict() if self.query_trace is not None else None})
        if self.query_trace is not None:
            self.query_trace.save()
        status={**self._status(complete=complete),'stage':stage,'epoch':self.state['epoch'],'slot':self.state['slot'],
                'unix_time':time.time(),'identity':self.identity}
        temporary=self.directory/'progress.json.tmp'
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
        os.replace(temporary,self.directory/'progress.json')

    def _query(self,count,*,stages):
        record=self.query_trace.record_for_stage(self.state['stage_index'])
        if record is not None:
            for cid in record['selected_ids']:
                if cid not in self.labels.queried:
                    self._access(cid)
            return
        # Acquisition does not read labels. All L/U representations are encoded
        # with this one version before any new root query or optimizer update.
        snapshots=[]; model=self.objectives.model
        with torch.no_grad():
            for cid in sorted(self.cases):
                output=model(self._case(cid),seed=keyed_seed(self.seed,cid,'selection'))
                snapshots.append(CaseTokens(cid,output.unmarked.cpu(),tuple(f.family for f in output.fragments),
                                            tuple(f.provenance for f in output.fragments),str(self.state['updates'])))
        selector=PropCover(snapshots,seed=self.seed)
        selection=selector.select(labeled_ids=self.labels.queried,
                                  unlabeled_ids=set(self.cases)-self.labels.queried,batch_size=count,
                                  remaining_budget=self.schedule.budget-self.labels.spent)
        self.query_trace.append(stage_index=self.state['stage_index'],update_version=self.state['updates'],
                                labeled_ids=self.labels.queried,spent=self.labels.spent,
                                selection=selection,selector=selector)
        # Bind the full plan and capture post-selection RNG before any answer.
        self._save(stages)
        for cid in selection.case_ids:
            self._access(cid)

    def _eligible(self,epoch,weights):
        """Freeze eligibility, masks and U memberships without optimizer updates.

        This conservative reference pass uses actual head eligibility, not
        heuristics based on case length. It is separately reported as setup work.
        """
        names=self.objectives.names
        populations={name:[] for name in names}
        with torch.no_grad():
            for cid in sorted(self.cases):
                labeled=cid in self.labels.queried
                requested={name for name in names if weights[name] and
                           (labeled if name in ('pn','source') else not labeled if name=='U_soft' else True)}
                if not requested:
                    continue
                roots=self._access(cid) if requested & {'pn','source'} else None
                values=self.objectives.case_losses(self._case(cid),epoch=epoch,roots=roots,seed=self.seed,components=requested)
                for name in requested:
                    if values[name] is not None:
                        populations[name].append(cid)
        return populations

    def run(self,*,regime='closed_loop',resume=False,max_updates=None,fixed_query_ids=None):
        a0=self.objectives.variant=='A0'
        if a0 and regime=='closed_loop':
            raise ValueError('A0 requires a delivered fixed list or all-label regime; it cannot acquire from U')
        if self.pool_size!=len(self.cases) and not (a0 and regime=='fixed_list'):
            raise ValueError('this regime requires the complete permitted pool')
        stages=self.schedule.stages(regime,pool_size=self.pool_size,a0=a0)
        pre=self.objectives.model.frontend.preprocessing
        if regime=='fixed_list':
            fixed_query_ids=tuple(fixed_query_ids or ())
            if (len(fixed_query_ids)!=min(self.schedule.budget,self.pool_size)
                    or len(set(fixed_query_ids))!=len(fixed_query_ids)
                    or not set(fixed_query_ids)<=set(self.cases)):
                raise ValueError('fixed-list regime requires one frozen budget-sized list')
            if set(pre.fitted_case_ids)!=set(fixed_query_ids):
                raise ValueError('preprocessing must be fitted on exactly the fixed-list L reference cases')
        self._validate_label_ledger(regime,fixed_query_ids)
        settings={'variant':self.objectives.variant,'schedule':asdict(self.schedule),'seed':self.seed,
                  'mask_seed':self.objectives.views.seed,
                  'regime':regime,'pool_size':self.pool_size,'max_roots':self.objectives.max_roots,
                  'fixed_query_ids':list(fixed_query_ids) if regime=='fixed_list' else None}
        self.identity.update(variant=self.objectives.variant,
                             training_hash=hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                             learner_preprocessing_hash=pre.identity)
        self.preprocessing_metadata={'learner_hash':pre.identity,'fitted_case_ids':list(pre.fitted_case_ids),
                                     'scope':'fixed-L-reference' if regime=='fixed_list' else 'permitted-training-reference',
                                     'producer_hash':self.identity.get('producer_preprocessing_hash')}
        self.query_trace=(OrderedQueryTrace(self.directory/'query-trace.json',identity=self.identity,
                                           pool_ids=self.cases,schedule=self.schedule,stages=stages,seed=self.seed)
                          if regime=='closed_loop' else None)
        checkpoint=self.directory/'latest.pt'
        if resume:
            payload=load_checkpoint(checkpoint,model=self.objectives,optimizer=self.optimizer,
                                    expected_identity=self.identity)
            self.state=payload['extra_state']['loop']
            journal=self.directory/'query-access.json'
            access=None
            if journal.exists():
                access=json.loads(journal.read_text(encoding='utf-8'))
                if access['identity']!=self.identity:
                    raise ValueError('query-access identity mismatch')
                self._validate_label_ledger(regime,fixed_query_ids,ledger=access['labels'])
            if self.query_trace is not None:
                self.query_trace.load(checkpoint_trace=payload['extra_state'].get('query_trace'))
                checkpoint_binding=payload['extra_state'].get('query_plans')
                if checkpoint_binding is None:
                    raise ValueError('checkpoint lacks ordered query-trace binding; cannot invent historical order')
                ledger=access['labels'] if access is not None else self.labels.state_dict()
                self.query_trace.reconcile(ledger,binding=access.get('query_plans') if access is not None else [],
                                           checkpoint_binding=checkpoint_binding,loop_state=self.state)
            if access is not None:
                self.labels.load_state_dict(access['labels'])
                if not set(payload['query_sets']['L'])<=self.labels.queried:
                    raise ValueError('query-access ledger lost checkpoint-confirmed labels')
            elif payload['query_sets']['L']:
                raise ValueError('checkpoint labels require the write-ahead query-access ledger')
            self.objectives._library_epoch=payload['extra_state']['library_epoch']
            self.objectives._libraries={(r['case_id'],r['epoch']):tuple(frozenset(s) for s in r['members'])
                                        for r in payload['extra_state']['libraries']}
            self._validate_label_ledger(regime,fixed_query_ids)
            if self.query_trace is not None:
                self.query_trace.save()
        elif checkpoint.exists() or (self.directory/'query-access.json').exists() or (self.directory/'query-trace.json').exists():
            raise ValueError('existing checkpoint requires explicit resume; no overwrite/restart')
        else:
            if self.query_trace is not None:
                if self.labels.spent:
                    raise ValueError('closed-loop labels require an existing ordered query-trace')
            self._save(stages,complete=not stages)
        # A failed validation is a durable blocker; an interrupted reservation
        # can only read its already-paid case again, not a different free label.
        if regime!='closed_loop':
            for cid in sorted(self.labels.pending):
                self._access(cid)
        if regime in ('all_label','lofo'):
            for cid in sorted(self.cases):
                self._access(cid)
        if regime=='fixed_list':
            for cid in fixed_query_ids:
                if cid not in self.labels.queried:
                    self._access(cid)
        slots=self.schedule.update_slots(self.pool_size); started=self.state['updates']
        while self.state['stage_index']<len(stages):
            si=self.state['stage_index']; stage=stages[si]
            if regime=='closed_loop' and stage['query_count'] and si not in self.state['query_stages']:
                self._query(stage['query_count'],stages=stages)
                self.state['query_stages'].append(si); self._save(stages)
            while self.state['epoch']<stage['epochs']:
                epoch=self.state['epoch']; global_epoch=sum(s['epochs'] for s in stages[:si])+epoch
                weights=self.schedule.loss_weights(self.objectives.variant,epoch=epoch,phase=stage['phase'],u_enabled=stage['u_enabled'])
                if self.state['stream'] is None:
                    populations=self._eligible(global_epoch,weights)
                    streams=IndependentEligibleStreams(populations,seed=keyed_seed(self.seed,global_epoch,'stream'),
                                                       batch_size=self.schedule.case_batch_size)
                else:
                    stored=self.state['stream']
                    streams=IndependentEligibleStreams(stored['populations'],seed=0,batch_size=self.schedule.case_batch_size)
                    streams.load_state_dict(stored)
                while self.state['slot']<slots:
                    batches=streams.draw_all(); requested=defaultdict(set)
                    for name,batch in batches.items():
                        if weights[name]:
                            for cid in batch:
                                requested[cid].add(name)
                    self.optimizer.zero_grad(set_to_none=True)
                    has_grad=False
                    for cid,names in requested.items():
                        roots=self._access(cid) if names & {'pn','source'} else None
                        values=self.objectives.case_losses(self._case(cid),epoch=global_epoch,roots=roots,seed=self.seed,components=names)
                        loss=None
                        for name in names:
                            value=values[name]
                            if value is None:
                                raise RuntimeError('head eligibility changed inside frozen epoch')
                            if not torch.isfinite(value):
                                raise FloatingPointError(f'nonfinite {name} loss for {cid}')
                            term=value*weights[name]/len(batches[name]); loss=term if loss is None else loss+term
                            self.state['epoch_totals'][name]=self.state['epoch_totals'].get(name,0.)+float(value.detach())
                            self.state['epoch_counts'][name]=self.state['epoch_counts'].get(name,0)+1
                        if loss is not None and loss.requires_grad:
                            loss.backward(); has_grad=True
                    if has_grad:
                        torch.nn.utils.clip_grad_norm_(self.objectives.parameters(),5.,error_if_nonfinite=True)
                        self.optimizer.step()
                    self.state['updates']+=1; self.state['slot']+=1; self.state['stream']=streams.state_dict()
                    self._save(stages)
                    if max_updates is not None and self.state['updates']-started>=max_updates:
                        return self._status()
                report={'stage':stage['stage'],'epoch':epoch,'global_epoch':global_epoch,
                        'update_slots':slots,'weights':weights,
                        'eligible_counts':{name:len(ids) for name,ids in streams.populations.items()}}
                for name in self.objectives.names:
                    count=self.state['epoch_counts'].get(name,0)
                    report[name+'_count']=count
                    report[name+'_mean']=self.state['epoch_totals'].get(name,0.)/count if count else None
                self.state['epochs'].append(report)
                self.state.update(epoch=epoch+1,slot=0,stream=None,epoch_totals={},epoch_counts={})
                self._save(stages)
            self.state.update(stage_index=si+1,epoch=0,slot=0,stream=None)
            self._save(stages,complete=self.state['stage_index']==len(stages))
        return self._status(complete=True)
