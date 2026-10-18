import importlib.util
import torch
from test_ranker import setup
from vast_v3.data.schema import LabelStore
from vast_v3.training.objectives import TrainingObjectives
from vast_v3.training.schedule import TrainingSchedule


def trainer_api():
    assert importlib.util.find_spec('vast_v3.training.loop') is not None, 'resumable loop missing'
    from vast_v3.training.loop import Trainer
    return Trainer


def test_closed_loop_budget_schedule_and_resume_do_not_requery(tmp_path):
    from dataclasses import replace
    Trainer=trainer_api(); c,model=setup()
    other=replace(c,case_id='y')
    pool={'x':c,'y':other}; labels=LabelStore({'x':('a',),'y':('b',)},budget=1,max_roots=2)
    objectives=TrainingObjectives(model,max_roots=2)
    schedule=TrainingSchedule(budget=1,query_batch_size=1,warmup_epochs=1,nonfinal_epochs=1,final_epochs=1,u_disabled_epochs=0,u_ramp_epochs=0)
    ident=dict(data_hash='data',config_hash='config',preprocessing_hash='pre',code_hash='code')
    trainer=Trainer(objectives,pool,labels,schedule,checkpoint_dir=tmp_path,identity=ident,seed=42)
    result=trainer.run(max_updates=1)
    assert not result['complete'] and result['updates']==1
    result=trainer.run(resume=True)
    assert result['complete'] and result['updates']==2
    assert labels.spent==1
    done=trainer.run(resume=True)
    assert done==result


def test_lofo_uses_all_permitted_labels_without_U_or_acquisition(tmp_path):
    Trainer=trainer_api(); c,model=setup(); labels=LabelStore({'x':('a',)},budget=1,max_roots=2)
    schedule=TrainingSchedule(budget=1,warmup_epochs=0,final_epochs=1)
    trainer=Trainer(TrainingObjectives(model,max_roots=2),{'x':c},labels,schedule,
                    checkpoint_dir=tmp_path,identity=dict(data_hash='a',config_hash='b',preprocessing_hash='c',code_hash='d'))
    result=trainer.run(regime='lofo')
    assert result['complete'] and result['queried_ids']==['x'] and result['unqueried_ids']==[]
    assert all(event.get('U_soft_count',0)==0 for event in result['epochs'])


def test_failed_root_query_remains_charged_and_blocked_after_restart(tmp_path):
    import pytest
    Trainer=trainer_api(); c,model=setup()
    labels=LabelStore({'x':('a','b')},budget=1,max_roots=1)
    schedule=TrainingSchedule(budget=1,warmup_epochs=0,final_epochs=1)
    identity=dict(data_hash='a',config_hash='b',preprocessing_hash='c',code_hash='d')
    trainer=Trainer(TrainingObjectives(model,max_roots=1),{'x':c},labels,schedule,
                    checkpoint_dir=tmp_path,identity=identity)
    with pytest.raises(ValueError,match='prior'):
        trainer.run()
    assert (tmp_path/'query-access.json').is_file()
    newlabels=LabelStore({'x':('a','b')},budget=1,max_roots=1)
    new=Trainer(TrainingObjectives(setup()[1],max_roots=1),{'x':c},newlabels,schedule,
                checkpoint_dir=tmp_path,identity=identity)
    with pytest.raises(ValueError,match='blocked'):
        new.run(resume=True)
    assert newlabels.spent==1
