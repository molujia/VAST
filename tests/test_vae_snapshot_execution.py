from concurrent.futures import ThreadPoolExecutor
import importlib
from pathlib import Path
import pytest


def api(): return importlib.import_module("rcl_study.vae_snapshot_execution")


def producer(calls,key,text="valid"):
    def work(root):
        calls.append(key); (root/"output.txt").write_text(text)
        return {"outputs":["output.txt"],"metadata":{"key":key}}
    return work


def test_stage_commit_exact_hit_and_targeted_invalidation(tmp_path):
    store=api().StageStore(tmp_path/"run"); calls=[]
    cvae=store.run("cvae",{"inputs":"historical-v1"},producer(calls,"cvae"))
    first=store.run("adapter",{"fit":cvae["artifact_sha256"],"adapter":"v1"},producer(calls,"adapt1"))
    again=store.run("cvae",{"inputs":"historical-v1"},producer(calls,"unexpected"))
    second=store.run("adapter",{"fit":again["artifact_sha256"],"adapter":"v2"},producer(calls,"adapt2"))
    assert again["reused"] and calls==["cvae","adapt1","adapt2"]
    assert first["stage_root"]!=second["stage_root"]
    assert Path(first["stage_root"],"output.txt").read_text()=="valid"


@pytest.mark.parametrize("failed_stage",["base","prediction","report"])
def test_downstream_failure_never_relaunches_successful_upstream(tmp_path,failed_stage):
    store=api().StageStore(tmp_path/"run"); calls=[]; failed=[False]
    def pipeline():
        parent=None
        for name in ("cvae","base","prediction","report"):
            def work(root,name=name):
                if name==failed_stage and not failed[0]:
                    failed[0]=True; raise RuntimeError("injected downstream failure")
                return producer(calls,name)(root)
            parent=store.run(name,{"parent":parent["artifact_sha256"] if parent else "input"},work)
        return parent
    with pytest.raises(RuntimeError,match="injected"): pipeline()
    final=pipeline()
    assert calls.count("cvae")==1 and calls.count("base")==1 and calls.count("prediction")==1
    assert final["metadata"]["key"]=="report"


def test_output_commit_recovery_and_corrupt_artifact_preservation(tmp_path):
    store=api().StageStore(tmp_path/"run"); calls=[]
    with pytest.raises(RuntimeError,match="after_outputs"):
        store.run("predict",{"id":1},producer(calls,"predict"),fail_after_outputs=True)
    recovered=store.run("predict",{"id":1},producer(calls,"unexpected"))
    assert calls==["predict"] and recovered["recovered"]
    old=Path(recovered["stage_root"])/"output.txt"; old.write_text("corrupted")
    repaired=store.run("predict",{"id":1},producer(calls,"repair"))
    assert repaired["stage_root"]!=recovered["stage_root"] and old.read_text()=="corrupted"
    assert Path(repaired["stage_root"],"output.txt").read_text()=="valid"


def test_concurrent_arms_share_one_immutable_stage(tmp_path):
    store=api().StageStore(tmp_path/"shared"); calls=[]
    def arm(name):
        shared=store.run("cvae",{"same":"fit"},producer(calls,"cvae"))
        owned=api().StageStore(tmp_path/name).run("ranker",{"arm":name,"fit":shared["artifact_sha256"]},producer(calls,name))
        return shared,owned
    with ThreadPoolExecutor(max_workers=2) as pool: a,b=list(pool.map(arm,["A","B"]))
    assert calls.count("cvae")==1 and a[0]["stage_root"]==b[0]["stage_root"]
    assert a[1]["stage_root"]!=b[1]["stage_root"]


def test_namespace_escape_and_stopped_run_writes_are_rejected(tmp_path):
    old=tmp_path/"old"; old.mkdir(); marker=old/"COMPLETED.json"; marker.write_text("preserve")
    with pytest.raises(ValueError,match="protected"):
        api().StageStore(old/"subdir",protected_roots=[old])
    store=api().StageStore(tmp_path/"new",protected_roots=[old])
    with pytest.raises(ValueError,match="stage name"):
        store.run("../old",{},producer([],"bad"))
    assert marker.read_text()=="preserve"
