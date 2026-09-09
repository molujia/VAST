"""Optional two-interpreter integration test using synthetic feature rows only."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.environ.get("VAST_WORKER_PYTHON"),reason="set VAST_WORKER_PYTHON for synthetic CLI integration")
def test_smoke_resume_and_changed_input_refusal(tmp_path):
    import pandas as pd
    from vast.method import REPOSITORY_ROOT
    from rcl_study.vae_snapshot_contract import read_json,file_sha256
    # Set pytest --basetemp outside the checkout, just like real runtime outputs.
    feature=tmp_path/"fixture"/"rcabench"; feature.mkdir(parents=True)
    columns=["metric_abs_z_max","log_error_count","trace_error_ratio","has_metric_signal","has_log_signal","has_trace_signal"]
    windows=[]; entities=[]
    for i in range(10):
        case="fixture-%02d"%i
        meta={"fault_type":"type-%s"%(i%2)}
        base={"dataset":"rcabench","source_id":case,"day":"fixture","start_ts":i*100,
              "end_ts":i*100+60,"window_id":case,"window_kind":"fault"}
        windows.append({**base,"positive_ids":"svc-0","metadata_json":json.dumps(meta),"duration_seconds":60})
        for j in range(3):
            entities.append({**base,"entity_id":"svc-%s"%j,"entity_index":j,"entity_type":"service",
                "metric_abs_z_max":float(3-j+i*.1),"log_error_count":float(j+i%2),"trace_error_ratio":j*.1,
                "has_metric_signal":1,"has_log_signal":1,"has_trace_signal":1})
    pd.DataFrame(windows).to_csv(feature/"windows.csv",index=False)
    pd.DataFrame(entities).to_csv(feature/"entity_features.csv",index=False)
    (feature/"metadata.json").write_text(json.dumps({"all_feature_columns":columns}))
    split={"outer_train_case_ids":["fixture-%02d"%i for i in range(7)],"outer_test_case_ids":["fixture-%02d"%i for i in range(7,10)]}
    (tmp_path/"split.json").write_text(json.dumps(split))
    spec={"schema_version":"vast-runtime-v1","output_root":"run","worker_python":os.environ["VAST_WORKER_PYTHON"],
        "device":"cpu","regimes":["oracle_full"],"datasets":[{"dataset_id":"rcabench","feature_dir":"fixture/rcabench","split_manifest":"split.json"}]}
    runtime=tmp_path/"runtime.json"; runtime.write_text(json.dumps(spec))
    cli=[sys.executable,str(REPOSITORY_ROOT/"scripts/run_vast.py")]
    def command(*args,success=True):
        result=subprocess.run(cli+list(args),cwd=REPOSITORY_ROOT,text=True,capture_output=True,timeout=180)
        if success: assert result.returncode==0,result.stdout+result.stderr
        return result
    command("run","--runtime",str(runtime),"--mode","smoke")
    output=tmp_path/"run"
    done=read_json(output/"run.done.json")
    assert done["mode"]=="smoke" and len(done["cells"])==1
    cell=done["cells"][0]
    assert cell["method"]=="B" and cell["metrics"]["denominator"]==3
    assert cell["evidence_role"]=="bounded_debug_only"
    checkpoints={p:(file_sha256(p),p.stat().st_mtime_ns) for p in output.rglob("checkpoint.pt")}
    assert len(checkpoints)==2
    command("run","--runtime",str(runtime),"--mode","smoke","--resume")
    assert checkpoints=={p:(file_sha256(p),p.stat().st_mtime_ns) for p in checkpoints}
    report=command("report","--output",str(output))
    assert json.loads(report.stdout)["cells"][0]["metrics"]==cell["metrics"]
    status=command("status","--output",str(output))
    assert json.loads(status.stdout)["units"][0]["status"]=="completed"
    with (feature/"windows.csv").open("a") as stream: stream.write("\n")
    result=command("run","--runtime",str(runtime),"--mode","smoke","--resume",success=False)
    assert result.returncode!=0 and "identity differs" in result.stderr
