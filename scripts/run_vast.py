"""Run final B with explicit external runtime inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"src"),str(ROOT)]

def main():
    parser=argparse.ArgumentParser(description=__doc__); actions=parser.add_subparsers(dest="action",required=True)
    launch=actions.add_parser("run"); launch.add_argument("--runtime",type=Path,required=True)
    launch.add_argument("--mode",choices=["smoke","formal"],required=True); launch.add_argument("--resume",action="store_true"); launch.add_argument("--prepare-only",action="store_true")
    for action in ("status","report","unit"):
        item=actions.add_parser(action); item.add_argument("--output",type=Path,required=True)
        if action=="unit": item.add_argument("--unit-id",required=True)
    args=parser.parse_args()
    if args.action=="run":
        from vast.runtime import run
        result=run(args.runtime,args.mode,args.resume,args.prepare_only)
    elif args.action=="unit":
        from vast.runtime import configure_environment
        from rcl_study.vae_snapshot_contract import read_json
        from rcl_study.vae_snapshot_pipeline import run_unit
        configure_environment(args.output); manifest=read_json(args.output/"manifest.json")
        unit=next(u for u in manifest["units"] if u["unit_id"]==args.unit_id); result=run_unit(ROOT,manifest,unit)
    elif args.action=="report":
        from vast.runtime import collect
        full=collect(args.output)
        result={"status":full["status"],"mode":full["mode"],"cells":[{k:c[k] for k in ("dataset_id","regime","method","metrics","pre_OSER_metrics")} for c in full["cells"]]}
    else:
        from rcl_study.vae_snapshot_contract import read_json,file_sha256
        manifest=read_json(args.output/"manifest.json"); rows=[]
        for unit in manifest["units"]:
            folder=Path(unit["unit_output_root"]); state=folder/"status.json"
            row={"unit_id":unit["unit_id"],**(read_json(state) if state.is_file() else {"status":"queued"})}
            if (folder/"unit.done.json").is_file():
                receipt=read_json(folder/"unit.done.json")
                if file_sha256(receipt["result_path"])!=receipt["result_sha256"]: raise ValueError("completed result changed")
            rows.append(row)
        heartbeat=args.output/"heartbeat.json"; result={"mode":manifest["mode"],"units":rows,"heartbeat":read_json(heartbeat) if heartbeat.is_file() else {}}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
