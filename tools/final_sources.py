"""Deterministic selection of the approved B implementation and historical code."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

SNAPSHOT = "openspec/changes/deepen-query-only-active-learning/evidence/current-label-free-reference-package/current_label_free_reference_20260821_final/code"
ROOTS = tuple("rcl_study/"+name+".py" for name in (
    "vae_snapshot_historical", "vae_snapshot_latent", "vae_snapshot_cvae_stage",
    "vae_snapshot_oser", "vae_snapshot_oser_inference", "vae_snapshot_encoding",
    "vae_snapshot_execution", "vae_snapshot_pipeline", "vae_snapshot_worker"))


def select_definitions(text, names):
    lines=text.splitlines(True); tree=ast.parse(text)
    pieces=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)) or getattr(node,"name",None) in names:
            pieces.append("".join(lines[node.lineno-1:node.end_lineno]).rstrip())
    return "\n\n".join(pieces)+"\n"


def transform(relative,text):
    """Remove obsolete orchestration, preserving numerical functions verbatim."""
    if relative=="rcl_study/vae_snapshot_contract.py":
        text=select_definitions(text,{"SnapshotContractError","semantic_sha256","file_sha256","read_json"})
        start=text.index("from rcl_study.historical_pairwise_comparison import (")
        end=text.index(")",start)+1
        text=text[:start]+text[end:]
        text+='\nOUTPUT_NAMESPACE = "outputs/vast"\n'
    elif relative=="scripts/run_historical_pairwise_hdbscan_compare.py":
        text=select_definitions(text,{"_derive_metrics_without_package","_read_jsonl","_semantic_sha256","_json_default"})
    elif relative=="rcl_study/vae_snapshot_pipeline.py":
        names={"save_pickle","load_pickle","_print","_run_worker_unlocked","run_worker","_posterior",
            "_validate_rankings","run_unit","_base_predictions","_oser_training_request","_finish_unit"}
        text=select_definitions(text,names)
        text=text.replace("OUTPUT_NAMESPACE,build_snapshot_manifest,","OUTPUT_NAMESPACE,")
        text=text.replace("file_sha256,load_snapshot_config,read_json,semantic_sha256,snapshot_run_root", "file_sha256,read_json,semantic_sha256")
        start=text.index('        if unit["method"]=="A":\n')
        end=text.index('        def base_fit(root):',start)
        block=text[start:end]; b=block[block.index('        else:\n')+len('        else:\n'):]
        text=text[:start]+"".join(line[4:] if line.startswith("    ") else line for line in b.splitlines(True))+text[end:]
        start=text.index('            if unit["method"]=="A":\n')
        end=text.index('            save_pickle(root/"model.pkl",model)',start)
        text=text[:start]+'            from rcl_study.vae_snapshot_latent import fit_B\n            model,audit=fit_B(api,bundle,mass=cfg["augmentation_mass_per_parent"])\n'+text[end:]
        text=text.replace('    repo=Path(repo); cfg=', '    if unit["method"] != "B": raise SnapshotContractError("this release implements selected method B only")\n    repo=Path(repo); cfg=')
        text=text.replace('repo/"rcl_study/', 'repo/"src/rcl_study/')
        text=text.replace('if kind in ("cvae","generate","encode"):', 'if kind in ("cvae","encode"):')
        text+='\ndef _cache(repo):\n    return Path(os.environ["VAST_CACHE_ROOT"]).resolve()\n\n'
        text+='def input_stage(repo,cfg,api,manifest,unit):\n    return manifest["inputs"][unit["unit_id"]]\n'
    elif relative=="rcl_study/vae_snapshot_worker.py":
        text=text.replace('"runtime","cvae","generate","encode","oser","oser_predict"','"runtime","cvae","encode","oser","oser_predict"')
        start=text.index('    elif args.kind=="generate":')
        end=text.index('    elif args.kind=="encode":',start)
        text=text[:start]+text[end:]
    elif relative=="rcl_study/vae_snapshot_execution.py":
        text=text.replace('return {name:file_sha256(Path(root)/name) for name in names}',
            'return {name:file_sha256(Path(root)/name if (Path(root)/name).is_file() else Path(root)/"src"/name) for name in names}')
    elif relative=="tests/test_vae_snapshot_historical.py":
        text=text.replace('from rcl_study.vae_snapshot_contract import load_snapshot_config',
            'from vast.method import load_default_config, with_historical_authority')
        text=text.replace('load_snapshot_config(ROOT / "configs/final_rcl/vae_snapshot_rebase_seed42.json", repo_root=ROOT)',
            'with_historical_authority(load_default_config())')
    return text


def imports(text,current):
    package=current.removesuffix(".py").replace("/",".").split(".")[:-1]
    values=set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,ast.Import): values.update(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom):
            prefix=package[:len(package)-node.level+1] if node.level else []
            if node.module: prefix+=node.module.split(".")
            module=".".join(prefix); values.add(module)
            values.update(module+"."+a.name for a in node.names if a.name!="*")
    return sorted(values)


def materialize(source,destination,copy_file,write_inventory,sanitize):
    source=Path(source).resolve(); destination=Path(destination).resolve()
    if destination.name!="VAST" or source not in destination.parents:
        raise ValueError("build must target the authority workspace's VAST checkout")
    previous=destination/"docs/provenance/source-inventory.json"
    old=json.loads(previous.read_text(encoding="utf-8"))["files"] if previous.exists() else []
    rows=[]; selected=set(); queue=list(ROOTS)+["rcl_study/__init__.py","scripts/__init__.py","fixed_active_learning/__init__.py","fixed_active_learning/pipeline.py",
        "tests/test_vae_snapshot_historical.py","tests/test_vae_snapshot_latent.py","tests/test_vae_snapshot_execution.py"]
    fixed=source/"artifacts/rcl-active-learning-fixed-v1-20260906/src"
    while queue:
        relative=queue.pop(0)
        if relative in selected: continue
        selected.add(relative)
        origin=(fixed if relative.startswith("fixed_active_learning/") else source)/relative
        text=origin.read_text(encoding="utf-8"); original=origin.read_bytes()
        changed=transform(relative,text); changed,_=sanitize(changed)
        target=relative if relative.startswith(("scripts/","tests/")) else "src/"+relative
        path=destination/target; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(changed.encode("utf-8"))
        rows.append({"source_path":str(origin.relative_to(source)).replace("\\","/"),"destination_path":target,
            "source_sha256":hashlib.sha256(original).hexdigest(),"destination_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "role":"selected_B_dependency","transforms":{"documented_final_B_portability":int(changed!=text)}})
        for module in imports(changed,relative):
            if module.split(".")[0] not in {"rcl_study","scripts","fixed_active_learning"}: continue
            base=fixed if module.startswith("fixed_active_learning.") else source
            for candidate in (module.replace(".","/")+".py",module.replace(".","/")+"/__init__.py"):
                if (base/candidate).is_file(): queue.append(candidate); break
    # Dynamic historical imports run under a private package plus the original
    # nexusrcl_rebuild name. Keep that isolated source tree and its import closure.
    historical=source/SNAPSHOT
    search=[historical/"half_supervise/src",historical,historical/"self_supervise",historical/"metric_AD"]
    pending=[historical/"rcl_study/__init__.py",historical/"rcl_study/semi_supervised_bounded.py",
        search[0]/"nexusrcl_rebuild/training/semisupervised.py",search[0]/"nexusrcl_rebuild/training/pairwise_backend.py",
        search[0]/"nexusrcl_rebuild/evaluation/splits.py"]
    visited=set()
    while pending:
        origin=pending.pop(0).resolve()
        if origin in visited: continue
        visited.add(origin); relative=origin.relative_to(historical).as_posix()
        text=origin.read_text(encoding="utf-8"); changed,_=sanitize(text)
        # Preserve exact bytes where no portability replacement is necessary.
        content=origin.read_bytes() if changed==text else changed.encode("utf-8")
        target="src/vast/_historical/"+relative
        path=destination/target; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(content)
        rows.append({"source_path":SNAPSHOT+"/"+relative,"destination_path":target,"source_sha256":hashlib.sha256(origin.read_bytes()).hexdigest(),
            "destination_sha256":hashlib.sha256(content).hexdigest(),"role":"historical_snapshot","transforms":{"machine_paths":int(changed!=text)}})
        import_root=next(base for base in search if base in origin.parents)
        for module in imports(changed,origin.relative_to(import_root).as_posix()):
            for base in search:
                found=next((base/p for p in (module.replace(".","/")+".py",module.replace(".","/")+"/__init__.py") if (base/p).is_file()),None)
                if found:
                    pending.append(found)
                    for parent in found.parents:
                        if parent==base: break
                        if (parent/"__init__.py").is_file(): pending.append(parent/"__init__.py")
                    break
    desired={r["destination_path"] for r in rows}
    for row in old:
        relative=row["destination_path"]; target=(destination/relative).resolve()
        if destination not in target.parents: raise ValueError("old inventory escapes checkout")
        if relative not in desired and target.is_file(): target.unlink()
    return write_inventory(destination,rows=rows,source_authorities={"selected_method":"historical-snapshot-latent-B-oser-p02",
        "experiment":"vae-snapshot-rebase-seed42-20260908-01","active_learning":"rcl-active-learning-fixed-v1-20260906"})
