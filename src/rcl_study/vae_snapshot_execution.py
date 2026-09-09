"""Content-addressed durable stages; preserve corrupt/stopped/unrelated outputs."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime,timezone
import os
from pathlib import Path
import re
import sys
import time

from rcl_study.vae_snapshot_contract import SnapshotContractError,file_sha256,read_json,semantic_sha256
from rcl_study.vae_snapshot_cvae_stage import _write


def now(): return datetime.now(timezone.utc).isoformat()


@contextmanager
def file_lock(path):
    """Kernel-owned Linux lock is released on process exit, including failure."""
    import fcntl
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


class StageStore:
    def __init__(self,root,*,protected_roots=(),progress=None):
        self.root=Path(root).resolve(); self.progress=progress
        for path in protected_roots:
            protected=Path(path).resolve()
            if self.root==protected or protected in self.root.parents or self.root in protected.parents:
                raise SnapshotContractError("stage store intersects protected outputs")
        self.root.mkdir(parents=True,exist_ok=True)

    def _event(self,name,status,**extra):
        event={"stage":name,"status":status,"pid":os.getpid(),"updated_at":now(),**extra}
        # Latest status is mutable, stage commits remain immutable.
        _write(self.root/"status.json",event)
        if self.progress: self.progress(event)

    def _valid(self,root,receipt,identity_sha,validator):
        try:
            sealed={k:v for k,v in receipt.items() if k!="artifact_sha256"}
            if (receipt["identity_sha256"]!=identity_sha or semantic_sha256(sealed)!=receipt["artifact_sha256"]
                    or not receipt["output_sha256s"]): return False
            for relative,digest in receipt["output_sha256s"].items():
                path=(root/relative).resolve()
                if root.resolve() not in path.parents or not path.is_file() or file_sha256(path)!=digest: return False
            if validator: validator(root,receipt["metadata"])
            return True
        except (OSError,ValueError,KeyError,TypeError): return False

    def run(self,name,identity,producer,*,validator=None,fail_after_outputs=False):
        if not re.fullmatch(r"[A-Za-z0-9_-]+",name): raise SnapshotContractError("invalid stage name")
        digest=semantic_sha256(identity); key_root=self.root/"stages"/name/digest
        if self.root not in key_root.resolve().parents: raise SnapshotContractError("stage namespace escaped through link")
        with file_lock(key_root/"stage.lock"):
            attempts=sorted(key_root.glob("attempt-*")); incomplete=None
            for candidate in reversed(attempts):
                commit=candidate/"stage.done.json"; ready=candidate/"outputs.ready.json"
                receipt_path=commit if commit.is_file() else ready
                if receipt_path.is_file():
                    try: receipt=read_json(receipt_path)
                    except ValueError: continue
                    if self._valid(candidate,receipt,digest,validator):
                        recovered=not commit.is_file()
                        if recovered: _write(commit,receipt)
                        self._event(name,"reused",artifact_sha256=receipt["artifact_sha256"],recovered=recovered)
                        return {**receipt,"stage_root":str(candidate),"reused":True,"recovered":recovered}
                    # Do not repair committed/certified bytes in place.
                elif incomplete is None:
                    incomplete=candidate
            work=incomplete if incomplete is not None else key_root/("attempt-%04d" % (len(attempts)+1))
            work.mkdir(exist_ok=True); started=time.monotonic()
            self._event(name,"running",stage_root=str(work),identity_sha256=digest)
            try:
                result=producer(work); hashes={}
                for relative in result["outputs"]:
                    path=(work/relative).resolve()
                    if work.resolve() not in path.parents or not path.is_file():
                        raise SnapshotContractError("stage producer output escapes or is absent")
                    hashes[relative]=file_sha256(path)
                receipt={"schema_version":"snapshot-stage-commit-v1","name":name,"identity_sha256":digest,
                    "identity":identity,"output_sha256s":hashes,"metadata":result.get("metadata",{})}
                receipt["artifact_sha256"]=semantic_sha256(receipt)
                if not self._valid(work,receipt,digest,validator): raise SnapshotContractError("stage output validation failed")
                _write(work/"outputs.ready.json",receipt)
                if fail_after_outputs: raise RuntimeError("injected after_outputs failure")
                _write(work/"stage.done.json",receipt)
                self._event(name,"completed",artifact_sha256=receipt["artifact_sha256"],elapsed_seconds=time.monotonic()-started)
                return {**receipt,"stage_root":str(work),"reused":False,"recovered":False}
            except BaseException as exc:
                error={"stage":name,"error_type":type(exc).__name__,"message":str(exc),"updated_at":now(),"pid":os.getpid()}
                _write(work/"last-error.json",error); self._event(name,"failed",**{k:v for k,v in error.items() if k!="stage"})
                raise


def effective_sources(root,names):
    return {name:file_sha256(Path(root)/name if (Path(root)/name).is_file() else Path(root)/"src"/name) for name in names}


def controller_runtime():
    import numpy,sklearn
    from threadpoolctl import threadpool_info
    return {"python":sys.version,"numpy":numpy.__version__,"sklearn":sklearn.__version__,
        "thread_environment":{key:os.environ.get(key) for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
        "threadpools":threadpool_info()}
