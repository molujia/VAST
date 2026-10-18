"""Explicitly pinned offline Sentence Transformer; never fetch missing assets."""
from pathlib import Path
import hashlib
import copy
import re
import shutil
import tempfile
import sys
from collections import OrderedDict
import torch


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def verify_asset_lock(directory,manifest):
    if not isinstance(manifest.get('revision'),str) or not re.fullmatch(r'[0-9a-f]{40}',manifest['revision']):
        raise ValueError('exact repository revision is required')
    if manifest.get('repository')!='sentence-transformers/all-MiniLM-L6-v2':
        raise ValueError('unexpected fixed text checkpoint')
    root=Path(directory).absolute()
    for name,expected in manifest.get('files',{}).items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('asset path must stay inside snapshot')
        path=root/relative
        # HuggingFace snapshots legitimately use blob symlinks; the manifest
        # hashes their resolved bytes, not untrusted path names from a model.
        if not path.is_file() or file_sha256(path)!=expected:
            raise ValueError(f'asset hash mismatch: {name}')
    return manifest


class FrozenSentenceEncoder:
    """Pinned, frozen text model with a bounded, instance-local exact-call cache.

    Only complete ordered inputs are reused: neither deduplication nor re-batching
    is allowed because those can change padding and floating-point results. The
    CPU cache is an engineering resource, not trainable/checkpoint state.
    """
    def __init__(self,directory,manifest,*,device='cpu',batch_size=64,staging_parent=None,
                 cache_max_bytes=512*1024*1024):
        if type(cache_max_bytes) is not int or cache_max_bytes < 0:
            raise ValueError('text cache maximum bytes must be a nonnegative integer')
        self.manifest=copy.deepcopy(verify_asset_lock(directory,manifest))
        required={'config.json','modules.json','tokenizer.json','1_Pooling/config.json','model.safetensors'}
        if not required<=set(manifest['files']):
            raise ValueError('incomplete Sentence Transformer asset manifest')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('text batch size must be positive')
        from sentence_transformers import SentenceTransformer
        # A shared HF cache can contain alternative weights, adapters and tokenizer
        # overrides. Present only the pinned files, without altering that cache.
        # Copies (not symlinks) also isolate the loader from later cache changes.
        with tempfile.TemporaryDirectory(prefix='vast-v3-pinned-text-',dir=staging_parent) as staging:
            for name in self.manifest['files']:
                target=Path(staging)/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(Path(directory)/name,target)
            verify_asset_lock(staging,self.manifest)
            self.model=SentenceTransformer(staging,device=device,local_files_only=True,trust_remote_code=False,
                                           model_kwargs={'use_safetensors':True})
        self.model.requires_grad_(False); self.model.eval()
        if self.model.get_sentence_embedding_dimension()!=384:
            raise ValueError('fixed checkpoint must emit 384 coordinates')
        self.batch_size=batch_size
        self._cache_max_bytes=cache_max_bytes
        self._cache=OrderedDict()
        self._cache_payload_bytes=0
        self._cache_context=None
        self._cache_hits=0
        self._cache_misses=0

    def clear_cache(self):
        """Release retained texts/vectors; operational counters remain cumulative."""
        self._cache.clear()
        self._cache_payload_bytes=0
        self._cache_context=None

    def cache_info(self):
        """Conservative retained bytes, including Python keys and index overhead."""
        return {'entries':len(self._cache),
                'bytes':self._cache_payload_bytes+(sys.getsizeof(self._cache) if self._cache else 0),
                'max_bytes':self._cache_max_bytes,
                'hits':self._cache_hits,'misses':self._cache_misses}

    def _runtime_context(self):
        # Standard load_state_dict/in-place updates and device/dtype moves must
        # invalidate old values. Raw .data mutation or tokenizer/config edits are
        # outside this immutable, pinned encoder contract: construct a new encoder.
        named=(*self.model.named_parameters(),*self.model.named_buffers())
        if any(torch.is_inference(value) for _,value in named):
            return None  # No reliable version counters: preserve uncached support.
        tensors=tuple((name,id(value),value._version,str(value.device),value.dtype,
                       tuple(value.shape),value.data_ptr())
                      for name,value in named)
        return (id(self.model),tensors,self.batch_size,
                getattr(self.model,'max_seq_length',None),getattr(self.model,'truncate_dim',None),
                torch.is_autocast_enabled('cpu'),torch.get_autocast_dtype('cpu'),
                torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'),
                torch.get_float32_matmul_precision(),torch.get_num_threads(),
                torch.are_deterministic_algorithms_enabled(),torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,torch.backends.cudnn.allow_tf32)

    def _remember(self,key,output):
        # Counting repeated string references repeatedly is conservative, and
        # importantly prevents huge textual inputs from evading a tensor-only cap.
        device=output.device
        payload=(sys.getsizeof(key)+sum(sys.getsizeof(text) for text in key)
                 +sys.getsizeof(output)+output.numel()*output.element_size()
                 +sys.getsizeof(device)+sys.getsizeof((None,None,0))+sys.getsizeof(0))
        if payload+sys.getsizeof(OrderedDict()) > self._cache_max_bytes:
            return
        while self._cache and self.cache_info()['bytes']+payload > self._cache_max_bytes:
            _,(_,_,charge)=self._cache.popitem(last=False)
            self._cache_payload_bytes-=charge
        # Never retain a training graph, GPU tensor or caller-visible storage.
        cached=output.detach().to(device='cpu',copy=True).contiguous()
        self._cache[key]=(cached,device,payload)
        self._cache_payload_bytes+=payload
        # OrderedDict can grow its index during insertion, so check the actual
        # container size too. An oversized single entry simply remains uncached.
        while self._cache and self.cache_info()['bytes'] > self._cache_max_bytes:
            _,(_,_,charge)=self._cache.popitem(last=False)
            self._cache_payload_bytes-=charge
        if not self._cache:
            self._cache.clear()

    def __call__(self,texts):
        key=None
        if self._cache_max_bytes and type(texts) in (list,tuple) and all(type(t) is str for t in texts):
            context=self._runtime_context()
            if context is None:
                self.clear_cache()
            else:
                if context!=self._cache_context:
                    self.clear_cache()
                    self._cache_context=context
                key=tuple(texts)
                cached=self._cache.get(key)
                if cached is not None:
                    self.model.eval()  # SentenceTransformer.encode does this on misses.
                    self._cache.move_to_end(key)
                    self._cache_hits+=1
                    return cached[0].to(device=cached[1],copy=True)
        self._cache_misses+=1
        with torch.no_grad():
            output=self.model.encode(texts,batch_size=self.batch_size,convert_to_tensor=True,
                                     show_progress_bar=False,normalize_embeddings=False)
        if key is not None:
            self._remember(key,output)
        return output
