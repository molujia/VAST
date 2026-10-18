import importlib.util
import hashlib
import pytest


def api():
    assert importlib.util.find_spec('vast_v3.encoders.text') is not None, 'offline text-asset lock missing'
    from vast_v3.encoders.text import verify_asset_lock
    return verify_asset_lock


def test_offline_asset_lock_checks_every_file_and_revision(tmp_path):
    verify=api(); (tmp_path/'config.json').write_text('{}')
    manifest={'repository':'sentence-transformers/all-MiniLM-L6-v2','revision':'a'*40,
              'files':{'config.json':hashlib.sha256(b'{}').hexdigest()}}
    assert verify(tmp_path,manifest)['revision']=='a'*40
    (tmp_path/'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash'):
        verify(tmp_path,manifest)


def test_asset_lock_rejects_directory_escape_and_missing_revision(tmp_path):
    verify=api()
    with pytest.raises(ValueError,match='revision'):
        verify(tmp_path,{'repository':'sentence-transformers/all-MiniLM-L6-v2','revision':None,'files':{}})
    with pytest.raises(ValueError,match='path'):
        verify(tmp_path,{'repository':'sentence-transformers/all-MiniLM-L6-v2','revision':'a'*40,'files':{'../secret':'b'*64}})


def test_loader_can_only_see_pinned_files_and_forces_safetensors(tmp_path, monkeypatch):
    import sys
    import types
    from pathlib import Path
    from vast_v3.encoders.text import FrozenSentenceEncoder
    required = ('config.json', 'modules.json', 'tokenizer.json', '1_Pooling/config.json', 'model.safetensors')
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{}')
    (tmp_path / 'special_tokens_map.json').write_text('{"extra":true}')
    (tmp_path / 'pytorch_model.bin').write_bytes(b'unlocked-alternative')
    lock = {'repository':'sentence-transformers/all-MiniLM-L6-v2', 'revision':'a' * 40,
            'files':{name:hashlib.sha256(b'{}').hexdigest() for name in required}}
    seen = {}

    class FakeDependency:
        def __init__(self, path, **kwargs):
            seen['files'] = {p.relative_to(path).as_posix() for p in Path(path).rglob('*') if p.is_file()}
            seen['kwargs'] = kwargs
        def requires_grad_(self, value):
            assert value is False
        def eval(self):
            pass
        def get_sentence_embedding_dimension(self):
            return 384

    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=FakeDependency))
    FrozenSentenceEncoder(tmp_path, lock, staging_parent=tmp_path)
    assert seen['files'] == set(required)
    assert seen['kwargs']['model_kwargs']['use_safetensors'] is True
    assert seen['kwargs']['local_files_only'] is True
    assert (tmp_path / 'special_tokens_map.json').exists()  # Original cache untouched.
