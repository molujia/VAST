"""Exact-call cache tests; the tiny encoder substitutes only offline model loading."""
import hashlib
import random
import sys
import types

import pytest
import torch

from vast_v3.encoders.text import FrozenSentenceEncoder


class TinySentenceModel(torch.nn.Module):
    """Deterministic encode whose values depend on call-wide order and padding."""
    def __init__(self, path, *, device, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(384, device=device))
        self.register_buffer('bias', torch.zeros(384, device=device))
        self.calls = []
        self.max_seq_length = 256

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        self.eval()
        self.calls.append((list(texts), dict(kwargs)))
        width = max([len(t) for t in texts] or [0])
        rows = [self.weight * (sum(map(ord, text)) + j + width + kwargs['batch_size']) + self.bias
                for j, text in enumerate(texts)]
        return torch.stack(rows) if rows else self.weight.new_empty((0, 384))


@pytest.fixture
def make_encoder(tmp_path, monkeypatch):
    required = ('config.json', 'modules.json', 'tokenizer.json',
                '1_Pooling/config.json', 'model.safetensors')
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{}')
    manifest = {'repository': 'sentence-transformers/all-MiniLM-L6-v2',
                'revision': 'a' * 40,
                'files': {name: hashlib.sha256(b'{}').hexdigest() for name in required}}
    monkeypatch.setitem(sys.modules, 'sentence_transformers',
                        types.SimpleNamespace(SentenceTransformer=TinySentenceModel))
    return lambda **kwargs: FrozenSentenceEncoder(tmp_path, manifest, staging_parent=tmp_path, **kwargs)


def test_identical_ordered_call_is_reused_without_changing_encode_arguments(make_encoder):
    encoder = make_encoder()
    texts = ['repeat', 'x', 'repeat', 'longer text']
    first = encoder(texts)
    second = encoder(list(texts))
    assert torch.equal(first, second)
    assert len(encoder.model.calls) == 1
    assert encoder.model.calls[0] == (texts, {'batch_size': 64, 'convert_to_tensor': True,
        'show_progress_bar': False, 'normalize_embeddings': False})
    assert encoder.cache_info()['hits'] == 1


def test_input_order_duplicates_and_full_call_boundaries_are_not_rewritten(make_encoder):
    encoder = make_encoder()
    original = ['a', 'bbb', 'a']
    first = encoder(original)
    reordered = encoder(['bbb', 'a', 'a'])
    separate = torch.cat((encoder(['a']), encoder(['bbb', 'a'])))
    assert not torch.equal(first, reordered)
    assert not torch.equal(first, separate)
    assert len(encoder.model.calls) == 4
    encoder(original)
    assert len(encoder.model.calls) == 4


def test_returned_mutation_and_input_list_mutation_cannot_corrupt_cache(make_encoder):
    encoder = make_encoder()
    texts = ['alpha', 'beta']
    expected = encoder(texts).clone()
    first = encoder(texts)
    first.fill_(-123)
    texts[0] = 'new'
    encoder(texts)
    assert torch.equal(encoder(['alpha', 'beta']), expected)
    encoder(['alpha', 'beta']).resize_(1)
    assert torch.equal(encoder(['alpha', 'beta']), expected)


def test_instances_never_share_embeddings_or_cache(make_encoder):
    first, second = make_encoder(), make_encoder()
    a = first(['same'])
    with torch.no_grad():
        second.model.weight.mul_(2)
    b = second(['same'])
    assert torch.equal(b, 2 * a)
    assert len(first.model.calls) == len(second.model.calls) == 1


@pytest.mark.parametrize('change', ['weight', 'buffer', 'load', 'dtype', 'batch', 'model'])
def test_runtime_model_changes_cannot_reuse_stale_entries(make_encoder, change):
    encoder = make_encoder()
    encoder(['same'])
    with torch.no_grad():
        if change == 'weight':
            encoder.model.weight.add_(1)
        elif change == 'buffer':
            encoder.model.bias.add_(1)
        elif change == 'load':
            state = encoder.model.state_dict()
            state['weight'] = torch.full_like(state['weight'], 3)
            encoder.model.load_state_dict(state)
        elif change == 'dtype':
            encoder.model.double()
        elif change == 'batch':
            encoder.batch_size = 32
        else:
            old = encoder.model
            encoder.model = TinySentenceModel(None, device='cpu').requires_grad_(False)
            encoder.model.calls = old.calls
    result = encoder(['same'])
    assert len(encoder.model.calls) == 2
    expected = encoder.model.encode(['same'], batch_size=encoder.batch_size,
                                    convert_to_tensor=True, show_progress_bar=False,
                                    normalize_embeddings=False)
    assert torch.equal(result, expected)


def test_disabled_cache_preserves_call_behavior(make_encoder):
    encoder = make_encoder(cache_max_bytes=0)
    expected = encoder(['same'])
    assert torch.equal(encoder(['same']), expected)
    assert len(encoder.model.calls) == 2
    assert encoder.cache_info()['entries'] == 0


@pytest.mark.parametrize('quota', [-1, 1.5, True, None])
def test_cache_quota_requires_nonnegative_integer(make_encoder, quota):
    with pytest.raises(ValueError, match='cache.*bytes'):
        make_encoder(cache_max_bytes=quota)


def test_quota_counts_retained_text_not_just_tensor(make_encoder):
    encoder = make_encoder(cache_max_bytes=8192)
    texts = ['x' * 20000]  # One 1536-byte vector must not hide the large key.
    expected = encoder(texts)
    assert torch.equal(encoder(texts), expected)
    assert encoder.cache_info()['entries'] == 0
    assert encoder.cache_info()['bytes'] <= 8192
    assert len(encoder.model.calls) == 2


def test_lru_eviction_and_quota_hold_at_every_insertion(make_encoder):
    encoder = make_encoder(cache_max_bytes=5000)
    encoder(['a'])
    encoder(['b'])
    assert encoder.cache_info()['entries'] == 2
    encoder(['a'])  # b becomes the least recently used.
    encoder(['c'])
    assert encoder.cache_info()['entries'] == 2
    assert encoder.cache_info()['bytes'] <= 5000
    encoder(['a'])
    assert len(encoder.model.calls) == 3
    encoder(['b'])
    assert len(encoder.model.calls) == 4
    for n in range(40):
        encoder([str(n)])
        assert encoder.cache_info()['bytes'] <= 5000


def test_cache_stores_cpu_detached_outputs_and_clear_releases_payload(make_encoder):
    encoder = make_encoder()
    encoder(['x'])
    assert encoder.cache_info()['bytes'] > 0
    assert encoder.cache_info()['entries'] == 1
    assert encoder(['x']).grad_fn is None
    encoder.clear_cache()
    assert encoder.cache_info()['entries'] == encoder.cache_info()['bytes'] == 0
    encoder(['x'])
    assert len(encoder.model.calls) == 2


def test_cache_does_not_consume_randomness_or_mutate_global_math_settings(make_encoder):
    encoder = make_encoder()
    torch_before = torch.get_rng_state().clone()
    random_before = random.getstate()
    deterministic = torch.are_deterministic_algorithms_enabled()
    encoder(['x'])
    encoder(['x'])
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == random_before
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_autocast_context_does_not_reuse_other_precision_result(make_encoder):
    encoder = make_encoder()
    encoder(['x'])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        encoder(['x'])
        encoder(['x'])
    assert len(encoder.model.calls) == 2


def test_empty_ordered_input_is_safe(make_encoder):
    encoder = make_encoder()
    assert encoder([]).shape == (0, 384)
    assert encoder([]).shape == (0, 384)
    assert len(encoder.model.calls) == 1


def test_inference_created_weights_without_version_counters_bypass_cache(make_encoder):
    with torch.inference_mode():
        encoder = make_encoder()
        expected = encoder(['x'])
        actual = encoder(['x'])
    assert torch.equal(actual, expected)
    assert encoder.cache_info()['entries'] == 0
    assert len(encoder.model.calls) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_device_move_invalidates_cache_and_preserves_device(make_encoder):
    encoder = make_encoder(device='cpu')
    expected = encoder(['x'])
    encoder.model.cuda()
    actual = encoder(['x'])
    assert actual.is_cuda
    assert torch.equal(actual.cpu(), expected)
    assert encoder(['x']).is_cuda
    assert len(encoder.model.calls) == 2
