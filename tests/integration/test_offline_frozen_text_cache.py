"""Opt-in, no-download bitwise check against the real pinned Sentence Transformer."""
import json
import os
from pathlib import Path

import pytest
import torch

from vast_v3.encoders.text import FrozenSentenceEncoder


@pytest.mark.skipif(not os.environ.get('VAST_V3_TEXT_SNAPSHOT'),
                    reason='set VAST_V3_TEXT_SNAPSHOT for the pinned offline text checkpoint')
def test_real_offline_checkpoint_cache_matches_uncached_encode_bitwise(tmp_path):
    directory = os.environ['VAST_V3_TEXT_SNAPSHOT']
    lock_path = Path(os.environ.get('VAST_V3_TEXT_LOCK', 'configs/text-minilm-lock.json'))
    manifest = json.loads(lock_path.read_text(encoding='utf-8'))
    device = os.environ.get('VAST_V3_TEXT_DEVICE', 'cpu')
    encoder = FrozenSentenceEncoder(directory, manifest, device=device, staging_parent=tmp_path)
    # Cross two batch boundaries. Preserve repetitions and unequal token lengths.
    texts = [('service request failed: ' + 'retry ' * (j % 31) + str(j % 7)) for j in range(131)]
    rng_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    cold = encoder(texts)
    warm = encoder(list(texts))
    assert encoder.cache_info()['hits'] == 1
    with torch.no_grad():
        reference = encoder.model.encode(texts, batch_size=64, convert_to_tensor=True,
                                          show_progress_bar=False, normalize_embeddings=False)
    assert torch.equal(cold, reference)
    assert torch.equal(warm, reference)
    warm.fill_(0)
    assert torch.equal(encoder(texts), reference)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_before))
