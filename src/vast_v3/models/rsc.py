"""Source-only response transport and discrete source proposal libraries (v1.3).

The decoder deliberately has no API for edge numerics, local skips, entity IDs,
or candidate-specific anomaly vectors. Only the shared masked driver sees h.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


Relation = tuple[int, int, int]
SourceLibrary = tuple[frozenset[int], ...]


class RSCDecoder(nn.Module):
    """Equations (9)-(10), with source membership batched on the first axis.

    ``active[e,t]`` means relation e is valid at receiving time t. Each valid
    incoming edge contributes lags whose sending time is in range. Visibility
    affects the shared driver, not the source-independent target eligibility.
    Relation categories and entity categories are integer vocabulary indices.
    All transport layers share W0, Wr and the structural attention parameters.
    """

    lags = (0, 1, 2, 4, 8)

    def __init__(self, width: int = 64, *, entity_types: int,
                 relation_types: int, metric_channels: int,
                 trace_channels: int, text_dim: int = 384, layers: int = 3):
        super().__init__()
        if min(width, entity_types, relation_types, metric_channels,
               trace_channels, text_dim, layers) < 1:
            raise ValueError("Widths, category counts and layers must be positive")
        self.width = width
        self.layers = layers
        self.entity_type_count = entity_types
        self.relation_type_count = relation_types
        self.driver = nn.Sequential(nn.Linear(width, 16, bias=False), nn.Tanh(),
                                    nn.Linear(16, 8, bias=False), nn.Tanh())
        self.source_injections = nn.ModuleList(
            nn.Linear(8, width, bias=False) for _ in range(entity_types))
        self.self_transform = nn.Linear(width, width, bias=False)
        self.relation_transforms = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(relation_types))
        # Relation, both endpoint categories, each endpoint's per-relation
        # in/out degrees, lag category, and edge validity. Nothing numerical.
        context_width = relation_types * 5 + entity_types * 2 + len(self.lags) + 1
        self.attention = nn.Sequential(nn.Linear(context_width, width, bias=False),
                                       nn.Tanh(), nn.Linear(width, 1, bias=False))
        self.node_heads = nn.ModuleDict({
            "metric": nn.Linear(width, metric_channels, bias=False),
            "log_frequency": nn.Linear(width, 1, bias=False),
            "text": nn.Linear(width, text_dim, bias=False),
        })
        self.edge_head = nn.Linear(2 * width, trace_channels, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def _validate(self, h, visible, entity_types, relations, active, source_sets):
        if h.ndim != 3 or h.shape[-1] != self.width:
            raise ValueError("h must have shape [N,T,width]")
        n, t, _ = h.shape
        if visible.shape != (n, t) or visible.dtype != torch.bool:
            raise ValueError("visible must be bool[N,T]")
        if entity_types.shape != (n,) or entity_types.dtype != torch.long:
            raise ValueError("entity_types must be long[N]")
        if n and (entity_types.min() < 0 or entity_types.max() >= self.entity_type_count):
            raise ValueError("Unknown entity category")
        if source_sets.ndim != 2 or source_sets.shape[1] != n or source_sets.dtype != torch.bool:
            raise ValueError("source_sets must be bool[H,N]")
        if active.shape != (len(relations), t) or active.dtype != torch.bool:
            raise ValueError("active must be bool[E,T]")
        for u, v, r in relations:
            if not (0 <= u < n and 0 <= v < n and 0 <= r < self.relation_type_count):
                raise ValueError("Relation endpoint or category is out of range")
        if not torch.isfinite(h[visible]).all():
            raise ValueError("Visible h must be finite")

    def _transport_plan(self, h, entity_types, relations, active):
        """Cache differentiable structural weights within this forward only."""
        n, steps, _ = h.shape
        r_count = self.relation_type_count
        categories = torch.eye(self.entity_type_count, device=h.device, dtype=h.dtype)
        relation_codes = torch.eye(r_count, device=h.device, dtype=h.dtype)
        lag_codes = torch.eye(len(self.lags), device=h.device, dtype=h.dtype)
        degrees = h.new_zeros(n, 2 * r_count)
        incoming = [[] for _ in range(n)]
        for e, (u, v, r) in enumerate(relations):
            degrees[u, r_count + r] += 1
            degrees[v, r] += 1
            incoming[v].append(e)
        degrees = torch.log1p(degrees)
        logits = {}
        for e, (u, v, r) in enumerate(relations):
            for j, _ in enumerate(self.lags):
                context = torch.cat((relation_codes[r], categories[entity_types[u]],
                                     categories[entity_types[v]], degrees[u], degrees[v],
                                     lag_codes[j], h.new_ones(1)))
                logits[e, j] = self.attention(context).squeeze(-1)
        plan = []
        for v in range(n):
            node_plan = []
            for t in range(steps):
                entries, weights = [], []
                for e in incoming[v]:
                    if not active[e, t]:
                        continue
                    u, _, r = relations[e]
                    for j, lag in enumerate(self.lags):
                        if t >= lag:
                            entries.append((r, u, t - lag))
                            weights.append(logits[e, j])
                node_plan.append((entries, torch.softmax(torch.stack(weights), 0)
                                  if weights else None))
            plan.append(node_plan)
        return plan

    def forward(self, h: Tensor, visible: Tensor, entity_types: Tensor,
                relations: Sequence[Relation], active: Tensor,
                source_sets: Tensor) -> Tensor:
        """Return states [H,N,T,width] from h[N,T,width] and bool sources[H,N]."""
        self._validate(h, visible, entity_types, relations, active, source_sets)
        n, steps, _ = h.shape
        if n == 0 or steps == 0:
            return h.new_zeros(source_sets.shape[0], n, steps, self.width)
        masked = torch.where(visible.unsqueeze(-1), h, torch.zeros_like(h))
        pooled = masked.sum(0) / visible.sum(0).clamp_min(1).unsqueeze(-1)
        driver = self.driver(pooled)
        typed = torch.stack([self.source_injections[int(category)](driver)
                             for category in entity_types], 0)
        injected = typed.unsqueeze(0) * source_sets[:, :, None, None]
        states = injected
        plan = self._transport_plan(h, entity_types, relations, active)
        for _ in range(self.layers):
            transformed = [transform(states) for transform in self.relation_transforms]
            messages = []
            for v in range(n):
                node_messages = []
                for t in range(steps):
                    entries, weights = plan[v][t]
                    if entries:
                        values = torch.stack([transformed[r][:, u, s]
                                              for r, u, s in entries], dim=1)
                        message = (values * weights[None, :, None]).sum(1)
                    else:
                        message = states.new_zeros(states.shape[0], self.width)
                    node_messages.append(message)
                messages.append(torch.stack(node_messages, dim=1))
            states = injected + torch.tanh(self.self_transform(states)
                                           + torch.stack(messages, dim=1))
        return states

    def predict_nodes(self, node_states: Tensor) -> dict[str, Tensor]:
        """Bias-free metric, log_frequency and centered-text residual heads."""
        return {name: head(node_states) for name, head in self.node_heads.items()}

    def predict_edges(self, node_states: Tensor, relations: Sequence[Relation]) -> Tensor:
        """One trace residual per directed edge, from paired endpoint states."""
        sources = [u for u, _, _ in relations]
        targets = [v for _, v, _ in relations]
        paired = torch.cat((node_states[:, sources], node_states[:, targets]), -1)
        return self.edge_head(paired)


def _cap(n: int, K_max: int | None) -> int:
    if not isinstance(n, int) or n < 0:
        raise ValueError("Candidate count must be nonnegative")
    if K_max is not None and (not isinstance(K_max, int) or K_max < 1):
        raise ValueError("K_max must be a declared positive integer or None")
    return min(n, K_max) if K_max is not None else n


def _unused_subset(n: int, k: int, used: set[frozenset[int]],
                   rng: random.Random) -> frozenset[int]:
    """Uniform unused subset, without constructing the full power set.

    Tiny single-cardinality supports are materialized (at most 32 sets). Large
    supports use uniform subset rejection, with at most 16 excluded proposals,
    so every draw has acceptance probability at least one half.
    """
    if comb(n, k) <= 32:
        remaining = [frozenset(s) for s in combinations(range(n), k)
                     if frozenset(s) not in used]
        return rng.choice(remaining)
    while True:
        proposed = frozenset(rng.sample(range(n), k))
        if proposed not in used:
            return proposed


def unlabeled_source_library(scores: Tensor, *, K_max: int | None = None,
                             seed: int = 0, max_extras: int = 16,
                             max_prefixes: int = 8) -> SourceLibrary:
    """All singletons plus capped prefixes and uniform-cardinality proposals.

    Membership is detached Python data. Callers cache it per case/epoch after
    masking; they must recompute scores and completion errors on every access.
    Candidate order is the canonical tie key; transport it under permutations.
    No hidden root argument is accepted.
    """
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("scores must be a finite candidate vector")
    if not 0 <= max_extras <= 16 or not 0 <= max_prefixes <= 8:
        raise ValueError("Proposal quotas exceed the reference maxima")
    n = len(scores)
    cap = _cap(n, K_max)
    ranking = sorted(range(n), key=lambda i: (-float(scores.detach()[i]), i))
    result = [frozenset([i]) for i in range(n)]
    used = {k: set() for k in range(2, cap + 1)}
    extras = []
    for k in range(2, min(9, cap) + 1):
        if len(extras) >= min(max_prefixes, max_extras):
            break
        prefix = frozenset(ranking[:k])
        used[k].add(prefix)
        extras.append(prefix)
    # Only sampled/prefix cardinalities need binomial counts. This avoids
    # enumerating the exponentially large support, even with no prior cap.
    available = [k for k in used if not used[k] or len(used[k]) < comb(n, k)]
    rng = random.Random(seed)
    while available and len(extras) < max_extras:
        k = rng.choice(available)
        proposal = _unused_subset(n, k, used[k], rng)
        used[k].add(proposal)
        extras.append(proposal)
        if len(used[k]) == comb(n, k):
            available.remove(k)
    return tuple(result + extras)


def labeled_source_library(num_candidates: int, true_sources: Iterable[int], *,
                           K_max: int | None = None, seed: int = 0,
                           max_wrong: int = 16) -> SourceLibrary:
    """True set first; all other sets replace roots with trusted nonroots.

    Assumes the supplied complete truth establishes the catalog complement as
    trusted nonroots. A cardinality-prior contradiction blocks rather than
    truncating the truth. No possible replacement means true completion only.
    """
    cap = _cap(num_candidates, K_max)
    truth = frozenset(true_sources)
    if not truth:
        raise ValueError("Labeled source completion requires a nonempty true root set")
    if any(not isinstance(v, int) or not 0 <= v < num_candidates for v in truth):
        raise ValueError("True source is outside the candidate catalog")
    if len(truth) > cap:
        raise ValueError("True cardinality contradicts K_max prior; expert/protocol revision required")
    if not 0 <= max_wrong <= 16:
        raise ValueError("At most 16 wrong source sets are permitted")
    used = {truth}
    result = [truth]
    rng = random.Random(seed)
    for _ in range(min(max_wrong, comb(num_candidates, len(truth)) - 1)):
        proposed = _unused_subset(num_candidates, len(truth), used, rng)
        used.add(proposed)
        result.append(proposed)
    return tuple(result)


def source_set_mask(library: Sequence[frozenset[int]], num_candidates: int, *,
                    device: torch.device | str | None = None) -> Tensor:
    """Convert cached membership to the bool[H,N] decoder/loss interface."""
    _cap(num_candidates, None)
    result = torch.zeros(len(library), num_candidates, dtype=torch.bool, device=device)
    for row, sources in enumerate(library):
        if any(not 0 <= v < num_candidates for v in sources):
            raise ValueError("Source is outside the candidate catalog")
        result[row, list(sources)] = True
    return result
