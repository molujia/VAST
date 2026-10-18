"""Shared, candidate-marked SCOPE with owner-bound numerical edge skips."""

from __future__ import annotations

from itertools import groupby, permutations, product

import torch
from torch import Tensor, nn

from .fragments import Fragment, ordered_paths


def signature_order_two(increments: Tensor) -> Tensor:
    """Piecewise-linear signature (1, level 1, flattened level 2), differentiable."""
    if increments.ndim != 2:
        raise ValueError("increments must have shape [steps, channels]")
    q = increments.shape[1]
    first = increments.new_zeros(q)
    second = increments.new_zeros(q, q)
    for increment in increments:
        second = second + torch.outer(first, increment) + 0.5 * torch.outer(increment, increment)
        first = first + increment
    return torch.cat((increments.new_ones(1), first, second.flatten()))


class ScopeEncoder(nn.Module):
    """One encoder for unmarked acquisition and marked candidate readouts.

    ``availability``/``coverage`` channels are numeric, log, topology.  Edge
    ownership is addressed by ``FragmentEdge.owner``; bins between transition
    endpoints are pooled using ``edge_mask``.  Delay is in seconds between
    interval midpoints.  Provenance, names and numeric entity indices are never
    serialized; entity indices are used only to gather states and test equality.
    """

    def __init__(self, width: int = 64, entity_types: int = 2, relation_types: int = 3,
                 max_edges: int = 3, q: int = 8, p: int = 2) -> None:
        super().__init__()
        if width < 1 or entity_types < 1 or relation_types < 1 or max_edges != 3 or q < 2 or p != 2:
            raise ValueError("SCOPE requires positive widths, max_edges=3, q>=2 and p=2")
        self.width, self.entity_types, self.relation_types = width, entity_types, relation_types
        self.max_edges, self.q, self.p = max_edges, q, p
        # category indicators; typed/response adjacency; entity equality;
        # four-way order (before/after/unknown/padding); four fixed mark slots;
        # event validity and edge validity.  No raw IDs enter the code.
        self.code_width = 4 * entity_types + 32 * relation_types + 16 + 64 + 16 + 4 + 3
        self.role_projection = nn.Linear(self.code_width, 32)
        self.edge_width = width + 4  # numeric mean, availability, coverage, two direction flags
        increment_input = 2 * width + self.edge_width + relation_types + 32 + 12
        self.increment_mlp = nn.Sequential(nn.Linear(increment_input, 64), nn.GELU(), nn.Linear(64, q - 1))
        self.branch_mlp = nn.Sequential(nn.Linear(1 + q + q * q + 32 + 1, 64), nn.GELU(), nn.Linear(64, 64))
        token_input = 129 + 2 * width + 3 * self.edge_width + 32 + 24
        self.token_mlp = nn.Sequential(nn.Linear(token_input, 64), nn.GELU(), nn.Linear(64, width))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _serialize(self, fragment: Fragment, permutation: tuple[int, ...], candidate: int | None,
                   endpoint: tuple[int, int] | None, branch: frozenset[int]) -> tuple[int, ...]:
        n = len(fragment.events)
        position = {old: new for new, old in enumerate(permutation)}
        output: list[int] = []
        for index in range(4):
            output.extend(int(index < n and fragment.events[permutation[index]].category == category)
                          for category in range(self.entity_types))
        adjacency = [[[0 for _ in range(2 * self.relation_types)] for _ in range(4)] for _ in range(4)]
        for edge in fragment.edges:
            adjacency[position[edge.source]][position[edge.target]][2 * edge.relation + int(edge.response)] += 1
        output.extend(value for row in adjacency for cell in row for value in cell)
        for i in range(4):
            for j in range(4):
                output.append(int(i < n and j < n and
                                  fragment.events[permutation[i]].entity == fragment.events[permutation[j]].entity))
        for i in range(4):
            for j in range(4):
                order = 3
                if i < n and j < n:
                    left, right = fragment.events[permutation[i]], fragment.events[permutation[j]]
                    order = 0 if left.end < right.start else (1 if right.end < left.start else 2)
                output.extend(int(order == value) for value in range(4))
        for i in range(4):
            old = permutation[i] if i < n else -1
            output.extend((int(i < n and candidate is not None and fragment.events[old].entity == candidate),
                           int(endpoint is not None and old == endpoint[0]),
                           int(endpoint is not None and old == endpoint[1]), int(old in branch)))
        output.extend(int(i < n) for i in range(4))
        output.extend(int(i < len(fragment.edges)) for i in range(3))
        return tuple(output)

    def _canonical(self, fragment: Fragment, candidate: int | None,
                   endpoint: tuple[int, int] | None = None,
                   branch: frozenset[int] = frozenset()) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        best = None
        mappings = []
        for permutation in permutations(range(len(fragment.events))):
            code = self._serialize(fragment, permutation, candidate, endpoint, branch)
            if best is None or code < best:
                best, mappings = code, [permutation]
            elif code == best:
                mappings.append(permutation)
        assert best is not None
        return best, tuple(mappings)

    @staticmethod
    def _edge_orders(fragment: Fragment, position: dict[int, int]) -> tuple[tuple[int, ...], ...]:
        def key(index: int) -> tuple:
            edge = fragment.edges[index]
            return position[edge.source], position[edge.target], edge.relation, edge.response
        ordered = sorted(range(len(fragment.edges)), key=key)
        # Anonymous parallel-edge ties cannot be broken by owner array indices.
        groups = [tuple(group) for _, group in groupby(ordered, key=key)]
        return tuple(tuple(index for group in combination for index in group)
                     for combination in product(*(tuple(permutations(group)) for group in groups)))

    def _edge_slot(self, fragment: Fragment, edge_index: int, position: dict[int, int],
                   states: Tensor, mask: Tensor) -> Tensor:
        edge = fragment.edges[edge_index]
        first, last = sorted((fragment.events[edge.source].bin, fragment.events[edge.target].bin))
        current_direction = 1.0 if position[edge.source] <= position[edge.target] else -1.0
        owner_direction = (current_direction * (-1.0 if edge.response else 1.0)) if edge.owner >= 0 else 0.0
        numeric = states.new_zeros(self.width)
        available = coverage = 0.0
        if edge.owner >= 0:
            if edge.owner >= states.shape[0] or last >= states.shape[1]:
                raise ValueError("edge owner/time index is outside edge_states")
            valid = mask[edge.owner, first:last + 1].bool()
            if bool(valid.any()):
                numeric = states[edge.owner, first:last + 1][valid].mean(0)
                available = 1.0
                coverage = float(valid.float().mean())
        return torch.cat((numeric, states.new_tensor((available, coverage, owner_direction, current_direction))))

    def forward(self, fragment: Fragment, h: Tensor, local: Tensor, edge_states: Tensor,
                availability: Tensor, coverage: Tensor, edge_mask: Tensor,
                candidate: int | None = None) -> tuple[Tensor, Tensor]:
        if not fragment.events or not fragment.edges:
            raise ValueError("empty fragments are not observed occurrences")
        if len(fragment.events) > 4 or len(fragment.edges) > self.max_edges:
            raise ValueError("fragment exceeds four-event/three-edge contract")
        if h.ndim != 3 or h.shape[-1] != self.width or local.shape != h.shape:
            raise ValueError("h and local must have matching [N,T,width] shape")
        if availability.shape != h.shape[:2] + (3,) or coverage.shape != availability.shape:
            raise ValueError("availability and coverage must have shape [N,T,3]")
        if edge_states.ndim != 3 or edge_states.shape[-1] != self.width or edge_mask.shape != edge_states.shape[:2]:
            raise ValueError("edge_states and edge_mask shapes do not agree")
        for event in fragment.events:
            if event.entity >= h.shape[0] or event.bin >= h.shape[1] or event.category >= self.entity_types:
                raise ValueError("event entity/time/category is outside declared schema")
        if any(edge.relation >= self.relation_types for edge in fragment.edges):
            raise ValueError("relation is outside declared schema")
        if candidate is not None and not any(event.entity == candidate for event in fragment.events):
            raise ValueError("candidate is absent from fragment")
        code, mappings = self._canonical(fragment, candidate)
        role = self.role_projection(h.new_tensor(code))
        paths = ordered_paths(fragment)
        role_cache: dict[tuple, Tensor] = {}

        def marked_role(endpoint=None, branch=frozenset()):
            key = endpoint, branch
            if key not in role_cache:
                marked_code, _ = self._canonical(fragment, candidate, endpoint, branch)
                role_cache[key] = self.role_projection(h.new_tensor(marked_code))
            return role_cache[key]

        outputs = []
        for mapping in mappings:
            position = {old: new for new, old in enumerate(mapping)}
            slots = [self._edge_slot(fragment, index, position, edge_states, edge_mask)
                     for index in range(len(fragment.edges))]
            branch_values = []
            for path in paths:
                increments = []
                branch_vertices = set()
                for index in path:
                    edge = fragment.edges[index]
                    source, target = fragment.events[edge.source], fragment.events[edge.target]
                    branch_vertices.update((edge.source, edge.target))
                    relation = h.new_zeros(self.relation_types)
                    relation[edge.relation] = 1
                    source_mask = availability[source.entity, source.bin].to(h)
                    target_mask = availability[target.entity, target.bin].to(h)
                    metadata = torch.cat((source_mask, target_mask,
                                          coverage[source.entity, source.bin].to(h),
                                          coverage[target.entity, target.bin].to(h)))
                    phi_input = torch.cat((h[source.entity, source.bin], h[target.entity, target.bin],
                                           slots[index], relation, marked_role((edge.source, edge.target)), metadata))
                    delay = (target.start + target.end - source.start - source.end) / 2.0
                    increments.append(torch.cat((self.increment_mlp(phi_input), h.new_tensor([delay]))))
                signature = signature_order_two(torch.stack(increments))
                branch_input = torch.cat((signature, marked_role(branch=frozenset(branch_vertices)), h.new_ones(1)))
                branch_values.append(self.branch_mlp(branch_input))
            if branch_values:
                stacked = torch.stack(branch_values)
                branch_pool = torch.cat((stacked.mean(0), stacked.amax(0), h.new_tensor([len(branch_values)])))
            else:
                branch_pool = h.new_zeros(129)
            local_values = []
            mask_rows, coverage_rows = [], []
            for old in mapping:
                event = fragment.events[old]
                visible = availability[event.entity, event.bin].to(h)
                mask_rows.append(visible)
                coverage_rows.append(coverage[event.entity, event.bin].to(h))
                if bool(visible[:2].bool().any()):
                    local_values.append(local[event.entity, event.bin])
            if local_values:
                local_stack = torch.stack(local_values)
                local_pool = torch.cat((local_stack.mean(0), local_stack.amax(0)))
            else:
                local_pool = h.new_zeros(2 * self.width)
            while len(mask_rows) < 4:
                mask_rows.append(h.new_zeros(3))
                coverage_rows.append(h.new_zeros(3))
            metadata = torch.cat((*mask_rows, *coverage_rows))
            for edge_order in self._edge_orders(fragment, position):
                edge_slots = [slots[index] for index in edge_order]
                edge_slots.extend(h.new_zeros(self.edge_width) for _ in range(3 - len(edge_slots)))
                token_input = torch.cat((branch_pool, local_pool, *edge_slots, role, metadata))
                outputs.append(self.token_mlp(token_input))
        y = torch.stack(outputs).mean(0)
        norm = torch.linalg.vector_norm(y)
        fallback = torch.zeros_like(y)
        fallback[0] = 1
        z = torch.where(norm > 1e-12, y / norm.clamp_min(1e-12), fallback)
        return z, role
