"""Fixed v1.3 epoch schedules and independently shuffled eligible streams."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import random
from typing import Iterable, Mapping
from .variants import NAMES, variant_components


@dataclass(frozen=True)
class TrainingSchedule:
    budget: int = 30
    query_batch_size: int = 5
    warmup_epochs: int = 20
    nonfinal_epochs: int = 20
    final_epochs: int = 50
    u_disabled_epochs: int = 5
    u_ramp_epochs: int = 5
    u_weight: float = 0.1
    case_batch_size: int = 4
    observation_weight: float = 0.2
    source_weight: float = 0.2

    def __post_init__(self):
        for name in ("budget", "warmup_epochs", "u_disabled_epochs", "u_ramp_epochs"):
            _nonnegative_int(getattr(self, name), name)
        for name in ("query_batch_size", "nonfinal_epochs", "final_epochs", "case_batch_size"):
            _nonnegative_int(getattr(self, name), name)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in ('observation_weight', 'source_weight', 'u_weight'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def loss_weights(self, variant='A3', *, epoch, phase, u_enabled=True):
        """Fixed independent objective means; never renormalize missing terms."""
        enabled = variant_components(variant)
        if phase not in ('warmup', 'joint'):
            raise ValueError('phase must be warmup or joint')
        local = self.observation_weight if variant == 'A1' else self.observation_weight / 3
        weights = dict(pn=1., source=self.source_weight, local_recovery=local,
                       process_recovery=self.observation_weight / 3,
                       invariance=self.observation_weight / 3,
                       U_soft=self.u_weight_at(epoch, enabled=u_enabled))
        if phase == 'warmup':
            weights.update(pn=0., source=0., U_soft=0.)
        return {name: weights[name] if name in enabled else 0. for name in NAMES}

    def query_batches(self, pool_size: int) -> tuple[int, ...]:
        """Sequential complete-case queries, capped by budget and remaining U."""
        _nonnegative_int(pool_size, "pool_size")
        remaining = min(self.budget, pool_size)
        result = []
        while remaining:
            batch = min(self.query_batch_size, remaining)
            result.append(batch)
            remaining -= batch
        return tuple(result)

    def round_epochs(self, round_index: int, total_rounds: int) -> int:
        _nonnegative_int(round_index, "round_index")
        _nonnegative_int(total_rounds, "total_rounds")
        if round_index >= total_rounds:
            raise ValueError("round_index must identify an existing round")
        return self.final_epochs if round_index == total_rounds - 1 else self.nonfinal_epochs

    def u_weight_at(self, epoch: int, *, enabled: bool = True) -> float:
        """Zero-based joint epoch: 0..4 off, 5..9 ramp 1/5..5/5 of weight."""
        _nonnegative_int(epoch, "epoch")
        if not enabled or epoch < self.u_disabled_epochs:
            return 0.0
        if self.u_ramp_epochs == 0:
            return self.u_weight
        return self.u_weight * min(1.0, (epoch - self.u_disabled_epochs + 1) / self.u_ramp_epochs)

    def update_slots(self, pool_size: int) -> int:
        """Common slots use the permitted pool size, not individual streams."""
        _nonnegative_int(pool_size, "pool_size")
        return (pool_size + self.case_batch_size - 1) // self.case_batch_size

    def stages(self, regime: str, *, pool_size: int, a0: bool = False) -> tuple[dict, ...]:
        """Describe durations without owning models or selecting checkpoints.

        Fixed-list controls must instantiate their own prescribed initialization
        and warm-up; these descriptors never carry selector-trained weights.
        Warm-up has observation objectives only. A0 skips it and never sees U.
        All-label/LOFO have no acquisition and no unlabeled explanation term.
        """
        if regime not in {"closed_loop", "fixed_list", "all_label", "lofo"}:
            raise ValueError("Unknown training regime")
        _nonnegative_int(pool_size, "pool_size")
        result = []
        if self.warmup_epochs and not a0:
            result.append(dict(stage="warmup", phase="warmup", epochs=self.warmup_epochs,
                               query_count=0, round_index=None, u_enabled=False))
        u_enabled = not a0 and regime not in {"all_label", "lofo"}
        if regime == "closed_loop":
            batches = self.query_batches(pool_size)
            for index, count in enumerate(batches):
                result.append(dict(stage=f"round-{index}", phase="joint",
                                   epochs=self.round_epochs(index, len(batches)), query_count=count,
                                   round_index=index, u_enabled=u_enabled))
        else:
            result.append(dict(stage="joint", phase="joint", epochs=self.final_epochs,
                               query_count=min(self.budget, pool_size) if regime == "fixed_list" else 0,
                               round_index=0, u_enabled=u_enabled))
        return tuple(result)


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


class IndependentEligibleStreams:
    """Independent finite shuffled decks, one per objective's eligible cases.

    A draw returns up to batch_size cases without replacement. A short tail is
    returned as-is; the following draw reshuffles that component's whole deck.
    Empty components return (), so callers can retain fixed objective weights
    while omitting their gradient. Adding/drawing other components cannot
    perturb a stream. Capture this object's state in checkpoint.extra_state.
    """

    def __init__(self, populations: Mapping[str, Iterable[str]], *, seed: int,
                 batch_size: int = 4):
        _nonnegative_int(batch_size, "batch_size")
        if batch_size == 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.populations = {}
        self._streams = {}
        for name, case_ids in populations.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Component names must be nonempty strings")
            population = tuple(case_ids)
            if any(not isinstance(case_id, str) or not case_id for case_id in population):
                raise ValueError("Case IDs must be nonempty strings")
            if len(set(population)) != len(population):
                raise ValueError("Each eligible population must have no duplicate case IDs")
            self.populations[name] = population
            key = int.from_bytes(sha256(f"{seed}\0{name}".encode()).digest(), "big")
            rng = random.Random(key)
            deck = list(population)
            rng.shuffle(deck)
            self._streams[name] = dict(rng=rng, deck=deck, position=0)

    def draw(self, component: str) -> tuple[str, ...]:
        stream = self._streams[component]
        if not stream["deck"]:
            return ()
        if stream["position"] == len(stream["deck"]):
            stream["deck"] = list(self.populations[component])
            stream["rng"].shuffle(stream["deck"])
            stream["position"] = 0
        start = stream["position"]
        end = min(start + self.batch_size, len(stream["deck"]))
        stream["position"] = end
        return tuple(stream["deck"][start:end])

    def draw_all(self) -> dict[str, tuple[str, ...]]:
        return {name: self.draw(name) for name in self.populations}

    def state_dict(self) -> dict:
        return {"schema": "vast-v3-eligible-streams/1", "batch_size": self.batch_size,
                "populations": {name: list(values) for name, values in self.populations.items()},
                "streams": {name: {"deck": list(stream["deck"]), "position": stream["position"],
                                    "rng_state": stream["rng"].getstate()}
                            for name, stream in self._streams.items()}}

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("schema") != "vast-v3-eligible-streams/1" or state.get("batch_size") != self.batch_size:
            raise ValueError("Incompatible stream schema or batch size")
        expected = {name: list(values) for name, values in self.populations.items()}
        if state.get("populations") != expected or set(state.get("streams", {})) != set(expected):
            raise ValueError("Eligible population mismatch during stream restore")
        restored = {}
        for name, population in self.populations.items():
            stored = state["streams"][name]
            deck, position = list(stored["deck"]), stored["position"]
            _nonnegative_int(position, "stream position")
            if sorted(deck) != sorted(population) or position > len(deck):
                raise ValueError("Invalid stream deck or cursor")
            rng = random.Random(0)
            rng.setstate(stored["rng_state"])
            restored[name] = dict(rng=rng, deck=deck, position=position)
        self._streams = restored
