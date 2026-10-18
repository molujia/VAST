"""Train-reference-only robust scaling, physical time, and identity sanitization."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import re
import numpy as np
from .schema import ObservableCase


LEGACY_SCHEMA = 'vast-v3-preprocessing/1'
SCHEMA = 'vast-v3-preprocessing/2'
LEGACY_SCALING_POLICY = 'median_mad_iqr_unit'
SCALING_POLICY = 'median_mad_iqr_population_std_unit_if_constant'


@dataclass(frozen=True)
class Scale:
    center: float
    scale: float
    observations: int


@dataclass(frozen=True)
class Preprocessor:
    stats: dict[tuple[str,str],Scale]
    delta: float
    fitted_case_ids: tuple[str,...]
    gap_series_count: int
    text_reference_mean: tuple[float,...] = ()
    text_model_identity: str | None = None
    schema: str = SCHEMA
    scaling_policy: str = SCALING_POLICY

    def __post_init__(self):
        policies = {LEGACY_SCHEMA: LEGACY_SCALING_POLICY, SCHEMA: SCALING_POLICY}
        if self.schema not in policies:
            raise ValueError('unsupported preprocessing schema')
        if self.scaling_policy != policies[self.schema]:
            raise ValueError('preprocessing scaling policy does not match schema')

    @classmethod
    def fit(cls, cases: list[ObservableCase], *, permitted_case_ids: set[str],text_encoder=None,text_identity=None):
        if any(c.case_id not in permitted_case_ids for c in cases):
            raise ValueError('fit case outside permitted training population')
        if len({c.case_id for c in cases}) != len(cases):
            raise ValueError('duplicate fit case')
        return cls._fit_two_passes(cases, cases, tuple(sorted(c.case_id for c in cases)),
                                   text_encoder=text_encoder, text_identity=text_identity)

    @classmethod
    def fit_repository(cls, repository, ordered_case_ids, *, permitted_case_ids: set[str],
                       text_encoder=None, text_identity=None):
        """Fit exactly from two ordered repository passes, without a case list.

        The caller must provide a stable/content-pinned repository. Only the
        explicit ID sequence is inspected before loading any observations;
        repository iteration could include nonpermitted/test cases. Exact
        reference scalar pools still reside in memory; no quantile is sampled.
        """
        case_ids = tuple(ordered_case_ids)
        if any(case_id not in permitted_case_ids for case_id in case_ids):
            raise ValueError('fit case outside permitted training population')
        if len(set(case_ids)) != len(case_ids):
            raise ValueError('duplicate fit case')

        def cases():
            for case_id in case_ids:
                case = repository[case_id]
                if case.case_id != case_id:
                    raise ValueError('repository case identity differs from requested case ID')
                yield case

        return cls._fit_two_passes(cases(), cases(), tuple(sorted(case_ids)),
                                   text_encoder=text_encoder, text_identity=text_identity)

    @classmethod
    def _fit_two_passes(cls, numeric_cases, log_cases, fitted_case_ids, *, text_encoder, text_identity):
        pools, gaps = {}, []
        for c in numeric_cases:
            types = {e.id:e.category for e in c.entities}
            edge_types = {r.id:types[r.source]+'->'+types[r.target] for r in c.relations}
            for s in c.numeric:
                category = types[s.owner] if s.modality == 'metric' else edge_types[s.owner]
                observed = [(t,v) for t,v,m in zip(s.times,s.values,s.observed)
                            if m and c.reference_start <= t < c.incident_start]
                pools.setdefault((category,s.channel),[]).extend(v for _,v in observed)
                if len(observed) >= 3:
                    dt = np.diff([t for t,_ in observed])
                    dt = dt[dt > 0]
                    if len(dt):
                        gaps.append(float(np.median(dt)))
        if not gaps:
            raise ValueError('no eligible training reference series to determine grid interval')
        delta=float(np.median(gaps))
        text_sum=np.zeros(384,dtype=np.float64); text_mass=0
        if text_encoder is not None and not text_identity:
            raise ValueError('fixed text model identity is required for reference fitting')
        for c in log_cases:
            types={e.id:e.category for e in c.entities}
            frequencies={}
            texts=[]; masses=[]
            for log in c.logs:
                if c.reference_start<=log.time<c.incident_start:
                    key=(log.owner,c.bin_index(log.time,delta))
                    frequencies[key]=frequencies.get(key,0)+log.frequency
                    if text_encoder is not None:
                        texts.append(sanitize_log(log.text,tuple(types))); masses.append(log.frequency)
            for (owner,_),frequency in frequencies.items():
                pools.setdefault((types[owner],'__log_frequency__'),[]).append(math.log1p(frequency))
            for offset in range(0,len(texts),256):
                embedded=text_encoder(texts[offset:offset+256]).detach().cpu().numpy()
                weights=np.asarray(masses[offset:offset+256])
                if embedded.shape!=(len(weights),384) or not np.isfinite(embedded).all():
                    raise ValueError('invalid fixed reference text embeddings')
                text_sum+=(embedded*weights[:,None]).sum(0); text_mass+=int(weights.sum())
        stats = {}
        for key,values in pools.items():
            if not values:
                continue
            arr = np.asarray(values)
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    center = float(np.median(arr))
                    residuals = arr-center
                    scale = float(1.4826*np.median(np.abs(residuals)))
                    if scale == 0:
                        scale = float((np.quantile(arr,.75)-np.quantile(arr,.25))/1.349)
                    if scale == 0:
                        # Same permitted reference pool, population std (ddof=0).
                        # Rescaling residuals avoids squaring huge native units;
                        # std remains centered on its own mean, not the median.
                        amplitude = float(np.max(np.abs(residuals)))
                        scale = (float(np.std(residuals/amplitude, ddof=0))*amplitude
                                 if amplitude else 1.)
            except FloatingPointError as exc:
                raise ValueError(f'reference scaling statistics must be finite for {key!r}') from exc
            if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f'reference scaling statistics must be finite and positive for {key!r}')
            stats[key] = Scale(center,scale,len(values))
        return cls(stats,delta,fitted_case_ids,len(gaps),
                   tuple((text_sum/text_mass).tolist()) if text_mass else (),text_identity)

    def transform(self, category: str, channel: str, value: float):
        stat = self.stats.get((category,channel))
        if stat is None or not math.isfinite(value):
            return 0.,False
        return (value-stat.center)/stat.scale,True

    def to_dict(self):
        payload = {'schema':self.schema,'delta':self.delta,
                'fitted_case_ids':list(self.fitted_case_ids),'gap_series_count':self.gap_series_count,
                'text_reference_mean':list(self.text_reference_mean),'text_model_identity':self.text_model_identity,
                'stats':[{'category':k[0],'channel':k[1],**asdict(v)} for k,v in sorted(self.stats.items())]}
        if self.schema == SCHEMA:
            payload['scaling_policy'] = self.scaling_policy
        return payload

    @classmethod
    def from_dict(cls, data):
        schema = data.get('schema')
        if schema == LEGACY_SCHEMA:
            if 'scaling_policy' in data:
                raise ValueError('legacy preprocessing schema cannot contain a scaling policy')
            policy = LEGACY_SCALING_POLICY
        elif schema == SCHEMA:
            policy = data.get('scaling_policy')
            if policy != SCALING_POLICY:
                raise ValueError('unsupported preprocessing scaling policy')
        else:
            raise ValueError('unsupported preprocessing schema')
        return cls({(r['category'],r['channel']):Scale(r['center'],r['scale'],r['observations']) for r in data['stats']},
                   data['delta'],tuple(data['fitted_case_ids']),data['gap_series_count'],
                   tuple(data.get('text_reference_mean',())),data.get('text_model_identity'),
                   schema=schema,scaling_policy=policy)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


@lru_cache(maxsize=64)
def _sanitization_patterns(catalog_ids: tuple[str,...]):
    # Preserve stable longest-first *sequential* substitutions, including
    # collisions with replacement markers. A single alternation is not equal.
    # Cache only bounded catalog patterns, never native/private log strings.
    return tuple((entity_id, re.compile(r'(?<![\w])'+re.escape(entity_id)+r'(?![\w])'))
                 for entity_id in sorted(catalog_ids,key=len,reverse=True) if entity_id)


_IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_UUID_PATTERN = re.compile(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b')
_REQUEST_PATTERN = re.compile(r'(?i)\b(request[_-]?id|trace[_-]?id|span[_-]?id)\s*[=:]\s*[\w-]+')


def sanitize_log(text: str, catalog_ids: tuple[str,...]) -> str:
    for entity_id, pattern in _sanitization_patterns(tuple(catalog_ids)):
        # A missing literal cannot match its escaped/bounded regex. Evaluate
        # against the current text, not the original, to preserve cascades.
        if entity_id in text:
            text = pattern.sub('<ENTITY>',text)
    text = _IP_PATTERN.sub('<IP>',text)
    text = _UUID_PATTERN.sub('<UUID>',text)
    text = _REQUEST_PATTERN.sub(r'\1=<REQUEST>',text)
    return text
