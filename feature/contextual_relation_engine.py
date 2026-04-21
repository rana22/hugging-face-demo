"""Generic contextual-text relation engine.

Designed for nodes whose fields are narrative / semi-structured biomedical text.
It avoids hardcoded project-specific field values by relying on a node contract.

Typical uses:
- build per-property profiles from a record
- score property-pair relations within a record
- flag structural / semantic issues
- optionally use an NLI backend for pairwise relation scoring

The module is intentionally light on dependencies.
Optional NLI backend uses Hugging Face transformers + torch.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    BASE_STOPWORDS = set(ENGLISH_STOP_WORDS)
except Exception:  # pragma: no cover
    BASE_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
        "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
        "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
    }

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/]*")
META_KEY_RE = re.compile(r"(^|_)(uuid|created|updated|timestamp|type|record_id|id|url|link)$", re.I)
ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}.*z?", re.I)
HTTP_RE = re.compile(r"https?://", re.I)

# Generic, project-agnostic noise terms. Keep this small.
DEFAULT_NOISE_TERMS = {
    "study", "studies", "data", "analysis", "result", "results", "important",
    "inconclusive", "general", "wellbeing", "performance", "computer", "computing",
    "network", "machine", "cloud", "analytics", "user", "behavior", "infrastructure",
}

# Default role compatibility. The contract can override / extend this.
DEFAULT_ROLE_COMPATIBILITY = {
    ("anchor_entity", "supporting_entity"): 0.65,
    ("supporting_entity", "supporting_entity"): 0.55,
    ("anchor_entity", "anchor_entity"): 0.75,
    ("intervention", "anchor_entity"): 0.58,
    ("intervention", "supporting_entity"): 0.60,
    ("narrative_context", "anchor_entity"): 0.40,
    ("narrative_context", "supporting_entity"): 0.42,
    ("narrative_context", "intervention"): 0.38,
    ("narrative_context", "narrative_context"): 0.30,
}


@dataclass
class NodeContract:
    """Metadata describing a node.

    property_roles:
        mapping of property key -> {role, entity_type}
    anchor_fields:
        fields that define the study / anchor context
    negative_terms:
        optional domain-independent or project-specific noise lexicon
    relation_thresholds:
        score thresholds for labels
    """

    node_name: str
    data_mode: str = "contextual_text"
    property_roles: Dict[str, Dict[str, str]] = field(default_factory=dict)
    anchor_fields: List[str] = field(default_factory=list)
    negative_terms: List[str] = field(default_factory=list)
    relation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "related": 0.55,
        "complementary": 0.70,
        "unrelated": 0.35,
    })
    fk_fields: List[str] = field(default_factory=list)
    pk_field: Optional[str] = None


@dataclass
class PropertyProfile:
    key: str
    raw_value: Any
    text: str
    tokens: List[str]
    normalized_tokens: List[str]
    role: str = "unknown"
    entity_type: str = "unknown"
    token_count: int = 0
    unique_token_count: int = 0
    noise_hits: int = 0
    noise_ratio: float = 0.0
    anchor_like: bool = False
    confidence: float = 0.0


@dataclass
class PairRelation:
    property_a: str
    property_b: str
    relation: str
    score: float
    components: Dict[str, float]
    shared_tokens: List[str]
    notes: List[str] = field(default_factory=list)


# --------------------------
# Basic text utilities
# --------------------------

def parse_listish(value: Any) -> List[str]:
    """Parse JSON-string lists, plain strings, or empty-list strings into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if not isinstance(value, str):
        s = str(value).strip()
        return [s] if s else []

    s = value.strip()
    if not s:
        return []

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except Exception:
        pass

    # If it is a textual value, keep it as a single item.
    return [s]


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def looks_like_metadata_key(key: str) -> bool:
    key_l = (key or "").lower()
    return bool(META_KEY_RE.search(key_l) or key_l.endswith("_id") or key_l.endswith("_type"))


def looks_like_metadata_value(value: str) -> bool:
    s = (value or "").strip()
    if not s:
        return True
    if HTTP_RE.match(s):
        return True
    if ISO_TS_RE.fullmatch(s.lower()):
        return True
    return False


def build_stopword_set(dynamic_stop_terms: Optional[Iterable[str]] = None) -> set:
    dynamic_stop_terms = dynamic_stop_terms or []
    return BASE_STOPWORDS.union({str(t).lower() for t in dynamic_stop_terms if str(t).strip()})


# --------------------------
# Structural helpers
# --------------------------

def extract_study_context(record: Mapping[str, Any], contract: NodeContract) -> Dict[str, Any]:
    """Return primary anchor / fk context for semantic alignment."""
    ctx: Dict[str, Any] = {}

    for field in contract.anchor_fields:
        if field in record and record[field] is not None:
            ctx[field] = record[field]

    for field in contract.fk_fields:
        if field in record and record[field] is not None:
            ctx[field] = record[field]

    if contract.pk_field and contract.pk_field in record and record[contract.pk_field] is not None:
        ctx[contract.pk_field] = record[contract.pk_field]

    return ctx


def _flatten_record_text(record: Mapping[str, Any], contract: NodeContract) -> str:
    parts: List[str] = []
    stop = set(contract.negative_terms)

    for key, value in record.items():
        if looks_like_metadata_key(str(key)):
            continue
        if value is None:
            continue
        if key in contract.fk_fields:
            continue

        values = parse_listish(value)
        if not values:
            continue

        for item in values:
            item = normalize_text(item)
            if not item or looks_like_metadata_value(item):
                continue
            if item.lower() in stop:
                continue
            parts.append(item)

    return normalize_text(" ".join(parts))


# --------------------------
# Profiling
# --------------------------

def _role_for_key(key: str, contract: NodeContract) -> Tuple[str, str]:
    meta = contract.property_roles.get(key, {})
    return meta.get("role", "unknown"), meta.get("entity_type", "unknown")


def build_property_profiles(record: Mapping[str, Any], contract: NodeContract) -> Dict[str, PropertyProfile]:
    stop = build_stopword_set(contract.negative_terms)
    profiles: Dict[str, PropertyProfile] = {}

    for key, value in record.items():
        if looks_like_metadata_key(str(key)):
            continue
        if value is None:
            continue

        items = parse_listish(value)
        if not items:
            continue

        text = normalize_text(" ".join(items))
        if not text or looks_like_metadata_value(text):
            continue

        tokens = [t for t in tokenize(text) if t not in stop and len(t) > 2 and "_" not in t]
        if not tokens:
            continue

        role, entity_type = _role_for_key(str(key), contract)
        noise_hits = sum(1 for t in tokens if t in stop or t in set(x.lower() for x in contract.negative_terms))
        noise_ratio = noise_hits / max(len(tokens), 1)

        # Confidence is intentionally simple here; the validator uses pair signals too.
        confidence = max(0.15, min(1.0, 1.0 - (noise_ratio * 0.8)))
        anchor_like = key in set(contract.anchor_fields)

        profiles[str(key)] = PropertyProfile(
            key=str(key),
            raw_value=value,
            text=text,
            tokens=tokens,
            normalized_tokens=[t.lower() for t in tokens],
            role=role,
            entity_type=entity_type,
            token_count=len(tokens),
            unique_token_count=len(set(tokens)),
            noise_hits=noise_hits,
            noise_ratio=round(noise_ratio, 3),
            anchor_like=anchor_like,
            confidence=round(confidence, 3),
        )

    return profiles


# --------------------------
# Pairwise relation scoring
# --------------------------

def _compatibility_score(role_a: str, role_b: str, contract: NodeContract) -> float:
    pair = (role_a, role_b)
    rev = (role_b, role_a)
    if pair in DEFAULT_ROLE_COMPATIBILITY:
        return DEFAULT_ROLE_COMPATIBILITY[pair]
    if rev in DEFAULT_ROLE_COMPATIBILITY:
        return DEFAULT_ROLE_COMPATIBILITY[rev]
    # generic fallback
    if role_a == role_b and role_a != "unknown":
        return 0.50
    if "narrative_context" in pair:
        return 0.25
    return 0.30


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _lexical_overlap(a: PropertyProfile, b: PropertyProfile) -> Tuple[float, List[str]]:
    sa, sb = set(a.normalized_tokens), set(b.normalized_tokens)
    shared = sorted(sa & sb)
    score = len(shared) / len(sa | sb) if sa and sb else 0.0
    return score, shared


def _context_alignment(a: PropertyProfile, b: PropertyProfile, study_ctx_text: str) -> float:
    # How much each property overlaps with study context.
    sa = set(a.normalized_tokens)
    sb = set(b.normalized_tokens)
    sc = set(tokenize(study_ctx_text))
    if not sc:
        return 0.0
    return 0.5 * (len(sa & sc) / max(len(sa | sc), 1)) + 0.5 * (len(sb & sc) / max(len(sb | sc), 1))


def _noise_penalty(a: PropertyProfile, b: PropertyProfile, contract: NodeContract) -> float:
    stop = build_stopword_set(contract.negative_terms)
    shared_noise = len(set(a.normalized_tokens) & set(b.normalized_tokens) & stop)
    noise_ratio = 0.5 * (a.noise_ratio + b.noise_ratio)
    # Penalize records with lots of generic / off-domain text.
    return min(0.6, (noise_ratio * 0.45) + (shared_noise * 0.05))


def score_property_pair(
    a: PropertyProfile,
    b: PropertyProfile,
    contract: NodeContract,
    study_ctx_text: str = "",
    backend: str = "heuristic",
    nli_predictor: Optional[Any] = None,
) -> PairRelation:
    """Score a property pair.

    backend:
      - heuristic: no ML required; robust to noisy mixed records
      - nli: use an NLI predictor (must return entailment/neutral/contradiction probs)
      - hybrid: average heuristic and NLI-based confidence
    """
    lex_score, shared_tokens = _lexical_overlap(a, b)
    role_score = _compatibility_score(a.role, b.role, contract)
    context_score = _context_alignment(a, b, study_ctx_text)
    noise_penalty = _noise_penalty(a, b, contract)

    heuristic_score = (
        0.36 * lex_score
        + 0.34 * role_score
        + 0.20 * context_score
        + 0.10 * min(a.confidence, b.confidence)
        - noise_penalty
    )
    heuristic_score = max(0.0, min(1.0, heuristic_score))

    components = {
        "lexical_overlap": round(lex_score, 3),
        "role_compatibility": round(role_score, 3),
        "context_alignment": round(context_score, 3),
        "min_profile_confidence": round(min(a.confidence, b.confidence), 3),
        "noise_penalty": round(noise_penalty, 3),
        "heuristic_score": round(heuristic_score, 3),
    }

    final_score = heuristic_score
    notes: List[str] = []

    if backend in {"nli", "hybrid"} and nli_predictor is not None:
        # Provide a pairwise declarative prompt. The NLI model then scores agreement.
        premise = f"Property A: {a.text}"
        hypothesis = f"Property B: {b.text}"
        try:
            nli = nli_predictor(premise, hypothesis)
            entail = float(nli.get("entailment", 0.0))
            neutral = float(nli.get("neutral", 0.0))
            contra = float(nli.get("contradiction", 0.0))
            # Convert NLI to a [0,1] relatedness-like score.
            nli_score = max(0.0, entail - contra + 0.5 * neutral)
            nli_score = max(0.0, min(1.0, nli_score))
            components.update({
                "nli_entailment": round(entail, 3),
                "nli_neutral": round(neutral, 3),
                "nli_contradiction": round(contra, 3),
                "nli_score": round(nli_score, 3),
            })
            notes.append("nli_backend_used")

            if backend == "nli":
                final_score = nli_score
            else:
                final_score = 0.6 * heuristic_score + 0.4 * nli_score
        except Exception as exc:  # pragma: no cover
            notes.append(f"nli_failed:{type(exc).__name__}")

    # Map score to labels.
    if final_score >= contract.relation_thresholds.get("complementary", 0.70):
        relation = "complementary"
    elif final_score >= contract.relation_thresholds.get("related", 0.55):
        relation = "related"
    elif final_score <= contract.relation_thresholds.get("unrelated", 0.35):
        relation = "unrelated"
    else:
        relation = "ambiguous"

    return PairRelation(
        property_a=a.key,
        property_b=b.key,
        relation=relation,
        score=round(final_score, 3),
        components=components,
        shared_tokens=shared_tokens,
        notes=notes,
    )


# --------------------------
# Record-level analysis
# --------------------------

def analyze_record(
    record: Mapping[str, Any],
    contract: NodeContract,
    backend: str = "heuristic",
    nli_predictor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Analyze one record and produce property-level relations."""
    profiles = build_property_profiles(record, contract)
    study_ctx = extract_study_context(record, contract)
    study_ctx_text = _flatten_record_text(study_ctx, contract)

    relations: List[PairRelation] = []
    for a_key, b_key in combinations(profiles.keys(), 2):
        rel = score_property_pair(
            profiles[a_key],
            profiles[b_key],
            contract,
            study_ctx_text=study_ctx_text,
            backend=backend,
            nli_predictor=nli_predictor,
        )
        relations.append(rel)

    return {
        "node": contract.node_name,
        "data_mode": contract.data_mode,
        "study_context": study_ctx,
        "profiles": {k: v.__dict__ for k, v in profiles.items()},
        "relations": [r.__dict__ for r in relations],
    }


# --------------------------
# Aggregation across records
# --------------------------

def aggregate_relation_results(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate the same property-pair relation across multiple records.

    Uses a trimmed mean-ish consensus by default and reports consistency.
    """
    buckets: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for res in results:
        for row in res.get("relations", []):
            key = tuple(sorted([row["property_a"], row["property_b"]]))
            buckets[key].append(row)

    out: List[Dict[str, Any]] = []
    for pair, rows in buckets.items():
        scores = sorted(float(r["score"]) for r in rows)
        if len(scores) >= 5:
            trim = max(1, int(len(scores) * 0.2))
            trimmed = scores[trim:-trim] if len(scores) > 2 * trim else scores
        else:
            trimmed = scores

        consensus_score = statistics.mean(trimmed) if trimmed else statistics.mean(scores)
        labels = [r["relation"] for r in rows]
        label_counts = Counter(labels)
        best_label, best_ct = label_counts.most_common(1)[0]
        consistency = best_ct / len(labels)

        out.append({
            "property_a": pair[0],
            "property_b": pair[1],
            "consensus_relation": best_label,
            "consensus_score": round(consensus_score, 3),
            "support": len(rows),
            "consistency": round(consistency, 3),
            "label_support": dict(label_counts),
        })

    return sorted(out, key=lambda x: (x["property_a"], x["property_b"]))


# --------------------------
# Optional NLI backend
# --------------------------

def load_nli_predictor(model_name: str = "lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli"):
    """Return a callable (premise, hypothesis) -> probability dict.

    This is optional. It will only be used if transformers/torch are installed.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise ImportError("transformers and torch are required for NLI backend") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Model labels should be confirmed from config/id2label, but most MNLI checkpoints expose these.
    id2label = getattr(model.config, "id2label", {})

    def predict(premise: str, hypothesis: str) -> Dict[str, float]:
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

        out: Dict[str, float] = {}
        for idx, p in enumerate(probs):
            label = id2label.get(idx, str(idx)).lower()
            out[label] = float(p)

        # Normalize common labels to a canonical output.
        canonical = {
            "entailment": out.get("entailment", out.get("0", 0.0)),
            "neutral": out.get("neutral", out.get("1", 0.0)),
            "contradiction": out.get("contradiction", out.get("2", 0.0)),
        }
        return canonical

    return predict


# --------------------------
# Example contract builder
# --------------------------

def build_human_relevance_contract() -> NodeContract:
    """A minimal contract example for the human_relevance node.

    This is intentionally kept outside the engine logic.
    """
    return NodeContract(
        node_name="human_relevance",
        data_mode="contextual_text",
        pk_field="human_relevance_record_id",
        anchor_fields=["relevant_human_cancer"],
        fk_fields=["human_relevance_record_id"],
        negative_terms=[
            "machine", "learning", "cloud", "computing", "analytics",
            "infrastructure", "behavior", "user", "random", "sandwich"
        ],
        property_roles={
            "relevant_human_cancer": {"role": "anchor_entity", "entity_type": "disease"},
            "relevant_human_genes": {"role": "supporting_entity", "entity_type": "gene"},
            "relevant_human_pathways": {"role": "supporting_entity", "entity_type": "pathway"},
            "relevant_experimental_therapeutic_intervention": {"role": "intervention", "entity_type": "therapy"},
            "human_relevance_statement": {"role": "narrative_context", "entity_type": "narrative"},
        },
        relation_thresholds={
            "related": 0.55,
            "complementary": 0.70,
            "unrelated": 0.35,
        },
    )


if __name__ == "__main__":  # pragma: no cover
    # Small self-test / demo
    demo_record = {
        "relevant_experimental_therapeutic_intervention": '["dietary supplements, yoga therapy, random lifestyle changes"]',
        "relevant_human_pathways": '["neural network optimization, machine learning pipelines, transcriptional regulation by TP53"]',
        "relevant_human_cancer": '["Bladder Cancer"]',
        "human_relevance_record_id": "NEG01",
        "human_relevance_statement": (
            "This study explores how unrelated computational models and lifestyle interventions influence general wellbeing. "
            "While some references are made to bladder cancer and TP53, most of the discussion focuses on algorithm performance, "
            "cloud computing infrastructure, and user behavior analytics. The results are inconclusive and largely disconnected from biological mechanisms."
        ),
        "relevant_human_genes": ["PhotosynthesisGeneX"],
    }

    contract = build_human_relevance_contract()
    result = analyze_record(demo_record, contract)
    print(json.dumps(result["relations"], indent=2))
