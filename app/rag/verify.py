"""Claim verification against cited passages (docs/DATA_AND_RAG.md section 7).

Primary: DeBERTa NLI (transformers) when installed and NLI_ENABLED=true.
Fallback: lexical support check, clearly labelled method="lexical" in every result.
"""

from __future__ import annotations

import logging
import threading

from ..config import get_settings
from ..knowledge import get_knowledge
from ..schemas import Claim, VerifiedClaim
from .text import best_sentence, normalize, overlap

log = logging.getLogger(__name__)

LEXICAL_ENTAIL = 0.6
LEXICAL_NEUTRAL = 0.3


# The NLI model (~370 MB) is downloaded and loaded in a background thread at startup. Until it is
# ready, claims are checked lexically, so the first debate never waits on (or freezes for) the model.
_pipe = None
_state = "idle"  # idle | loading | ready | failed | disabled
_lock = threading.Lock()


def _load() -> None:  # pragma: no cover - heavy optional dependency
    global _pipe, _state
    try:
        from transformers import pipeline  # type: ignore

        _pipe = pipeline("text-classification", model=get_settings().nli_model, top_k=None)
        _state = "ready"
        log.info("NLI model ready")
    except Exception as exc:  # noqa: BLE001
        _state = "failed"
        log.warning("NLI model unavailable, using lexical verification: %s", exc)


def warm_up() -> None:
    """Start loading the NLI model without blocking. Safe to call more than once."""
    global _state
    with _lock:
        if _state != "idle":
            return
        if not get_settings().nli_enabled:
            _state = "disabled"
            return
        _state = "loading"
    threading.Thread(target=_load, name="nli-warmup", daemon=True).start()


def nli_status() -> str:
    return _state


def _nli(premise: str, hypothesis: str) -> tuple[str, float] | None:  # pragma: no cover
    warm_up()
    pipe = _pipe
    if pipe is None:
        return None
    out = pipe({"text": premise, "text_pair": hypothesis})
    scores = out[0] if isinstance(out[0], list) else out
    best = max(scores, key=lambda s: s["score"])
    return best["label"].lower(), float(best["score"])


def _is_verbatim_quote(claim: str, passage: str) -> bool:
    import re

    quotes = re.findall(r"[\"“]([^\"”]{12,})[\"”]", claim)
    flat = normalize(" ".join(passage.split())).replace("_", "")
    return bool(quotes) and all(normalize(q.strip(" …")).replace("…", "") in flat for q in quotes)


def verify_claim(claim: Claim) -> VerifiedClaim:
    knowledge = get_knowledge()
    passages = [c.text for c in (knowledge.chunk(s) for s in claim.source_ids) if c is not None]
    if claim.inferred or not passages:
        return VerifiedClaim(
            claim_id=claim.claim_id, text=claim.text, status="inferred", source_ids=claim.source_ids,
            nli_label="neutral" if passages else None, nli_prob=None, method="lexical",
        )

    premise = "\n".join(passages)
    support = best_sentence(premise, claim.text)
    threshold = get_settings().nli_threshold

    nli = _nli(premise, claim.text)
    if nli is not None:  # pragma: no cover
        label, prob = nli
        status = "verified" if label == "entailment" and prob >= threshold else "inferred" if label == "neutral" else "dropped"
        return VerifiedClaim(claim_id=claim.claim_id, text=claim.text, status=status, source_ids=claim.source_ids,
                             nli_label=label, nli_prob=round(prob, 3), support=support, method="nli")

    if _is_verbatim_quote(claim.text, premise):
        prob = 1.0
    else:
        prob = round(max(overlap(claim.text, premise), overlap(claim.text, support)), 3)
    if prob >= LEXICAL_ENTAIL:
        label, status = "entailment", "verified"
    elif prob >= LEXICAL_NEUTRAL:
        label, status = "neutral", "inferred"
    else:
        label, status = "contradiction", "dropped"
    return VerifiedClaim(claim_id=claim.claim_id, text=claim.text, status=status, source_ids=claim.source_ids,
                         nli_label=label, nli_prob=prob, support=support, method="lexical")


def accuracy(verified: list[VerifiedClaim]) -> dict[str, float]:
    kept = [v for v in verified if v.status != "dropped"]
    if not kept:
        return {"verified": 0.0, "inferred": 0.0}
    n = len(kept)
    return {
        "verified": round(sum(v.status == "verified" for v in kept) / n, 2),
        "inferred": round(sum(v.status == "inferred" for v in kept) / n, 2),
    }
