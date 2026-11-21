"""Hallucination guard: citation validation + per-sentence NLI entailment."""
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import CFG

CITE_TAG = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]\s*")

_tok = _mdl = None
_ent_idx = None


def _load():
    global _tok, _mdl, _ent_idx
    if _mdl is not None:
        return
    _tok = AutoTokenizer.from_pretrained(CFG.nli_model)
    _mdl = AutoModelForSequenceClassification.from_pretrained(CFG.nli_model).eval()
    labels = {k: v.lower() for k, v in _mdl.config.id2label.items()}
    _ent_idx = next(k for k, v in labels.items() if v.startswith("entail"))


@torch.no_grad()
def entail(premise: str, hypothesis: str) -> float:
    _load()
    clean_hyp = CITE_TAG.sub(" ", hypothesis).strip()
    inputs = _tok(
        premise[:1800],
        clean_hyp,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    probs = torch.softmax(_mdl(**inputs).logits, dim=-1)[0]
    return float(probs[_ent_idx])


def check(answer_sentences, chunks, threshold: float | None = None):
    """Returns (kept_sentences, dropped_sentences, stats)."""
    thr = threshold if threshold is not None else min(CFG.entail_threshold, 0.35)
    n = len(chunks)
    kept, dropped = [], []

    for sent in answer_sentences:
        text = sent["text"]
        cites = [c for c in sent["citations"] if 1 <= c <= n]

        if not cites:
            premise = chunks[0]["text"] if chunks else ""
        else:
            premise = "\n".join(chunks[c - 1]["text"] for c in cites)

        score = entail(premise, text)
        entry = {"text": text, "citations": cites, "entail": round(score, 3)}

        if score >= thr and cites:
            kept.append(entry)
        else:
            dropped.append(entry)

    total = max(len(answer_sentences), 1)
    stats = {
        "total": len(answer_sentences),
        "kept": len(kept),
        "dropped": len(dropped),
        "drop_ratio": len(dropped) / total,
    }
    return kept, dropped, stats


def should_refuse(found: bool, top_dense: float, stats) -> bool:
    if not found:
        return True
    if top_dense < min(CFG.min_dense_score, 0.50):
        return True
    return stats["drop_ratio"] > CFG.max_drop_ratio
