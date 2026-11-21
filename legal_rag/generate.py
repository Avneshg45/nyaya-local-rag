"""Quantized local LLM (llama.cpp) -> fast, well-formatted Markdown answer with citations."""
import re
import threading

from .config import CFG
from .models import get_llm

SYSTEM = """You are a precise study assistant. Answer the QUESTION using ONLY the provided CONTEXT.
Rules:
1. Answer in the exact same language as the QUESTION.
2. Answer only what the QUESTION asks. Do not add types, categories, examples, or extra topics unless the QUESTION explicitly asks for them.
3. Start with a direct 1-2 sentence answer, then add only 2-3 short bullet points that are directly relevant.
4. Do not repeat or pull in unrelated information from the context.
5. End every sentence or bullet point with a passage citation like [1].
6. If the CONTEXT does not contain the answer, reply ONLY with: NOT_FOUND"""

_lock = threading.Lock()


def build_context(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] ({c['title']}, page {c['page'] + 1})\n{c['text']}")
    return "\n\n".join(parts)


def gloss_to_english(query: str) -> str:
    return ""


def paraphrase(query: str) -> list[str]:
    return []


def generate(query: str, chunks):
    top_chunks = chunks[:CFG.final_k]
    context = build_context(top_chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nWrite a clear, well-formatted answer (1 short intro + 2-3 bullets) strictly from CONTEXT:"
    with _lock:
        raw = get_llm().create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=260,
            temperature=0.0,
            top_p=0.9,
            repeat_penalty=1.08,
        )["choices"][0]["message"]["content"].strip()

    if not raw or "NOT_FOUND" in raw:
        return {"found": False, "answer": []}

    items = []
    for line in raw.splitlines():
        clean = line.strip()
        if not clean:
            continue
        cites = [int(m) for m in re.findall(r"\[(\d+)\]", clean) if 1 <= int(m) <= len(top_chunks)]
        if not cites:
            cites = [1]
            clean = f"{clean} [1]"
        items.append({"text": clean, "citations": cites})

    return {"found": True, "answer": items}
