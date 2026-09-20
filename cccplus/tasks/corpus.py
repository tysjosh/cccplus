"""Generic translator-fitting corpus (Sec. 4.3).

"Translator fitting uses a separate 50M-token corpus split by document into 25M-token
forward and reverse halves."

The unit that must not straddle the two legs is the **document**, so this module
always returns explicit document ids alongside the texts, and
``split_documents(..., document_ids=...)`` does the rest. Loaders, in order of
preference:

  wikitext   HuggingFace ``wikitext-103-raw-v1``; blank-line-delimited articles give
             genuine document boundaries
  textfile   a local file, documents separated by blank lines
  synthetic  deterministic filler, so the pipeline is runnable with no network at all
             (clearly marked in the provenance, since it is not natural text)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Corpus:
    texts: List[str]
    document_ids: np.ndarray
    provenance: Dict[str, object]

    def __len__(self) -> int:
        return len(self.texts)

    def subset(self, idx: Sequence[int]) -> "Corpus":
        idx = list(idx)
        return Corpus([self.texts[i] for i in idx], self.document_ids[np.asarray(idx)],
                      dict(self.provenance))


def load_corpus(
    n_passages: int,
    seed: int,
    source: str = "auto",
    path: Optional[str] = None,
    min_chars: int = 200,
    max_chars: int = 1200,
) -> Corpus:
    if source in ("auto", "wikitext"):
        try:
            return _wikitext(n_passages, seed, min_chars, max_chars)
        except Exception as exc:  # noqa: BLE001
            if source == "wikitext":
                raise
            print(f"    [corpus] wikitext unavailable ({exc}); trying next source")
    if source in ("auto", "textfile") and path:
        try:
            return _textfile(path, n_passages, seed, min_chars, max_chars)
        except Exception as exc:  # noqa: BLE001
            if source == "textfile":
                raise
            print(f"    [corpus] text file unavailable ({exc}); falling back to synthetic")
    return _synthetic(n_passages, seed)


def _wikitext(n: int, seed: int, min_chars: int, max_chars: int) -> Corpus:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    texts, doc_ids, buf, doc = [], [], [], 0
    for row in ds:
        line = row["text"]
        if line.strip().startswith("=") and line.strip().endswith("=") and buf:
            joined = " ".join(buf).strip()
            if len(joined) >= min_chars:
                texts.append(joined[:max_chars])
                doc_ids.append(doc)
                doc += 1
            buf = []
            if len(texts) >= n:
                break
        else:
            buf.append(line)
    return Corpus(texts[:n], np.asarray(doc_ids[:n]),
                  {"source": "wikitext-103-raw-v1", "n_documents": int(doc)})


def _textfile(path: str, n: int, seed: int, min_chars: int, max_chars: int) -> Corpus:
    raw = open(path, encoding="utf-8").read()
    docs = [d.strip() for d in raw.split("\n\n") if len(d.strip()) >= min_chars]
    rng = np.random.default_rng(seed)
    rng.shuffle(docs)
    docs = docs[:n]
    return Corpus([d[:max_chars] for d in docs], np.arange(len(docs)),
                  {"source": f"textfile:{path}", "n_documents": len(docs)})


def _synthetic(n: int, seed: int) -> Corpus:
    rng = np.random.default_rng(seed)
    vocab = ("time year people way day man thing woman life child world school state "
             "family student group country problem hand part place case week company "
             "system program question work government number night point home water "
             "room mother area money story month book job word business issue side").split()
    texts, ids = [], []
    for d in range(n):
        n_sent = int(rng.integers(3, 6))
        sents = []
        for _ in range(n_sent):
            k = int(rng.integers(8, 18))
            sents.append(" ".join(rng.choice(vocab, size=k)).capitalize() + ".")
        texts.append(" ".join(sents))
        ids.append(d)
    return Corpus(texts, np.asarray(ids),
                  {"source": "synthetic", "n_documents": n,
                   "warning": "not natural text; translator quality will be understated"})
