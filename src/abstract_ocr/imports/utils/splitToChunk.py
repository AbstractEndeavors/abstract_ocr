import re
from typing import List, Iterable, Literal, Optional

Unit = Literal["char", "word", "sentence"]

_SENTENCE_RX = re.compile(
    r"""
    # sentence-ish chunks with common punctuation
    (?:
        [^\n.!?]+
        (?:
            [.!?](?:"|'|\)|\]|\})?
        )
    )
    """,
    re.VERBOSE,
)

def _split_sentences(text: str) -> List[str]:
    # split by regex, but also capture trailing leftovers (no final punctuation)
    parts = _SENTENCE_RX.findall(text)
    tail = text[len("".join(parts)):]
    if tail.strip():
        parts.append(tail)
    # normalize whitespace around chunks
    return [re.sub(r"\s+", " ", s).strip() for s in parts if s.strip()]

def _window(seq: List[str], size: int, step: int) -> Iterable[List[str]]:
    for i in range(0, max(len(seq) - size, 0) + 1, step):
        yield seq[i:i + size]

def split_to_chunk(
    text: str,
    chunk_size: int = 1000,
    *,
    unit: Unit = "char",
    overlap: int = 0,
    hard_wrap: bool = False,
    # when unit="sentence", we can pack multiple sentences until we reach approx chunk_size chars
    approx_by_chars: bool = True,
    min_chunk_size: Optional[int] = None,
) -> List[str]:
    """
    Split text into chunks.

    Args:
        text: input string.
        chunk_size: size per chunk in chosen unit.
        unit: "char" | "word" | "sentence".
        overlap: number of units to overlap between consecutive chunks (0 = none).
        hard_wrap: if True, never exceed chunk_size in the chosen unit; if False, allow slight overflow
                   to avoid splitting in the middle of a sentence/word pack.
        approx_by_chars: when unit="sentence", pack sentences until ~chunk_size characters.
        min_chunk_size: if set, small trailing chunks are merged back into the previous chunk when possible.

    Returns:
        List[str]: chunks.
    """
    text = text or ""
    text = text.strip()
    if not text:
        return []

    # prepare sequence of units
    if unit == "char":
        seq = list(text)
    elif unit == "word":
        seq = re.findall(r"\S+\s*", text)  # keep following spaces with words
    elif unit == "sentence":
        if approx_by_chars:
            # we’ll pack sentences into ~chunk_size chars below
            sentences = _split_sentences(text)
            chunks: List[str] = []
            cur: List[str] = []
            cur_len = 0
            for s in sentences:
                s_len = len(s)
                if cur and (cur_len + 1 + s_len > chunk_size):
                    # flush current
                    chunks.append(" ".join(cur).strip())
                    # start next, with optional soft overlap (by sentence count)
                    if overlap > 0:
                        # take last `overlap` sentences into next chunk start
                        prev_tail = cur[-overlap:] if overlap < len(cur) else cur
                        chunks[-1] = " ".join(cur).strip()
                        cur = list(prev_tail)
                        cur_len = len(" ".join(cur))
                    else:
                        cur = []
                        cur_len = 0
                # add sentence
                cur.append(s)
                cur_len = len(" ".join(cur))
            if cur:
                chunks.append(" ".join(cur).strip())

            # optional min-chunk merge
            if min_chunk_size is not None and len(chunks) > 1 and len(chunks[-1]) < min_chunk_size:
                chunks[-2] = (chunks[-2] + " " + chunks[-1]).strip()
                chunks.pop()
            return chunks
        else:
            # treat sentences as atomic units and window them
            seq = [s + " " for s in _split_sentences(text)]
    else:
        raise ValueError(f"unknown unit: {unit}")

    # compute step based on overlap
    step = max(chunk_size - overlap, 1)

    # window over the sequence
    raw_chunks = []
    for block in _window(seq, chunk_size, step):
        if unit == "char":
            s = "".join(block)
        else:
            s = "".join(block).strip()
        raw_chunks.append(s)

    # handle trailing remainder if hard_wrap=False and we missed some tail
    if unit in {"char", "word"}:
        consumed = step * max(0, len(raw_chunks) - 1) + (len(seq) if raw_chunks else 0 and chunk_size)
        # safer tail detection
        last_end = step * (len(raw_chunks) - 1) + chunk_size if raw_chunks else 0
        if last_end < len(seq):
            tail = seq[last_end:]
            tail_s = "".join(tail) if unit == "char" else "".join(tail).strip()
            if tail_s:
                if hard_wrap or not raw_chunks:
                    raw_chunks.append(tail_s)
                else:
                    # append to last chunk (soft overflow)
                    raw_chunks[-1] = (raw_chunks[-1] + ("" if unit == "char" else " ") + tail_s).strip()

    # optional min_chunk merge (chars/words)
    if min_chunk_size is not None and len(raw_chunks) > 1 and len(raw_chunks[-1]) < min_chunk_size:
        raw_chunks[-2] = (raw_chunks[-2] + " " + raw_chunks[-1]).strip()
        raw_chunks.pop()

    return [c for c in raw_chunks if c.strip()]
