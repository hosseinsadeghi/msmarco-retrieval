"""Tokenization for BM25. Kept intentionally small and obvious."""
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# A short, conventional English stopword list. Removing these mostly helps BM25
# because they appear in nearly every document and inflate document length.
_STOPWORDS = frozenset("""
a an the and or but if then else of in on at to for from by with as is are was
were be been being do does did have has had this that these those it its i you
he she we they them his her our their what which who whom whose how why when
where there here not no nor so than too very can will just don should now
""".split())


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return list(tokens)
