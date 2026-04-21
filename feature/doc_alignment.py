from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from schema import NodeSchema

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class DocAlignmentModel:
    vectorizer: TfidfVectorizer | None = None
    matrix: np.ndarray | None = None
    names: list[str] | None = None
    texts: dict[str, str] | None = None

    def fit(self, node_schema: NodeSchema) -> "DocAlignmentModel":
        self.names = node_schema.property_names()
        self.texts = node_schema.property_texts()
        corpus = [self.texts[name] for name in self.names]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(corpus)
        return self

    # compute cosine similarity 
    def score(self, a: str, b: str) -> float:
        if self.matrix is None or self.names is None or self.texts is None:
            return 0.0
        if a not in self.texts or b not in self.texts:
            return 0.0

        i = self.names.index(a)
        j = self.names.index(b)
        cosine = float((self.matrix[i] @ self.matrix[j].T).toarray()[0][0])
        boost = self._explicit_boost(a, b)
        return float(min(1.0, cosine + boost))

    # One property name appears in the other’s description
    # They share tokens in their names
    # Descriptions contain phrases like:
    # “derived from”
    # “combination of”
    # etc.
    def _explicit_boost(self, a: str, b: str) -> float:
        assert self.texts is not None
        text_a = self.texts[a].lower()
        text_b = self.texts[b].lower()
        a_tokens = set(_WORD_RE.findall(a.lower()))
        b_tokens = set(_WORD_RE.findall(b.lower()))

        boost = 0.0
        if a.lower() in text_b:
            boost += 0.25
        if b.lower() in text_a:
            boost += 0.25

        shared = a_tokens & b_tokens
        if shared:
            boost += min(0.10, 0.03 * len(shared))

        if any(phrase in text_b for phrase in ("combination of", "based upon", "derived from", "generated during data transformation")):
            # give a stronger boost if the parent property is explicitly mentioned in the child description
            if a.lower() in text_b:
                boost += 0.25

        return min(0.5, boost)
