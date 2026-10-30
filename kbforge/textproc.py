from __future__ import annotations

import re
import string
import unicodedata

# Simple English stop words (no external dependency)
DEFAULT_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
    "or", "but", "if", "this", "they", "we", "you", "your", "all", "can", "had", "her",
    "his", "not", "so", "up", "out", "about", "what", "when", "where", "who", "which",
    "why", "how", "do", "does", "did", "been", "have", "having", "am", "could", "would",
    "should", "may", "might", "must", "their", "there", "here", "than", "then", "over",
    "under", "after", "before"
}


class TextProcessor:
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True,
                 remove_stopwords: bool = False, stem: bool = False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.stop_words = DEFAULT_STOP_WORDS

    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()
        return text

    def tokenize(self, text: str) -> list[str]:
        text = self.normalize(text)
        if self.remove_punctuation:
            # Replace punctuation with spaces
            text = re.sub(r'[^\w\s]', ' ', text)
        # Split on whitespace
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.stem:
            # Very simple Porter-like stemmer (minimal, no external deps)
            tokens = [self._simple_stem(t) for t in tokens]
        return tokens

    def _simple_stem(self, word: str) -> str:
        if len(word) <= 3:
            return word
        # Very basic rules
        for suffix in ['ing', 'ed', 'er', 'est', 'ly']:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)]
                break
        if word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
            word = word[:-1]
        return word


# Default processor - consistent for docs and queries
default_processor = TextProcessor(lowercase=True, remove_punctuation=True, remove_stopwords=False, stem=False)


def tokenize(text: str) -> list[str]:
    return default_processor.tokenize(text)


def tokenize_query(text: str) -> list[str]:
    return default_processor.tokenize(text)
