import html
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = html.unescape(text).lower().replace('’', "'")
    text = re.sub(r"\bcan't\b", 'cannot', text)
    text = re.sub(r"n't\b", ' not', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', text)).strip()

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [clean_text(t) for t in X]
    def get_feature_names_out(self, input_features=None):
        return input_features

class TweetStructure(BaseEstimator, TransformerMixin):
    """Nonnegative, bounded raw-text structural features; no corpus fitting."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([[min(len(t)/280, 3), min(len(t.split())/50, 3),
                          min(t.count('#')/5, 3), min(t.count('@')/5, 3),
                          min(t.count('!')/5, 3)] for t in X])
    def get_feature_names_out(self, input_features=None):
        return np.array(['length', 'words', 'hashtags', 'mentions', 'exclamations'])
