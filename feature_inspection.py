"""Inspect the fitted production feature space without refitting or densifying it."""
import numpy as np
import pandas as pd
from src.preprocessing import TweetStructure


def fitted_features(model, texts, branch_name):
    branch = dict(model.named_steps['features'].transformer_list)[branch_name]
    matrix = branch.transform(list(texts))
    names = branch.get_feature_names_out()
    means = np.asarray(matrix.mean(axis=0)).ravel()
    active = np.asarray((matrix != 0).sum(axis=0)).ravel()
    frame = pd.DataFrame({'Feature': names, 'Mean weight': means, 'Active records': active})
    frame = frame[frame['Active records'] > 0].sort_values('Mean weight', ascending=False)
    return frame, matrix.shape, int(matrix.nnz) if hasattr(matrix, 'nnz') else int(np.count_nonzero(matrix))


def structural_frame(texts):
    transformer = TweetStructure()
    return pd.DataFrame(transformer.transform(list(texts)), columns=transformer.get_feature_names_out())
