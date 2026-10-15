import pandas as pd
from src.preprocessing import clean_text

def predict_batch(model, frame, text_column):
    if text_column not in frame.columns or frame.empty:
        raise ValueError('Select an existing text column in a nonempty file.')
    out = frame.copy()
    valid = out[text_column].map(lambda t: isinstance(t, str) and bool(clean_text(t)))
    # Use a collision-free namespace so even previous prediction columns are preserved.
    prefix = 'ds_'
    while any(str(c).startswith(prefix) for c in out.columns):
        prefix = '_' + prefix
    out[prefix+'status'] = 'Empty, non-text, or no usable tokens'
    out[prefix+'predicted_target'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    out[prefix+'label'] = ''
    if valid.any():
        texts = out.loc[valid, text_column]
        labels = model.predict(texts)
        out.loc[valid, prefix+'predicted_target'] = labels
        out.loc[valid, prefix+'label'] = ['Disaster-related' if x else 'Non-disaster' for x in labels]
        out.loc[valid, prefix+'status'] = 'Predicted'
        if hasattr(model, 'predict_proba'):
            out.loc[valid, prefix+'disaster_probability'] = model.predict_proba(texts)[:, 1]
    return out

def explain(model, text):
    estimator = model.named_steps['model']
    if not hasattr(estimator, 'coef_'):
        return None
    features = model.named_steps['features']
    names = features.get_feature_names_out()
    values = features.transform([text])
    contribution = values.multiply(estimator.coef_[0]).toarray()[0]
    positions = sorted((i for i, v in enumerate(contribution) if v), key=lambda i: abs(contribution[i]), reverse=True)[:12]
    return pd.DataFrame({'Feature': names[positions], 'Contribution toward disaster': contribution[positions]})
