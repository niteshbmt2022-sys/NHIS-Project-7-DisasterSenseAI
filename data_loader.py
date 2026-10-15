import hashlib
from pathlib import Path
import pandas as pd

def load_data(source):
    """Validate the original five-column labeled dataset without dropping metadata."""
    df = pd.read_excel(source)
    required = {'id', 'keyword', 'location', 'text', 'target'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    if df.empty or df.text.isna().any() or not df.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Training text must contain nonempty strings.')
    if df.target.isna().any() or not df.target.isin([0, 1]).all():
        raise ValueError('target must contain only numeric 0 and 1.')
    if df.target.nunique() != 2:
        raise ValueError('Training requires both classes.')
    df['target'] = df.target.astype(int)
    return df

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def summary(df):
    return {'rows': len(df), 'columns': len(df.columns), 'dtypes': df.dtypes.astype(str).to_dict(),
            'missing': df.isna().sum().astype(int).to_dict(), 'duplicates': int(df.duplicated().sum()),
            'duplicate_texts': int(df.text.duplicated().sum()),
            'class_counts': {str(k): int(v) for k, v in df.target.value_counts().items()}}
