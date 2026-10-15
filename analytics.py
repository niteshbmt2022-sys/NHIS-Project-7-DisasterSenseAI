"""Pure, dataset-derived analytics shared by the UI and regression tests."""
import pandas as pd

CLASS_NAMES = {0: 'Non-disaster', 1: 'Disaster-related'}
CLASS_COLORS = {'Non-disaster': '#60A5FA', 'Disaster-related': '#F87171'}

def dataset_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df.target.value_counts()
    missing = int(df[['keyword', 'location']].isna().sum().sum())
    return dict(total=total, disaster=int(counts.get(1, 0)), non_disaster=int(counts.get(0, 0)),
                disaster_share=float(df.target.mean()) if total else 0,
                metadata_missing=missing, metadata_completeness=1-missing/(2*total) if total else 0,
                missing_keyword=int(df.keyword.isna().sum()), missing_location=int(df.location.isna().sum()),
                unique_texts=int(df.text.nunique()), repeated_texts=int(df.text.duplicated().sum()),
                duplicate_rows=int(df.duplicated().sum()))

def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.target.value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame({'Class': [CLASS_NAMES[x] for x in counts.index], 'Tweets': counts.values,
                         'Share': counts.values / len(df) if len(df) else [0., 0.]})

def quality_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({'Column': df.columns, 'Missing': df.isna().sum().values,
                         'Missing share': df.isna().mean().values, 'Type': df.dtypes.astype(str).values})

def filter_records(df, targets, keywords=(), locations=(), query='', missing_label='(Missing)'):
    mask = df.target.isin(targets)
    for col, selections in [('keyword', keywords), ('location', locations)]:
        if selections:
            mask &= df[col].fillna(missing_label).astype(str).isin(selections)
    if query:
        mask &= df.text.str.contains(query, case=False, regex=False, na=False)
    return df.loc[mask].copy()

def batch_metrics(out, status_column):
    target = status_column[:-6]+'predicted_target'
    success = out[status_column].eq('Predicted')
    return dict(total=len(out), success=int(success.sum()), failed=int((~success).sum()),
                disaster=int(out.loc[success, target].eq(1).sum()),
                non_disaster=int(out.loc[success, target].eq(0).sum()))
