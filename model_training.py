import json
import logging
import platform
import time
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from config import ARTIFACTS, DATA
from src.preprocessing import TextCleaner, TweetStructure, clean_text
from src.data_loader import load_data, fingerprint, summary
from src.evaluation import evaluate

logger = logging.getLogger(__name__)

def pipeline(estimator, character=False, structure=False):
    word = Pipeline([('clean', TextCleaner()), ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=.98, max_features=20000, sublinear_tf=True))])
    branches = [('word', word)]
    if character:
        branches.append(('char', Pipeline([('clean', TextCleaner()), ('tfidf', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=20000, sublinear_tf=True))])))
    if structure:
        branches.append(('structure', TweetStructure()))
    return Pipeline([('features', FeatureUnion(branches)), ('model', estimator)])

def split_data(df, seed=42, test_size=.2):
    groups = df.text.map(clean_text)
    # Grouped stratification prevents repeated normalized texts crossing partitions.
    folds = max(2, round(1/test_size))
    train, test = next(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(df.text, df.target, groups))
    return train, test, groups

def train(df, seed=42, test_size=.2, folds=3, progress=None, artifact_dir=ARTIFACTS, source=DATA):
    train_idx, test_idx, groups = split_data(df, seed, test_size)
    X, y = df.text.iloc[train_idx], df.target.iloc[train_idx]
    candidates = [
        ('Multinomial Naive Bayes', pipeline(MultinomialNB()), {'model__alpha': [.5, 1.0]}),
        ('Logistic Regression', pipeline(LogisticRegression(max_iter=1500, random_state=seed)), {'model__C': [1, 3]}),
        ('Linear SVC', pipeline(LinearSVC(random_state=seed)), {'model__C': [.5, 1]}),
        ('Random Forest', pipeline(RandomForestClassifier(n_estimators=100, max_depth=40, n_jobs=2, random_state=seed)), {'model__min_samples_leaf': [1, 2]}),
        ('Logistic + character', pipeline(LogisticRegression(max_iter=1500, random_state=seed), character=True), {'model__C': [1, 3]}),
        ('Logistic + structure', pipeline(LogisticRegression(max_iter=1500, random_state=seed), structure=True), {'model__C': [1, 3]}),
    ]
    models, results = {}, {}
    for i, (name, estimator, grid) in enumerate(candidates):
        logger.info('Training candidate %s', name)
        if progress:
            progress(i / len(candidates), f'Training {name} ({i+1}/{len(candidates)})')
        start = time.perf_counter()
        search = GridSearchCV(estimator, grid, scoring='f1', cv=StratifiedGroupKFold(folds, shuffle=True, random_state=seed),
                              n_jobs=1, return_train_score=True, error_score='raise')
        search.fit(X, y, groups=groups.iloc[train_idx])
        duration = time.perf_counter() - start
        models[name] = search.best_estimator_
        results[name] = {'cv_f1': float(search.best_score_), 'cv_std': float(search.cv_results_['std_test_score'][search.best_index_]),
                         'cv_train_f1': float(search.cv_results_['mean_train_score'][search.best_index_]),
                         'best_params': search.best_params_, 'training_seconds': duration,
                         'cv_candidates': pd.DataFrame(search.cv_results_)[['params','mean_test_score','std_test_score','mean_train_score']].to_dict('records')}
    selected = max(results, key=lambda name: results[name]['cv_f1'])
    logger.info('Selected %s using training-only CV', selected)
    errors = None
    for name, model in models.items():
        metrics, pred = evaluate(model, df.text.iloc[test_idx], df.target.iloc[test_idx])
        results[name].update(metrics)
        results[name]['train_f1'] = float(__import__('sklearn.metrics', fromlist=['f1_score']).f1_score(y, model.predict(X)))
        if name == selected:
            errors = df.iloc[test_idx].copy()
            errors['predicted_target'] = pred
            errors = errors[errors.target != errors.predicted_target]
    metadata = {'artifact_version': 1, 'selected_model': selected, 'trained_at': datetime.now(timezone.utc).isoformat(),
                'selection_metric': 'Highest grouped cross-validation disaster-class F1 on training data only; holdout not used for selection.',
                'seed': seed, 'requested_test_size': test_size, 'actual_test_size': len(test_idx)/len(df), 'cv_folds': folds,
                'train_rows': len(train_idx), 'test_rows': len(test_idx), 'source_sha256': fingerprint(source),
                'dataset': summary(df), 'python': platform.python_version(), 'sklearn': sklearn.__version__,
                'pandas': pd.__version__, 'numpy': np.__version__, 'features': 'Word TF-IDF (1,2), optional char_wb (3,5) or bounded structural features',
                'train_indices': train_idx.tolist(), 'test_indices': test_idx.tolist(), 'results': results}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Save only trusted locally trained model objects.
    joblib.dump({'pipeline': models[selected], 'metadata': metadata}, artifact_dir / 'model.joblib')
    (artifact_dir / 'evaluation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    errors.to_csv(artifact_dir / 'holdout_errors.csv', index=False)
    if progress:
        progress(1.0, 'Training and evaluation complete')
    return metadata

def load_artifact(path=ARTIFACTS / 'model.joblib'):
    if not path.exists():
        raise ValueError('Model artifact missing. Train it in Model Lab or run python train.py.')
    try:
        bundle = joblib.load(path)
        meta = bundle['metadata']
        if meta['artifact_version'] != 1 or meta['sklearn'] != sklearn.__version__:
            raise ValueError('Version mismatch')
        if not hasattr(bundle['pipeline'], 'predict'):
            raise ValueError('Invalid pipeline')
        return bundle
    except Exception as exc:
        raise ValueError('Model artifact incompatible or damaged. Regenerate with python train.py.') from exc
