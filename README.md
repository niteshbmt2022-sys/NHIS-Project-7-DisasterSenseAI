# DisasterSense AI

**NLP Intelligence & Disaster Tweet Analytics**  
Turning Social Signals into Actionable Intelligence.

A working seven-page Streamlit application implementing the supplied academic project: exploration, preparation, feature engineering, model selection, evaluation and deployment. Single and batch predictions use a serialized fitted scikit-learn pipeline. All dashboard metrics derive from the supplied Excel or saved evaluation artifacts.

## Dataset and provenance

The original user files were copied, never modified, into `data/original/`. The sole dataset has **7,613 rows × 5 columns**; the PDF's reference to 10,000 tweets does not match the supplied file. No records were fabricated or obtained from elsewhere. Sheet: `twitter_disaster (1)`.

| Column | Meaning | Missing cells |
|---|---|---:|
| id | Record identifier | 0 |
| keyword | Optional source keyword | 61 |
| location | Optional unverified location context | 2,534 |
| text | Tweet text; sole model input | 0 |
| target | 0 non-disaster; 1 disaster-related | 0 |

Class counts: 4,342 non-disaster and 3,271 disaster-related. No duplicate complete rows; 110 repeated raw tweet texts. Application statistics are recomputed from the loaded dataset rather than copied from this documentation. Missing metadata remains intact. Mixed numeric/string metadata is displayed as text in filters only.

## Installation and Windows commands

Use Python 3.10 or newer. In PowerShell:

```powershell
cd "C:\Users\nites\Documents\Codex\2026-10-15\files-mentioned-by-the-user-project\outputs\DisasterSenseAI"
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The delivered model is already trained. Regenerate it after changing the training file or scikit-learn version:

```powershell
.\.venv\Scripts\python.exe train.py --seed 42
.\.venv\Scripts\python.exe -m pytest tests -q
```

The same suite also runs without pytest: `python -m unittest discover -s tests -v`.
Build verification: **23 tests and seven page subtests passed**. See `VERIFICATION.md` and `artifacts/test_results.xml`. The server reported startup, but localhost browser/health connections timed out in the managed environment; visual browser rendering is unverified.
With the existing verified system environment, start immediately with `python -m streamlit run app.py`.
Open `http://localhost:8501`. Stop the server with Ctrl+C. The default application needs no credentials, paid services, network connection, GPU or model downloads.

## Application workflows

1. **Executive Dashboard:** data and model KPIs, class balance, lengths, missingness, frequent terms and model comparison.
2. **Data Explorer:** class/keyword/location filters, literal text search, column selection, pagination, full text, class-specific words and phrases, hashtags, schema and exports.
3. **Tweet Analyzer:** illustrative examples, clear input, counts, normalized preview, actual prediction, supported native probability and weighted local coefficients.
4. **Model Lab:** explicit training, seed, approximate holdout fraction, three/five-fold grouped CV, tuning, feature variants, timing and comparison. Refreshing never automatically trains.
5. **Prediction Explorer:** CSV/XLSX upload, sheet and text-column selection, invalid-row reporting, search/status filters, pagination and CSV/XLSX export. Ground-truth evaluation requires an explicit checkbox and label-column selection.
6. **Model Performance:** saved holdout confusion matrices, ROC and precision–recall curves, per-class reports, CV/train results, metadata and misclassified holdout records.
7. **Reports & Project Information:** downloadable HTML report, JSON metadata, CSV/XLSX summaries, latest prediction export, objectives, methodology, limitations and references.

To analyze one tweet, open Tweet Analyzer, paste text and choose Analyze Tweet. Probability is shown only for models with `predict_proba`; SVC decision scores are not probabilities. The probability has not been calibration-validated. Weighted feature contributions are associations, not causal explanations or proof of an event.

To classify a file, open Prediction Explorer, upload CSV or XLSX, select the text column, then Run batch prediction. A target column is unnecessary. Original columns remain intact; collision-free `ds_` columns are appended. Empty, numeric-only cells and text without usable normalized tokens are reported as failed rows. Maximum upload 25 MB; maximum 100,000 rows. Download buttons export the current filtered results. Excel/CSV exports neutralize formula-leading text with an apostrophe for spreadsheet safety; original in-memory/source values are preserved.

## NLP methodology and reproducibility

Cleaning lowercases, decodes HTML, expands common negation contractions, removes URLs and mentions, preserves hashtag words and negation, normalizes punctuation and whitespace. Tokenization occurs in TF-IDF. Stop words are retained in the model; the descriptive frequency chart uses a small display-only stop list.

All learned features are fitted inside a Pipeline within each training fold. Word TF-IDF uses unigrams/bigrams, min_df=2, sublinear term frequency and a 20,000-feature cap. The character variant adds char_wb 3–5 grams, min_df=3 and a separate 20,000-feature cap. Structural features add bounded nonnegative length, word, hashtag, mention and exclamation counts from raw input. Their utility is measured as a separate candidate. Sentiment, embeddings and transformers are deferred because incremental value has not been established and they are unnecessary dependencies here.

StratifiedGroupKFold assigns identical normalized texts to the same partition, including conflicting labels. Seed 42, five outer folds for an approximately 20% holdout, then three grouped training CV folds. Groups make exact fractions/class proportions approximate. Actual split: **6,142 training / 1,471 holdout**. CV jointly tunes parameters with all feature fitting restricted to training folds. The best mean CV disaster-class F1 selects the model before holdout evaluation. The common holdout is used only for final reporting; no threshold tuning uses it. Near-duplicate paraphrases and related events can still cross partitions. The production artifact stays fitted on training data, so its evaluation lineage remains explicit.

Candidates: Multinomial Naive Bayes (alpha 0.5/1), Logistic Regression (C 1/3), Linear SVC (C 0.5/1), Random Forest (100 trees, depth 40, leaf size 1/2), Logistic + character (C 1/3), Logistic + structure (C 1/3). Each has the same outer holdout and CV folds. Model Lab reports CV standard deviation, train F1, training/search time and holdout inference time. ROC and AP use probabilities or genuine decision scores. Saved errors support qualitative error inspection.

## Actual initial results

Selected: **Logistic Regression with word + character TF-IDF**.

| Metric | Value |
|---|---:|
| Grouped training CV F1 | 0.7530 |
| Holdout accuracy | 0.8131 |
| Disaster precision | 0.8377 |
| Disaster recall | 0.6793 |
| Disaster F1 | 0.7502 |
| ROC-AUC | 0.8764 |

These are measured results of this build, not metrics borrowed from research references. Full comparisons, curves, split indices, chosen parameters, dataset hash, timestamp and library versions are in `artifacts/evaluation.json`. The `model.joblib` contains the complete selected pipeline and metadata; only trusted local artifacts are loaded. Arbitrary uploaded serialized models are never accepted. Version mismatch, missing or damaged models trigger a retraining message.

## Architecture

```text
app.py                       Streamlit Page/navigation entrypoint
config.py                    Project-relative paths and disclaimer
train.py                     Reproducible training CLI
src/data_loader.py           Schema validation, integrity hash, summaries
src/preprocessing.py         Shared cleaner and structural transformer
src/model_training.py        Feature pipelines, grouped tuning, artifacts
src/evaluation.py            Genuine metrics and curve coordinates
src/prediction.py            Batch prediction and local coefficients
src/reporting.py             Safe CSV/Excel and HTML generation
src/ui.py                    Seven callable page renderers and charts
assets/logo.svg              Original shield/waveform identity
.streamlit/config.toml       Dark theme and upload configuration
data/original/               Unmodified copied source files
artifacts/                   Model, evaluation JSON, holdout errors
tests/                       Core and Streamlit page/workflow tests
```

UI rendering is separate from training/prediction services. Callable pages use Streamlit's supported `st.Page` and `st.navigation` API. Dataset and artifact caches are keyed by file modification time. Expensive training runs only on explicit command/button action. Sparse text features avoid dense vocabulary matrices.

## Limitations and responsible use

This application predicts whether text resembles a disaster-related tweet. Predictions are model outputs, not verified emergency reports. Results require human verification before operational decisions.

Historical English tweets and noisy labels do not establish generalization to other languages, events or current social media. Native probabilities are uncalibrated. False negatives miss disaster language; false positives can create false alarms. Location metadata is not verified geolocation. There are no live feeds, automated emergency alerts or official emergency-service claims. No external-domain or temporal evaluation has been performed. Optional light mode and transformer experiments are not included.

## Troubleshooting and deployment

- Missing imports: install requirements with the same Python executable used to launch Streamlit.
- Missing/damaged/incompatible model: run `python train.py` or train in Model Lab. Loading scikit-learn artifacts requires the recorded version.
- Changed dataset: dashboard warns of source-hash mismatch; retrain before relying on evaluations.
- Invalid training schema/text/labels: restore the original file or correct a separate copy. Metadata missingness is allowed.
- Occupied port: add `--server.port 8502`.
- PowerShell activation restrictions: use `.venv\Scripts\python.exe` directly; activation is unnecessary.
- For optional Streamlit Community Cloud, publish an authorized repository containing app/source/requirements/data, choose app.py, and generate trusted compatible artifacts before startup. Do not commit secrets. Large/private datasets and model files require an appropriate deployment storage strategy. No cloud deployment was performed in this build.

## References

The supplied `data/original/Project statement.pdf` is the academic source. The user's pasted build request determines implementation scope; document instructions were treated as project requirements rather than separate tool authorization.

- https://www.kaggle.com/c/nlp-getting-started
- https://github.com/Prabhath32/Disaster-Tweet-Classification-
- https://github.com/myrazd/disaster-tweet-classification-nlp
- https://huggingface.co/mahmad932/disaster-tweet-classifier
- https://docs.streamlit.io/develop/concepts/multipage-apps/overview
- https://plotly.com/python/getting-started/

Public references inspired classical NLP pipelines and clear evaluation, without copying branding or reported metrics. Streamlit and repository pages were accessible during research; Kaggle returned no readable page body and the Plotly overview fetch exceeded the browsing content limit. No external pretrained weights were downloaded.

## Premium redesign upgrade

The seven pages now share a midnight theme, reusable left-accent KPI cards, SVG icons, responsive grids, consistent class colors and grouped navigation. The dashboard includes source provenance, class balance, keyword and length distributions, missing metadata and genuine saved-model comparisons. Explorer filters preserve original missing values in exports. The analyzer offers actual dataset examples, processing previews, native probabilities, measured request time and local feature explanations. Performance includes saved curves, train/CV/holdout comparisons and error filtering. Batch summaries report successful and failed rows separately.

New presentation modules: `src/analytics.py` (derived statistics), `src/ui_components.py` (escaped reusable components), `assets/theme.css` (responsive styling), and `assets/wordmark.svg` (navigation branding).

The initial results above are historical. At redesign time the existing saved run uses seed 38, 5,705 training rows and 1,908 holdout rows; selected model Logistic + character, holdout accuracy 0.817086 and F1 0.772934. The application reads the current artifact dynamically. No retraining was performed during the redesign.

Hinglish quick start: Project folder mein terminal kholkar `python -m streamlit run app.py` chalayein. Browser mein http://localhost:8501 kholein. Tweet Analyzer mein text daal kar Analyze Tweet dabayein. Prediction Explorer mein CSV/XLSX upload karke text column select karein; results CSV ya Excel mein download kar sakte hain.

## Native themes and feature workbench
Use the top-right Streamlit menu to select System, Light or Dark. Both native theme palettes are supported, including adaptive custom cards and native Plotly styling. If an already-running server does not pick up configuration edits, restart it once and reload the browser. Model Lab now includes fitted word/character TF-IDF inspection, sparse density, per-record weights, exact structural-feature distributions and saved feature-variant comparisons. See THEME_VERIFICATION.md for verification scope and limitations.
