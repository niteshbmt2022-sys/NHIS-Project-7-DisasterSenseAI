import json
import math
import re
import time
import hashlib
from datetime import datetime, timezone, timedelta
from collections import Counter
from io import BytesIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import DATA, ARTIFACTS, DISCLAIMER, ROOT
from src.data_loader import load_data, summary, fingerprint
from src.preprocessing import clean_text
from src.model_training import load_artifact, train
from src.prediction import predict_batch, explain
from src.reporting import csv_bytes, excel_bytes, html_report
from src.analytics import dataset_metrics, class_distribution, quality_frame, filter_records, batch_metrics, CLASS_COLORS
from src.ui_components import install_styles, header, section, status, kpis, panel, result_card, insight

COLORS = ['#2DD4BF', '#60A5FA', '#FBBF24', '#C4B5FD', '#F87171', '#F472B6']

@st.cache_data
def dataset(stamp):
    return load_data(DATA)

@st.cache_resource
def artifact(stamp):
    return load_artifact()

def context():
    install_styles()
    try:
        df = dataset(DATA.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f'Dataset unavailable: {exc}. Restore data/original/twitter_disaster.xlsx.'); st.stop()
    try:
        bundle = artifact((ARTIFACTS / 'model.joblib').stat().st_mtime_ns)
        if bundle['metadata']['source_sha256'] != fingerprint(DATA):
            st.warning('Dataset changed since training. Retrain in Model Lab before interpreting these evaluations.')
    except (ValueError, OSError) as exc:
        bundle = None
        st.info(f'Model artifact missing or incompatible. Train in Model Lab. {exc}')
    return df, bundle

def chart(fig, height=350):
    # Native frontend theming updates immediately when appearance changes.
    fig.update_layout(template='streamlit',
                      font=dict(family='Inter, Segoe UI, sans-serif', size=12),
                      colorway=COLORS, margin=dict(l=12,r=20,t=45,b=35), height=height,
                      legend=dict(orientation='h',y=-.25,x=0,font_size=10))
    fig.update_xaxes(zeroline=False,automargin=True)
    fig.update_yaxes(zeroline=False,automargin=True)
    st.plotly_chart(fig, theme='streamlit', width='stretch',config={'displaylogo':False,'scrollZoom':False})


def exports(df, name):
    a, b = st.columns(2)
    a.download_button('Download CSV', csv_bytes(df), f'{name}.csv', 'text/csv', key=f'{name}_csv')
    b.download_button('Download Excel', excel_bytes(df), f'{name}.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key=f'{name}_xlsx')

def table(df, key):
    size = st.selectbox('Rows per page', [20, 50, 100], key=f'{key}_size')
    pages = max(1, math.ceil(len(df)/size))
    if st.session_state.get(f'{key}_page',1)>pages: st.session_state[f'{key}_page']=1
    page = st.number_input(f'Page (1–{pages})', 1, pages, 1, key=f'{key}_page')
    st.caption(f'{len(df):,} matching records')
    if df.empty: st.info('No records match these filters. Broaden your selection or clear the search.')
    else: st.dataframe(df.iloc[(page-1)*size:page*size], width='stretch', hide_index=True)

def comparison(meta):
    return pd.DataFrame([{'Model': name, **{k:r.get(k) for k in ['cv_f1','cv_std','accuracy','precision','recall','f1','roc_auc','average_precision','train_f1','training_seconds','inference_seconds']}} for name,r in meta['results'].items()])

def frequencies(texts, n=1):
    stop = {'the','a','an','to','of','and','in','is','it','for','on','i','you','my','with','that','at','this','be','are','was','have','from','as','by','s','t','http','https','co','amp'}
    counts = Counter()
    for text in texts:
        tokens = [x for x in clean_text(text).split() if x not in stop and len(x)>1]
        counts.update(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return pd.DataFrame(counts.most_common(20), columns=['Term', 'Count'])

def overview():
    df,bundle=context()
    header('Disaster Intelligence Overview','Explore historical social signals with reproducible NLP classification and transparent evaluation.')
    loaded=datetime.fromtimestamp(DATA.stat().st_mtime,timezone(timedelta(hours=5,minutes=30))).strftime('%d %b %Y · %H:%M IST')
    status('Source loaded','twitter_disaster.xlsx · Local historical dataset · File modified '+loaded)
    m=dataset_metrics(df)
    kpis([
        dict(title='TOTAL TWEETS',value=f"{m['total']:,}",detail='Records analyzed',symbol='database'),
        dict(title='DISASTER-RELATED',value=f"{m['disaster']:,}",detail=f"{m['disaster_share']:.1%} of labeled records",tone='red',symbol='alert'),
        dict(title='NON-DISASTER',value=f"{m['non_disaster']:,}",detail=f"{1-m['disaster_share']:.1%} of labeled records",tone='blue',symbol='check'),
        dict(title='DISASTER SHARE',value=f"{m['disaster_share']:.1%}",detail='Share of classified tweets',symbol='chart'),
        dict(title='DATA QUALITY',value=f"{m['metadata_completeness']:.1%}",detail=f"Keyword + location completeness · {m['metadata_missing']:,} missing cells",tone='amber',symbol='shield')])
    section('The classification landscape','Two labeled classes. One historical dataset. Counts describe tweets, not verified incidents.')
    dist=class_distribution(df)
    left,right=st.columns([1.6,1])
    with left, panel('Disaster vs. non-disaster','Actual target labels from the supplied Excel'):
        fig=px.bar(dist,x='Tweets',y='Class',color='Class',orientation='h',text='Tweets',color_discrete_map=CLASS_COLORS,custom_data=['Share'])
        fig.update_traces(texttemplate='%{text:,}',textposition='outside',cliponaxis=False,hovertemplate='%{y}<br>%{x:,} tweets<br>Share: %{customdata[0]:.1%}<extra></extra>')
        fig.update_layout(showlegend=False,xaxis_range=[0,dist.Tweets.max()*1.2],yaxis_title=None,xaxis_title='Labeled tweets')
        chart(fig,300)
    with right, panel('Share of the conversation','Disaster-related share of all labeled records'):
        fig=px.pie(dist,names='Class',values='Tweets',color='Class',color_discrete_map=CLASS_COLORS,hole=.76)
        fig.update_traces(textinfo='none',hovertemplate='%{label}<br>%{value:,} tweets · %{percent}<extra></extra>')
        fig.add_annotation(text=f"<b>{m['disaster_share']:.1%}</b><br>disaster-related",x=.5,y=.5,showarrow=False,font_size=21)
        chart(fig,300)
    section('Language & context','Explore the words and text structures behind the labels.')
    left,right=st.columns([1.15,1.45])
    with left, panel('Most frequent keywords','Available metadata only; missing keywords are excluded'):
        words=df.keyword.dropna().astype(str).value_counts().head(10).rename_axis('Keyword').reset_index(name='Tweets')
        fig=px.bar(words,x='Tweets',y='Keyword',orientation='h',color_discrete_sequence=['#2DD4BF'])
        fig.update_layout(yaxis=dict(autorange='reversed'),yaxis_title=None)
        chart(fig,370)
    with right, panel('Tweet length patterns','Character counts grouped by actual class'):
        lengths=df.assign(Characters=df.text.str.len(),Class=df.target.map({0:'Non-disaster',1:'Disaster-related'}))
        fig=px.histogram(lengths,x='Characters',color='Class',nbins=35,barmode='overlay',opacity=.75,color_discrete_map=CLASS_COLORS)
        chart(fig,370)
    left,right=st.columns([1.3,1])
    with left, panel('Data quality audit','Missing cells by column · Metadata gaps are retained'):
        quality=quality_frame(df)
        fig=px.bar(quality,x='Missing',y='Column',orientation='h',text='Missing',color_discrete_sequence=['#FBBF24'],custom_data=['Missing share'])
        fig.update_traces(hovertemplate='%{y}: %{x:,} missing<br>%{customdata[0]:.1%} of records<extra></extra>')
        chart(fig,300)
    with right, panel('What the data tells us','Computed from the current loaded dataset'):
        insight('Metadata coverage',f"{m['missing_location']:,} records ({m['missing_location']/len(df):.1%}) have no location. Location values are unverified context.",'amber')
        insight('Repeated language',f"{m['repeated_texts']:,} repeated raw texts; {m['unique_texts']:,} distinct texts. Matching normalized texts stay together during model splitting.")
        insight('Text length',f"Median tweet length is {df.text.str.len().median():.0f} characters. {df.text.str.contains('#',regex=False).sum():,} records contain a hashtag.",'blue')
    section('Model intelligence','Verified saved evaluation · Production model chosen by training-only grouped cross-validation')
    if bundle:
        meta=bundle['metadata']; result=meta['results'][meta['selected_model']]
        status('Trained and evaluated',f"{meta['selected_model']} · {meta['test_rows']:,} holdout records · Seed {meta['seed']}")
        model_cards(meta,result)
        data=comparison(meta).melt(id_vars='Model',value_vars=['cv_f1','f1','recall'],var_name='Evaluation',value_name='Score').dropna()
        data['Evaluation']=data.Evaluation.map({'cv_f1':'Training CV F1','f1':'Holdout F1','recall':'Holdout recall'})
        fig=px.bar(data,x='Score',y='Model',color='Evaluation',barmode='group',orientation='h',color_discrete_sequence=COLORS)
        fig.update_xaxes(range=[0,1],tickformat='.0%'); chart(fig,470)
        with st.expander('Full model comparison'): st.dataframe(comparison(meta),hide_index=True)
    else: status('Model not available','Train explicitly in Model Lab to generate saved evaluation results.','amber')

def model_cards(meta,result):
    items=[dict(title='SELECTED MODEL',value=meta['selected_model'],detail='Highest training CV F1',symbol='model')]
    for key,title in [('accuracy','HOLDOUT ACCURACY'),('precision','DISASTER PRECISION'),('recall','DISASTER RECALL'),('f1','DISASTER F1')]:
        if result.get(key) is not None:
            items.append(dict(title=title,value=f"{result[key]:.1%}",detail='Saved holdout evaluation',tone='amber' if key=='recall' else 'teal',symbol='chart'))
    kpis(items)


def explorer():
    df,_=context(); header('Data Explorer','An analytical workspace for labels, language, metadata and data quality.')
    m=dataset_metrics(df)
    kpis([dict(title='DATASET',value=f"{len(df):,} × {len(df.columns)}",detail='Rows × columns',symbol='database'),
          dict(title='UNIQUE TWEETS',value=f"{m['unique_texts']:,}",detail=f"{m['repeated_texts']:,} repeated raw texts",symbol='text'),
          dict(title='MISSING KEYWORDS',value=f"{m['missing_keyword']:,}",detail=f"{m['missing_keyword']/len(df):.1%} of records",tone='amber',symbol='alert'),
          dict(title='MISSING LOCATIONS',value=f"{m['missing_location']:,}",detail=f"{m['missing_location']/len(df):.1%} of records",tone='amber',symbol='alert')])
    with st.container(border=True):
        section('Find the records that matter','Combine class, metadata and literal text filters.')
        a,b,c=st.columns(3)
        labels=a.multiselect('Target',[0,1],default=[0,1],format_func=lambda x:'Disaster-related (1)' if x else 'Non-disaster (0)')
        keywords=b.multiselect('Keyword',sorted(df.keyword.fillna('(Missing)').astype(str).unique()))
        locations=c.multiselect('Location',sorted(df.location.fillna('(Missing)').astype(str).unique()))
        query=st.text_input('Search tweet text',placeholder='Search a word or phrase…')
    view=filter_records(df,labels,keywords,locations,query)
    records,language,quality=st.tabs(['Records & exports','Language analytics','Schema & quality'])
    with records:
        columns=st.multiselect('Visible columns',list(df.columns),default=list(df.columns))
        display=view.copy()
        for col in ['keyword','location']: display[col]=display[col].fillna('(Missing)').astype(str)
        if columns: table(display[columns],'records')
        else: st.info('Select at least one visible column.')
        exports(view,'filtered_records')
        st.caption('Exports preserve original metadata values. “(Missing)” is a display-only label.')
        if len(view):
            with st.expander('Read full tweet'):
                index=st.selectbox('Read full tweet (source row)',view.index)
                st.write(view.loc[index,'text'])
    with language:
        if view.empty: st.info('No matching records. Adjust the filters to explore language patterns.')
        else:
            a,b=st.columns(2)
            mode=a.selectbox('Word-frequency class',['All','Disaster-related','Non-disaster'])
            n=b.radio('Phrase length',[1,2,3],horizontal=True)
            words=view if mode=='All' else view[view.target.eq(1 if mode=='Disaster-related' else 0)]
            if words.empty: st.info('No records in the selected class within the current filters.')
            else:
                a,b=st.columns([1.4,1])
                with a,panel('Words & phrases','Display-only stop-word filtering; model features retain stop words'):
                    fig=px.bar(frequencies(words.text,n).head(15),x='Count',y='Term',orientation='h',color_discrete_sequence=COLORS)
                    fig.update_yaxes(autorange='reversed'); chart(fig,440)
                with b,panel('Hashtag signals','Hashtag counts from the same selected class and filters'):
                    tags=Counter(tag.lower() for tweet in words.text for tag in re.findall(r'#\w+',tweet))
                    tags=pd.DataFrame(tags.most_common(12),columns=['Hashtag','Count'])
                    if tags.empty: st.info('No hashtags in these records.')
                    else:
                        fig=px.bar(tags,x='Count',y='Hashtag',orientation='h',color_discrete_sequence=['#60A5FA'])
                        fig.update_yaxes(autorange='reversed'); chart(fig,440)
                    st.dataframe(tags,hide_index=True)
                a,b=st.columns(2)
                with a,panel('Metadata keywords','Class-specific, nonmissing keyword frequency'):
                    kw=words.keyword.dropna().astype(str).value_counts().head(12).rename_axis('Keyword').reset_index(name='Count')
                    fig=px.bar(kw,x='Count',y='Keyword',orientation='h',color_discrete_sequence=COLORS); fig.update_yaxes(autorange='reversed'); chart(fig,370)
                with b,panel('Text structure','Actual character lengths for the filtered records'):
                    lengths=view.assign(Characters=view.text.str.len(),Class=view.target.map({0:'Non-disaster',1:'Disaster-related'}))
                    chart(px.histogram(lengths,x='Characters',color='Class',nbins=30,color_discrete_map=CLASS_COLORS),370)
                st.caption(f"Selected language subset: {len(words):,} records; {words.text.str.contains('@',regex=False).sum():,} contain mentions; {words.text.str.contains('#',regex=False).sum():,} contain hashtags.")
    with quality:
        st.dataframe(quality_frame(df),hide_index=True,width='stretch')
        status('Duplicate audit',f"{m['duplicate_rows']:,} complete duplicate rows · {m['repeated_texts']:,} repeated raw texts",'amber' if m['repeated_texts'] else 'teal')
        with st.expander('Inspect repeated raw texts'):
            table(df[df.text.duplicated(keep=False)].sort_values('text'),'duplicates')
        dist=class_distribution(view)
        chart(px.bar(dist,x='Class',y='Tweets',color='Class',color_discrete_map=CLASS_COLORS,title='Class distribution within current filters'))
        st.caption('Quality counts describe the complete source dataset. Missing keyword/location values are retained; location is not verified geolocation.')
        with st.expander('Dataset summary JSON'): st.json(summary(df))


def analyzer():
    df,bundle=context(); header('Tweet Analyzer','Read the language. Inspect the model. Keep human judgment in the loop.')
    input_col,result_col=st.columns([1.15,1])
    with input_col,panel('Compose & analyze','Paste a tweet or load an actual dataset record.'):
        sample=pd.concat([df[df.target.eq(1)].head(3),df[df.target.eq(0)].head(3)])
        choices=[None]+sample.index.tolist()
        choice=st.selectbox('Dataset examples',choices,format_func=lambda i:'Choose a source record' if i is None else f"ID {df.loc[i,'id']} · {'Disaster-related' if df.loc[i,'target'] else 'Non-disaster'} · {df.loc[i,'text'][:65]}")
        a,b=st.columns(2)
        if a.button('Use example',width='stretch') and choice is not None: st.session_state['tweet']=df.loc[choice,'text']
        if b.button('Clear input',width='stretch'):
            st.session_state['tweet']=''; st.session_state.pop('single_result',None)
        st.caption('Dataset examples carry known labels and may have been used in training. They are demonstrations, not independent validation.')
        tweet=st.text_area('Tweet text',height=210,key='tweet',placeholder='Paste the text you want to classify…',max_chars=10000)
        kpis([dict(title='CHARACTERS',value=f'{len(tweet):,}',detail='Raw input length',symbol='text'),dict(title='WORDS',value=f'{len(tweet.split()):,}',detail='Whitespace-separated words',symbol='text',tone='blue')])
        with st.expander('NLP processing preview'):
            cleaned=clean_text(tweet)
            st.caption('Normalized text · URLs/mentions removed, negation and hashtag words retained')
            st.write(cleaned or 'Enter a tweet to preview normalization.')
            st.caption('Word TF-IDF tokenizer output')
            if bundle:
                vectorizer=bundle['pipeline'].named_steps['features'].transformer_list[0][1].named_steps['tfidf']
                st.write(vectorizer.build_tokenizer()(cleaned))
            st.caption(f"Raw hashtags: {tweet.count('#')} · Raw mentions: {tweet.count('@')} · Exclamation marks: {tweet.count('!')}")
        if st.button('Analyze Tweet',type='primary',disabled=bundle is None,width='stretch'):
            if not clean_text(tweet):
                st.session_state.pop('single_result',None); st.error('Enter text containing usable words or numbers.')
            else:
                with st.spinner('Analyzing with the saved pipeline…'):
                    model=bundle['pipeline']; start=time.perf_counter(); pred=int(model.predict([tweet])[0])
                    probability=float(model.predict_proba([tweet])[0,1]) if hasattr(model,'predict_proba') else None
                    elapsed=time.perf_counter()-start
                    st.session_state['single_result']={'text':tweet,'prediction':pred,'probability':probability,'seconds':elapsed,'model_stamp':(ARTIFACTS/'model.joblib').stat().st_mtime_ns}
    with result_col,panel('Classification result','A model interpretation of text, not an event confirmation'):
        result=st.session_state.get('single_result')
        current_stamp=(ARTIFACTS/'model.joblib').stat().st_mtime_ns if bundle else None
        if result and result['text']==tweet and result['model_stamp']==current_stamp:
            label='Disaster-related' if result['prediction'] else 'Non-disaster'
            result_card(label,bundle['metadata']['selected_model'],result['probability'])
            st.caption(f"Prediction + probability processing: {result['seconds']*1000:.1f} ms (this request).")
            if result['probability'] is not None: st.caption('Native model estimate; calibration is not validated. This is not the probability that a real disaster occurred.')
            else: st.info('Probability unavailable: this estimator provides decision scores only.')
            explanation=explain(bundle['pipeline'],tweet)
            if explanation is not None and len(explanation):
                explanation=explanation.copy(); explanation['Direction']=explanation['Contribution toward disaster'].map(lambda x:'Disaster-related' if x>0 else 'Non-disaster')
                fig=px.bar(explanation,x='Contribution toward disaster',y='Feature',color='Direction',orientation='h',color_discrete_map=CLASS_COLORS)
                fig.update_yaxes(autorange='reversed'); chart(fig,390)
                st.caption('Weighted learned features: positive favors disaster; negative favors non-disaster. Associations do not establish causality or verify an event.')
            else: st.info('No supported local coefficient explanation for this model/input.')
        else:
            insight('Ready for analysis','Enter a tweet and select Analyze Tweet. Your result and learned feature contributions will appear here.','blue')
    st.info(DISCLAIMER)


def lab():
    df,bundle=context(); header('Model Lab','Controlled experiments. Reproducible splits. Transparent model selection.')
    with st.expander('Implemented NLP & feature engineering',expanded=True):
        st.dataframe(pd.DataFrame([
            ['Normalization + tokenization','Implemented','URLs/mentions removed; negation and hashtag words retained'],
            ['Word TF-IDF','Evaluated','Unigrams + bigrams; fitted within training folds'],
            ['Character TF-IDF','Evaluated variant','char_wb 3–5 grams; captures subword patterns'],
            ['Structural features','Evaluated variant','Length, word, hashtag, mention and exclamation counts'],
            ['Sentiment / pretrained embeddings','Future extension','Not implemented or evaluated']
        ],columns=['Technique','Status','Method']),hide_index=True,width='stretch')
    feature_workspace(df,bundle)
    section('Training controls','Four model families plus character and structural feature experiments. Training runs only when requested.')
    a,b,c=st.columns(3)
    seed=a.number_input('Random seed',0,100000,42); test_size=b.selectbox('Approximate holdout share',[.2,.25,.33]); folds=c.selectbox('Cross-validation folds',[3,5])
    st.caption('Normalized repeated texts remain together. Group stratification gives an approximate holdout share. Training can take several minutes; refreshes do not retrain.')
    if st.button('Train and compare models',type='primary'):
        bar=st.progress(0,text='Preparing training data')
        try:
            with st.spinner('Fitting leakage-safe pipelines…'):
                train(df,seed=int(seed),test_size=test_size,folds=folds,progress=lambda v,m:bar.progress(v,text=m))
            artifact.clear(); st.rerun()
        except Exception as exc: st.error(f'Training failed: {exc}. Check class counts and available memory.')
    if bundle:
        meta=bundle['metadata']; status('Selected model',meta['selected_model']); st.write(meta['selection_metric'])
        st.dataframe(comparison(meta),hide_index=True,width='stretch'); performance_detail(meta)

def performance_detail(meta):
    names=list(meta['results'])
    name=st.selectbox('Inspect evaluated model',names,index=names.index(meta['selected_model']))
    result=meta['results'][name]
    items=[]
    for key,label in [('accuracy','HOLDOUT ACCURACY'),('precision','DISASTER PRECISION'),('recall','DISASTER RECALL'),('f1','DISASTER F1')]:
        if result.get(key) is not None: items.append(dict(title=label,value=f'{result[key]:.1%}',detail=f"{meta['test_rows']:,} holdout records",tone='amber' if key=='recall' else 'teal',symbol='chart'))
    kpis(items)
    a,b=st.columns(2)
    with a: chart(px.imshow(result['confusion_matrix'],x=['Predicted non-disaster','Predicted disaster'],y=['Actual non-disaster','Actual disaster'],text_auto=True,color_continuous_scale='Teal',title='Holdout confusion matrix'))
    with b:
        st.dataframe(pd.DataFrame(result['classification_report']).T,width='stretch')
        insight('Cross-validation variability',f"Training CV F1: {result['cv_f1']:.3f} ± {result['cv_std']:.3f} across {meta['cv_folds']} grouped folds.")
    scores=pd.DataFrame({'Evaluation':['Fitted training','Mean CV training','Mean CV validation','Final holdout'],
                         'F1':[result.get('train_f1'),result.get('cv_train_f1'),result.get('cv_f1'),result.get('f1')]}).dropna()
    fig=px.bar(scores,x='Evaluation',y='F1',text='F1',color='Evaluation',color_discrete_sequence=COLORS)
    fig.update_traces(texttemplate='%{y:.3f}'); fig.update_layout(showlegend=False,title='Training vs. validation vs. holdout'); fig.update_yaxes(range=[0,1])
    chart(fig,320)
    metrics=[]
    for key,title in [('roc_auc','HOLDOUT ROC-AUC'),('average_precision','AVERAGE PRECISION'),('training_seconds','TRAINING SEARCH TIME'),('inference_seconds','HOLDOUT PREDICT + METRICS')]:
        if result.get(key) is not None:
            is_time=key.endswith('seconds')
            metrics.append(dict(title=title,value=f"{result[key]:.2f} s" if is_time else f"{result[key]:.3f}",
                                detail='Saved measured duration' if is_time else 'Saved holdout evaluation',symbol='clock' if is_time else 'chart',tone='blue'))
    kpis(metrics)
    st.caption('Saved holdout timing includes predict() plus scalar metric calculations; it is not pure per-tweet latency. Training time includes CV search and refitting.')
    a,b=st.columns(2)
    roc,pr=go.Figure(),go.Figure()
    for name,r in meta['results'].items():
        if r.get('roc') and r.get('roc_auc') is not None:
            roc.add_trace(go.Scatter(x=r['roc']['fpr'],y=r['roc']['tpr'],name=f"{name} (AUC {r['roc_auc']:.3f})"))
        if r.get('pr') and r.get('average_precision') is not None:
            pr.add_trace(go.Scatter(x=r['pr']['recall'],y=r['pr']['precision'],name=f"{name} (AP {r['average_precision']:.3f})"))
    roc.update_layout(title='Holdout ROC curves',xaxis_title='False positive rate',yaxis_title='True positive rate',legend=dict(orientation='h',y=-.3))
    pr.update_layout(title='Holdout precision–recall curves',xaxis_title='Recall',yaxis_title='Precision',legend=dict(orientation='h',y=-.3))
    with a:
        if roc.data: chart(roc,510)
        else: st.info('ROC scores unavailable for this evaluation.')
    with b:
        if pr.data: chart(pr,510)
        else: st.info('Precision–recall scores unavailable for this evaluation.')
    st.subheader('Training-only tuning results'); st.dataframe(pd.DataFrame(result['cv_candidates']))
    st.caption('ROC uses native probabilities or SVC decision scores. Decision scores are never displayed as confidence. No holdout threshold optimization is performed.')
    with st.expander('Artifact metadata'): st.json({k:v for k,v in meta.items() if k not in ['results','train_indices','test_indices']})

def performance():
    _,bundle=context(); header('Model Performance','Understand accuracy, failure modes and generalization using saved evaluations.')
    if bundle is None: st.info('Train a model in Model Lab to generate actual evaluation results.'); return
    meta=bundle['metadata']; status('Evaluation lineage',f"Seed {meta['seed']} · {meta['train_rows']:,} training / {meta['test_rows']:,} holdout · {meta['cv_folds']}-fold grouped CV")
    comp=comparison(meta).melt(id_vars='Model',value_vars=['precision','recall','f1'],var_name='Metric',value_name='Score').dropna()
    fig=px.bar(comp,x='Score',y='Model',color='Metric',orientation='h',barmode='group',color_discrete_sequence=COLORS,title='Common holdout comparison'); fig.update_xaxes(range=[0,1]); chart(fig,470)
    performance_detail(meta)
    st.subheader('Methodology and error analysis')
    st.write('The training partition alone supplies vocabulary, hyperparameters and model selection. Normalized matching texts stay in the same fold. The final holdout is untouched until all tuning is complete. The saved production pipeline remains fitted only on training data, preserving this evaluation lineage.')
    st.write('False positives flag non-disaster language and can cause false alarms. False negatives miss disaster-related language. Compare both errors alongside recall and precision. Training-to-validation gaps suggest overfitting; related events and paraphrases may still cross partitions.')
    error_path=ARTIFACTS/'holdout_errors.csv'
    if error_path.exists():
        errors=pd.read_csv(error_path)
        mode=st.radio('Inspect error type',['All errors','False positives','False negatives'],horizontal=True)
        if mode=='False positives': errors=errors[errors.target.eq(0) & errors.predicted_target.eq(1)]
        if mode=='False negatives': errors=errors[errors.target.eq(1) & errors.predicted_target.eq(0)]
        table(errors,'errors')
        exports(errors,'holdout_errors')
    st.warning('Historical English tweets, ambiguous labels and event-specific vocabulary limit generalization. No temporal or external-domain validation, probability calibration, live monitoring or verified geolocation is provided.')

def batch():
    _,bundle=context(); header('Prediction Explorer','Turn a file of text into reviewable predictions, with every input row accounted for.')
    section('Upload & configure','CSV / XLSX · Up to 25 MB and 100,000 rows · A target column is not required')
    upload=st.file_uploader('Prediction file',type=['csv','xlsx'])
    if upload:
        try:
            if upload.name.lower().endswith('.csv'): df=pd.read_csv(upload)
            else:
                book=pd.ExcelFile(upload); sheet=st.selectbox('Sheet',book.sheet_names); df=pd.read_excel(book,sheet_name=sheet)
            if df.empty or not len(df.columns): raise ValueError('The file has no records.')
            if len(df)>100000: raise ValueError('Maximum 100,000 rows per batch. Split the file into smaller batches.')
        except Exception as exc: st.error(f'Unable to read uploaded file: {exc}'); return
        names=list(df.columns); column=st.selectbox('Text column',names,index=names.index('text') if 'text' in names else 0)
        st.dataframe(df.head(5),hide_index=True)
        token=(upload.name,hashlib.sha256(upload.getvalue()).hexdigest(),column,sheet if upload.name.lower().endswith('.xlsx') else None,(ARTIFACTS/'model.joblib').stat().st_mtime_ns if bundle else None)
        if st.button('Run batch prediction',type='primary',disabled=bundle is None):
            with st.spinner('Classifying uploaded text…'):
                started=time.perf_counter()
                st.session_state['batch_result']=predict_batch(bundle['pipeline'],df,column); st.session_state['batch_token']=token
                st.session_state['batch_seconds']=time.perf_counter()-started
                st.session_state['batch_model']=bundle['metadata']['selected_model']
        if st.session_state.get('batch_token')==token and 'batch_result' in st.session_state:
            out=st.session_state['batch_result']; status=next(c for c in out.columns if c.endswith('status') and c not in df.columns); label=status[:-6]+'label'; target=status[:-6]+'predicted_target'
            batch_summary(out,status,st.session_state.get('batch_seconds'))
            failed=int(out[status].ne('Predicted').sum()); st.warning(f'{failed:,} rows could not be predicted. Filter by status to inspect them.') if failed else st.success(f'All {len(out):,} rows predicted.')
            counts=out.loc[out[status].eq('Predicted'),label].value_counts().rename_axis('Class').reset_index(name='Predictions')
            fig=px.bar(counts,x='Class',y='Predictions',color='Class',color_discrete_map=CLASS_COLORS,text='Predictions',title='Predicted text classes')
            fig.update_layout(showlegend=False); chart(fig)
            st.caption('These counts describe model predictions, not confirmed disaster incidents.')
            search=st.text_input('Search results'); classes=st.multiselect('Prediction status',list(out[status].unique()),default=list(out[status].unique()))
            shown=out[out[status].isin(classes)]
            selected_labels=st.multiselect('Predicted class',['Disaster-related','Non-disaster'],default=['Disaster-related','Non-disaster'])
            shown=shown[shown[label].isin(selected_labels) | shown[status].ne('Predicted')]
            if search: shown=shown[shown[column].astype(str).str.contains(search,case=False,regex=False)]
            table(shown,'predictions'); exports(shown,'predictions')
            if st.checkbox('Explicitly evaluate against supplied ground truth'):
                truth=st.selectbox('Ground-truth column',list(df.columns))
                valid=out[status].eq('Predicted') & df[truth].isin([0,1])
                if not valid.any(): st.error('No valid numeric 0/1 ground-truth labels for predicted rows.')
                else:
                    from sklearn.metrics import classification_report
                    st.caption(f'Evaluating {int(valid.sum())} valid rows; these are uploaded-file results, not saved holdout results.')
                    st.dataframe(pd.DataFrame(classification_report(df.loc[valid,truth].astype(int),out.loc[valid,target].astype(int),output_dict=True,zero_division=0)).T)
    else:
        insight('Ready for a batch','Upload a CSV or Excel file, select the text column, then run prediction. Original columns are preserved and invalid rows are reported.','blue')
    st.info(DISCLAIMER)

def batch_summary(out,status_column,seconds=None):
    metrics=batch_metrics(out,status_column)
    kpis([dict(title='ROWS RECEIVED',value=f"{metrics['total']:,}",detail='Original input rows',symbol='database'),
          dict(title='SUCCESSFUL',value=f"{metrics['success']:,}",detail='Text classified by the model',symbol='check'),
          dict(title='INVALID ROWS',value=f"{metrics['failed']:,}",detail='Empty, non-text or unusable input',tone='amber',symbol='alert'),
          dict(title='DISASTER PREDICTIONS',value=f"{metrics['disaster']:,}",detail='Predicted text class',tone='red',symbol='alert'),
          dict(title='NON-DISASTER',value=f"{metrics['non_disaster']:,}",detail='Predicted text class',tone='blue',symbol='check')])
    if seconds is not None: st.caption(f'Batch processing time: {seconds:.3f} seconds, measured for this run including validation and prediction.')

def reports():
    df,bundle=context(); header('Reports & Project Information','Export evidence, trace methodology and prepare a clear project handoff.')
    meta=bundle['metadata'] if bundle else {}; stats=summary(df)
    a,b=st.columns(2)
    with a,panel('Analytical report','Dataset summary and complete saved evaluation metadata'):
        st.download_button('Download HTML analytical report',html_report(stats,meta),'disastersense_report.html','text/html',width='stretch')
        st.download_button('Download metadata JSON',json.dumps(meta,indent=2),'model_metadata.json','application/json',width='stretch')
    with b,panel('Dataset quality report','Schema, data types and missing-cell counts'):
        exports(pd.DataFrame({'Column':df.columns,'Dtype':df.dtypes.astype(str).values,'Missing':df.isna().sum().values}),'dataset_summary')
    if bundle:
        with panel('Model comparison export','Training CV and holdout metrics remain separately labeled'):
            exports(comparison(meta),'model_evaluation')
    if 'batch_result' in st.session_state:
        with panel('Latest batch results',f"Model: {st.session_state.get('batch_model','Saved model')} · Most recent run in this session"):
            exports(st.session_state['batch_result'],'latest_prediction_report')
    st.subheader('Project objectives and methodology')
    st.write('DisasterSense AI implements the four project stages: data exploration and preparation; TF-IDF feature engineering and model selection; holdout evaluation and validation; deployment through a Python web interface. The original supplied Excel is the sole training source.')
    st.write(f'The supplied dataset has {len(df):,} rows and {len(df.columns)} columns. The project statement describes 10,000 tweets; no additional records were fabricated. Labels: 0 = non-disaster; 1 = disaster-related.')
    st.write('Stack: Python, Streamlit, Pandas, NumPy, scikit-learn, Plotly, Joblib, OpenPyXL. Classical NLP avoids external APIs and GPU requirements. Sentiment and pretrained embeddings are deferred because their incremental value has not been established.')
    st.subheader('Run locally'); st.code('python -m pip install -r requirements.txt\npython train.py\npython -m streamlit run app.py',language='powershell')
    st.write('Use Tweet Analyzer for a single text. Prediction Explorer accepts CSV and XLSX, preserves input columns, reports invalid rows and exports results. Retrain explicitly in Model Lab after data or dependency changes.')
    st.info(DISCLAIMER)
    st.write('Historical English data and noisy labels limit generalization. Location is unverified context. This application has no live X/Twitter connection or emergency notification service.')
    st.subheader('Sources')
    st.download_button('Read supplied project statement',(ROOT/'data/original/Project statement.pdf').read_bytes(),'Project statement.pdf','application/pdf')
    for title,url in REFERENCES: st.markdown(f'[{title}]({url})')

REFERENCES=[('Kaggle: NLP with Disaster Tweets','https://www.kaggle.com/c/nlp-getting-started'),('Prabhath32 project','https://github.com/Prabhath32/Disaster-Tweet-Classification-'),('myrazd NLP project','https://github.com/myrazd/disaster-tweet-classification-nlp'),('Hugging Face classifier reference','https://huggingface.co/mahmad932/disaster-tweet-classifier'),('Streamlit multipage documentation','https://docs.streamlit.io/develop/concepts/multipage-apps/overview'),('Plotly documentation','https://plotly.com/python/getting-started/')]

def feature_workspace(df, bundle):
    section('Feature engineering workbench','Inspect actual transformations used by the ML pipeline; no vocabulary is refitted here.')
    from src.feature_inspection import fitted_features, structural_frame
    mode=st.selectbox('Feature analysis class',['All','Disaster-related','Non-disaster'])
    subset=df if mode=='All' else df[df.target.eq(1 if mode=='Disaster-related' else 0)]
    structures=structural_frame(subset.text)
    a,b=st.columns(2)
    with a:
        feature=st.selectbox('Structural feature',list(structures.columns))
        chart(px.histogram(structures,x=feature,nbins=30,title='Pipeline-scaled structural distribution'))
    with b:
        st.dataframe(structures.describe().T,width='stretch')
        st.caption('Exact TweetStructure transform: characters / 280, words / 50, hashtags / 5, mentions / 5, exclamations / 5; each capped at 3. These are model inputs in the evaluated structural variant, not raw counts.')
    if bundle is None:
        st.info('Train a model to inspect its fitted TF-IDF vocabulary.'); return
    model=bundle['pipeline']
    branches=[name for name, transformer in model.named_steps['features'].transformer_list if name in ('word','char')]
    branch=st.selectbox('Fitted TF-IDF branch',branches)
    frame,shape,nonzero=fitted_features(model,subset.text,branch)
    kpis([dict(title='RECORDS TRANSFORMED',value=f'{shape[0]:,}',detail=mode),
          dict(title='FITTED VOCABULARY',value=f'{shape[1]:,}',detail='Learned from training records only'),
          dict(title='SPARSE DENSITY',value=f'{nonzero/(shape[0]*shape[1]):.3%}',detail='Nonzero feature values')])
    top=st.slider('Top TF-IDF features',5,40,15)
    chart(px.bar(frame.head(top).iloc[::-1],x='Mean weight',y='Feature',orientation='h',title='Mean TF-IDF across selected records'),max(350,top*20))
    st.caption('Mean weight includes zero values. This describes feature prevalence, not causal importance. Character features include word-boundary spaces.')
    row=st.selectbox('Inspect transformed source row',subset.index)
    st.write(subset.loc[row,'text']); st.code(clean_text(subset.loc[row,'text']),language=None)
    active,_,_=fitted_features(model,[subset.loc[row,'text']],branch)
    st.dataframe(active.rename(columns={'Mean weight':'TF-IDF weight'}),hide_index=True,width='stretch')
    if active.empty: st.info('This record has no terms in the fitted vocabulary for this branch.')
    comp=comparison(bundle['metadata'])
    variants=comp[comp.Model.isin(['Logistic Regression','Logistic + character','Logistic + structure'])]
    section('Feature variant comparison','Saved experiments on common grouped partitions; selection uses training CV only.')
    chart(px.bar(variants.melt(id_vars='Model',value_vars=['cv_f1','f1'],var_name='Evaluation',value_name='F1'),x='Model',y='F1',color='Evaluation',barmode='group'))
    baseline=variants[variants.Model.eq('Logistic Regression')]
    if len(baseline):
        deltas=variants[['Model','cv_f1','f1']].copy()
        deltas['CV F1 delta vs word-only']=deltas.cv_f1-baseline.cv_f1.iloc[0]
        st.dataframe(deltas,hide_index=True)
    st.caption('The selected saved pipeline contains: '+', '.join(name for name,_ in model.named_steps['features'].transformer_list)+'. Structural and character variants are compared separately; their combination was not evaluated.')
