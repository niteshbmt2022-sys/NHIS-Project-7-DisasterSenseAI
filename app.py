import streamlit as st
from config import ROOT
from src.ui import overview, explorer, analyzer, lab, batch, performance, reports
from src.ui_components import install_styles

st.set_page_config(page_title='DisasterSense AI',page_icon=str(ROOT/'assets/logo.svg'),layout='wide')
install_styles()
st.logo(str(ROOT/'assets/wordmark.svg'),icon_image=str(ROOT/'assets/logo.svg'),size='large')
with st.sidebar:
    st.html('<div class="ds-brand-note">NLP INTELLIGENCE PLATFORM<br>Turning Social Signals into<br>Actionable Intelligence.</div>')
pages=[st.Page(overview,title='Executive Dashboard',icon=':material/dashboard:',default=True),
       st.Page(explorer,title='Data Explorer',icon=':material/database:'),
       st.Page(analyzer,title='Tweet Analyzer',icon=':material/manage_search:'),
       st.Page(lab,title='Model Lab',icon=':material/science:'),
       st.Page(batch,title='Prediction Explorer',icon=':material/upload_file:'),
       st.Page(performance,title='Model Performance',icon=':material/monitoring:'),
       st.Page(reports,title='Reports & Project Information',icon=':material/description:')]
st.navigation({'INTELLIGENCE':pages[:3],'MODEL OPERATIONS':pages[3:6],'RESOURCES':pages[6:]}).run()
with st.sidebar:
    st.divider()
    st.caption('HISTORICAL DATASET\n\nModel outputs require human verification.')
st.caption('DisasterSense AI • Historical text intelligence • Predictions require human verification')
