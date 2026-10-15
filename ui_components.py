"""Escaped, reusable presentation components; no model or dataset mutation."""
from html import escape
import base64
from contextlib import contextmanager
import streamlit as st
from config import ROOT

TONES = {'teal', 'red', 'blue', 'amber', 'slate'}
ICONS = {
    'database': '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
    'alert': '<path d="m12 3 10 18H2L12 3Z"/><path d="M12 9v5m0 3v1"/>',
    'check': '<path d="m5 12 4 4L19 6"/><circle cx="12" cy="12" r="10"/>',
    'chart': '<path d="M4 3v17h17M8 16v-5m5 5V6m5 10V9"/>',
    'shield': '<path d="m12 2 8 3v7c0 5-8 10-8 10S4 17 4 12V5l8-3Z"/><path d="m8 12 3 3 5-6"/>',
    'text': '<path d="M4 5h16M4 10h16M4 15h10M4 20h7"/>',
    'model': '<rect x="5" y="5" width="14" height="14" rx="3"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/>',
    'clock': '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 3"/>',
}

def safe(value):
    return escape(str(value), quote=True)

def icon(name):
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#718096" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'+ICONS.get(name, ICONS['chart'])+'</svg>'
    encoded=base64.b64encode(svg.encode()).decode()
    return f'<span class="ds-icon-glyph" aria-hidden="true" style="mask-image:url(data:image/svg+xml;base64,{encoded})"></span>'

def install_styles():
    st.html((ROOT/'assets/theme.css').read_text(encoding='utf-8'))

def card_html(title, value, detail='', tone='teal', symbol='chart'):
    tone = tone if tone in TONES else 'teal'
    compact = ' ds-kpi-long' if len(str(value)) > 15 else ''
    return (f'<article class="ds-kpi ds-{tone}{compact}"><div class="ds-kpi-top">'
            f'<span class="ds-kpi-title">{safe(title)}</span><span class="ds-icon">{icon(symbol)}</span></div>'
            f'<div class="ds-kpi-value">{safe(value)}</div><div class="ds-kpi-detail">{safe(detail)}</div></article>')

def kpis(items):
    st.html(f'<div class="ds-kpi-grid ds-count-{len(items)}">'+''.join(card_html(**item) for item in items)+'</div>')

def header(title, subtitle, eyebrow='DISASTERSENSE / INTELLIGENCE WORKSPACE'):
    st.html(f'<header class="ds-header"><div class="ds-eyebrow">{safe(eyebrow)}</div><h1>{safe(title)}</h1><p>{safe(subtitle)}</p></header>')

def section(title, subtitle=''):
    st.html(f'<div class="ds-section"><h2>{safe(title)}</h2><p>{safe(subtitle)}</p></div>')

def status(label, detail='', tone='teal'):
    tone = tone if tone in TONES else 'slate'
    st.html(f'<div class="ds-status ds-{tone}"><span class="ds-dot"></span><strong>{safe(label)}</strong><span>{safe(detail)}</span></div>')

@contextmanager
def panel(title, subtitle=''):
    with st.container(border=True):
        section(title, subtitle)
        yield

def result_card(label, model, probability=None):
    tone = 'red' if label == 'Disaster-related' else 'blue'
    extra = '' if probability is None else f'<div class="ds-result-prob">{probability:.1%}<span>Estimated disaster probability</span></div>'
    st.html(f'<article class="ds-result ds-{tone}"><div class="ds-eyebrow">MODEL CLASSIFICATION</div><h2>Predicted: {safe(label)}</h2><p>{safe(model)}</p>{extra}<div class="ds-result-foot">Text classification · Human review required</div></article>')

def insight(title, body, tone='teal'):
    tone = tone if tone in TONES else 'teal'
    st.html(f'<div class="ds-insight ds-{tone}"><h3>{safe(title)}</h3><p>{safe(body)}</p></div>')
