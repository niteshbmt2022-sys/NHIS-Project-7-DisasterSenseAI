from io import BytesIO
import html
import json
import pandas as pd

def safe_export_frame(df):
    """Neutralize spreadsheet formula injection in untrusted input text."""
    result = df.copy()
    for col in result.columns:
        result[col] = result[col].map(lambda v: "'"+v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v)
    return result

def csv_bytes(df):
    return safe_export_frame(df).to_csv(index=False).encode('utf-8-sig')

def excel_bytes(df):
    output = BytesIO()
    safe_export_frame(df).to_excel(output, index=False, engine='openpyxl')
    return output.getvalue()

def html_report(summary, metadata):
    return ('<!doctype html><html><meta charset="utf-8"><title>DisasterSense AI Report</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color-scheme:light dark;background:light-dark(#ffffff,#121923);color:light-dark(#172338,#f8fafc)}pre{white-space:pre-wrap}h1{color:#14b8a6}</style>'
            '<h1>DisasterSense AI</h1><p>Historical dataset and model evaluation report. Model outputs require human verification.</p>'
            '<h2>Dataset</h2><pre>'+html.escape(json.dumps(summary, indent=2))+'</pre><h2>Evaluation and metadata</h2><pre>'
            +html.escape(json.dumps(metadata, indent=2))+'</pre></html>').encode('utf-8')
