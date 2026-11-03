"""Local HTTP application for the existing life expectancy model."""
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd
from dotenv import load_dotenv
from joblib import load
from hw2lib.predict import prepare_features
from webapp.datasets import dataset_info, predict_dataset
from webapp.custom_dataset import inspect_csv, train_csv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env', override=False)
MODEL = Path(os.getenv('HW2_MODEL_PATH', 'outputs/life_expectancy_model.joblib'))
if not MODEL.is_absolute(): MODEL = ROOT / MODEL
DATA = ROOT / 'dataset/Life Expectancy (WHO)/Life Expectancy Data.csv'
STATIC = ROOT / 'webapp/static'


def example_record():
    model = load(MODEL)
    raw = pd.read_csv(DATA)
    raw.columns = raw.columns.str.strip()
    row = raw.loc[0, model['features']]
    return {key: (None if pd.isna(value) else (value.item() if hasattr(value, 'item') else value)) for key, value in row.items()}


def predict_record(record):
    if not isinstance(record, dict): raise ValueError('輸入必須是物件')
    model = load(MODEL)
    features = model['features']
    missing = [key for key in features if key not in record]
    if missing: raise ValueError('缺少欄位: ' + ', '.join(missing))
    row = {}
    for key in features:
        value = record[key]
        if key == 'Status':
            row[key] = value
        elif value is None or value == '':
            row[key] = float('nan')
        else:
            try: row[key] = float(value)
            except (ValueError, TypeError): raise ValueError(f'{key} 必須為數字')
            if not math.isfinite(row[key]): raise ValueError(f'{key} 必須為有限數字')
    x = prepare_features(pd.DataFrame([row]), features)
    value = float(model['pipeline'].predict(x)[0])
    return {'prediction': value}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status); self.send_header('Content-Type', 'application/json; charset=utf-8'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ('/api/datasets/paris', '/api/datasets/student'):
            try: self.send_json(dataset_info(path.rsplit('/',1)[-1]))
            except Exception: self.send_json({'error':'資料集或模型無法讀取'},500)
            return
        if path == '/api/example':
            try: self.send_json(example_record())
            except Exception: self.send_json({'error':'無法讀取範例資料或模型'}, 500)
            return
        files = {'/':(STATIC/'index.html','text/html; charset=utf-8'), '/style.css':(STATIC/'style.css','text/css; charset=utf-8'), '/app.js':(STATIC/'app.js','text/javascript; charset=utf-8'), '/custom.js':(STATIC/'custom.js','text/javascript; charset=utf-8'), '/chart.png':(ROOT/'outputs/holdout_diagnostics.png','image/png'), '/intervals.png':(ROOT/'outputs/intervals.png','image/png'), '/report.pdf':(ROOT/'reports/life_expectancy_report.pdf','application/pdf'), '/reports/paris.pdf':(ROOT/'reports/paris_report.pdf','application/pdf'), '/reports/student.pdf':(ROOT/'reports/student_report.pdf','application/pdf')}
        if path not in files: self.send_error(404); return
        file, typ = files[path]
        try: body=file.read_bytes()
        except OSError: self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type',typ); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        path=urlsplit(self.path).path
        if path not in ('/api/predict','/api/predict/paris','/api/predict/student','/api/custom/inspect','/api/custom/train'): self.send_error(404); return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size <= 0 or size > (11 * 1024 * 1024 if path.startswith('/api/custom/') else 32768): raise ValueError('請求大小不正確')
            data=json.loads(self.rfile.read(size))
            if path == '/api/custom/inspect': result = inspect_csv(data.get('csv'))
            elif path == '/api/custom/train': result = train_csv(data.get('csv'), data.get('target'), data.get('features'))
            else: result = predict_record(data) if path == '/api/predict' else predict_dataset(path.rsplit('/',1)[-1],data)
            self.send_json(result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc: self.send_json({'error':str(exc)},400)
        except Exception: self.send_json({'error':'模型目前無法使用'},500)


def main():
    host=os.getenv('HW2_HOST','127.0.0.1'); port=int(os.getenv('PORT',os.getenv('HW2_PORT','8000')))
    print(f'Local: http://{host}:{port}',flush=True)
    ThreadingHTTPServer((host,port), Handler).serve_forever()

if __name__ == '__main__': main()

