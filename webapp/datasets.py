"""Read each dataset's own model and validate its inputs."""
from pathlib import Path
import math
import pandas as pd
from joblib import load

ROOT=Path(__file__).resolve().parent.parent


def bundle(key):
    if key not in ('paris','student'): raise ValueError('未知資料集')
    return load(ROOT/'outputs'/f'{key}_model.joblib')


def dataset_info(key):
    return bundle(key)['info']


def predict_dataset(key,record):
    model=bundle(key); info=model['info']
    if not isinstance(record,dict): raise ValueError('輸入必須是物件')
    missing=[x for x in info['features'] if x not in record]
    if missing: raise ValueError('缺少欄位: '+', '.join(missing))
    row={}
    for name in info['features']:
        val=record[name]; opts=info['categories'].get(name)
        if opts is not None and isinstance(opts[0],str):
            if val not in opts: raise ValueError(f'{name} 選項無效')
            row[name]=val
        elif val is None or val=='': row[name]=float('nan')
        else:
            try: row[name]=float(val)
            except (ValueError,TypeError): raise ValueError(f'{name} 必須為數字')
            if not math.isfinite(row[name]): raise ValueError(f'{name} 必須為有限數字')
            if opts is not None and row[name] not in opts: raise ValueError(f'{name} 選項無效')
    pred=float(model['pipeline'].predict(pd.DataFrame([row],columns=info['features']))[0])
    return {'prediction':pred}
