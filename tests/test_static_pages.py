import json, subprocess, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load


def test_static_predictions_match_python():
    from webapp.build_pages import export_models, ROOT
    models=export_models()
    node=shutil.which('node')
    if not node: pytest.skip('Node.js is required to verify the Pages predictor')
    for key, file in [('life','life_expectancy_model.joblib'),('paris','paris_model.joblib'),('student','student_model.joblib')]:
        m=load(ROOT/'outputs'/file); info=models[key]; sample=info['example'].copy()
        if key=='life':
            from hw2lib.predict import prepare_features
            x=prepare_features(pd.DataFrame([sample]),m['features'])
        else:
            x=pd.DataFrame([sample],columns=m['info']['features'])
        expected=m['pipeline'].predict(x)[0]
        inp=json.dumps({'model':info,'record':sample},ensure_ascii=False)
        result=subprocess.run([str(node),str(ROOT/'webapp'/'static_predict_test.cjs')],input=inp,text=True,capture_output=True,encoding='utf8',check=True)
        assert float(result.stdout)==pytest.approx(expected,rel=1e-10,abs=1e-7)
import pytest

def test_pages_build_contains_custom_analysis_assets():
    from webapp.build_pages import build, ROOT
    build()
    site = ROOT / 'site'
    html = (site / 'index.html').read_text(encoding='utf8')
    assert 'src="custom.js"' in html
    assert (site / 'custom.js').is_file()
    assert (site / 'custom_worker.js').is_file()
    assert (site / 'custom_dataset.py').read_text(encoding='utf8') == (ROOT / 'webapp/custom_dataset.py').read_text(encoding='utf8')

