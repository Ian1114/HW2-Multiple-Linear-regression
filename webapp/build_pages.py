"""Build portable GitHub Pages site from the fitted local OLS models."""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
from joblib import load
from webapp.server import example_record

ROOT=Path(__file__).resolve().parent.parent

def plain(v):
    return v.item() if isinstance(v,np.generic) else v

def export_models():
    result={}
    life=load(ROOT/'outputs/life_expectancy_model.joblib');m=life['pipeline'];features=life['features'];im=m.named_steps['impute'];sc=m.named_steps['scale'];co=m.named_steps['model'].coef_
    result['life']={'intercept':float(m.named_steps['model'].intercept_),'numeric':[{'name':f,'median':float(im.statistics_[i]),'mean':float(sc.mean_[i]),'scale':float(sc.scale_[i]),'coef':float(co[i])} for i,f in enumerate(features)],'categorical':[],'example':example_record()}
    for key in ('paris','student'):
        b=load(ROOT/'outputs'/f'{key}_model.joblib');m=b['pipeline'];prep=m.named_steps['prepare'];co=m.named_steps['model'].coef_; num=prep.named_transformers_['numeric'];cols=prep.transformers_[0][2];im=num.named_steps['imputer'];sc=num.named_steps['scale'];numeric=[{'name':f,'median':float(im.statistics_[i]),'mean':float(sc.mean_[i]),'scale':float(sc.scale_[i]),'coef':float(co[i])} for i,f in enumerate(cols)]; cat=[]
        if key=='student':
            cc=prep.transformers_[1][2]; enc=prep.named_transformers_['category'].named_steps['onehot']
            for i,f in enumerate(cc):
                options=[plain(x) for x in enc.categories_[i]]
                if len(options)!=2: raise ValueError(f'Expected binary categorical feature: {f}')
                cat.append({'name':f,'options':options,'positive':options[1],'coef':float(co[len(cols)+i])})
        result[key]={'intercept':float(m.named_steps['model'].intercept_),'numeric':numeric,'categorical':cat,'example':b['info']['example'],'info':b['info']}
    return result

def build():
    dest=ROOT/'site';(dest/'reports').mkdir(parents=True,exist_ok=True)
    for name in ['style.css','app.js','custom.js','custom_worker.js','static_predict.js']:
        shutil.copy2(ROOT/'webapp/static'/name,dest/name)
    shutil.copy2(ROOT/'webapp/custom_dataset.py',dest/'custom_dataset.py')
    html=(ROOT/'webapp/static/index.html').read_text(encoding='utf8').replace('href="/style.css"','href="style.css"').replace('src="/app.js"','src="app.js"').replace('src="/custom.js"','src="custom.js"').replace('src="/chart.png"','src="chart.png"').replace('href="/report.pdf"','href="report.pdf"')
    html=html.replace('資料僅傳送至目前網站的後端處理，不寫入檔案。','資料在瀏覽器內處理，不上傳至伺服器；首次使用需下載分析套件。')
    html=html.replace('後端將使用已訓練模型計算','瀏覽器將使用已訓練模型計算').replace('<script defer src="app.js"></script>','<script defer src="models.js"></script><script defer src="static_predict.js"></script><script defer src="static_api.js"></script><script defer src="app.js"></script>')
    js=(dest/'app.js').read_text(encoding='utf8').replace("'/report.pdf'","'report.pdf'").replace("'/reports/paris.pdf'","'reports/paris.pdf'").replace("'/reports/student.pdf'","'reports/student.pdf'")
    (dest/'app.js').write_text(js,encoding='utf8')
    (dest/'models.js').write_text('window.HW2_MODELS = '+json.dumps(export_models(),ensure_ascii=False,allow_nan=False)+';',encoding='utf8')
    shutil.copy2(ROOT/'webapp/static/static_api.js',dest/'static_api.js')
    for name in ('style.css','models.js','static_predict.js','static_api.js','app.js','custom.js'):
        version = hashlib.sha256((dest/name).read_bytes()).hexdigest()[:12]
        html=html.replace('="'+name+'"','="'+name+'?v='+version+'"')
    (dest/'index.html').write_text(html,encoding='utf8')
    for source,target in [('outputs/holdout_diagnostics.png','chart.png'),('reports/life_expectancy_report.pdf','report.pdf'),('reports/paris_report.pdf','reports/paris.pdf'),('reports/student_report.pdf','reports/student.pdf')]: shutil.copy2(ROOT/source,dest/target)
    (dest/'.nojekyll').write_text('',encoding='utf8')
    print('Pages site:',dest)
if __name__=='__main__':build()
