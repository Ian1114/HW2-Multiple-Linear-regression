"""Reproducible supplemental CRISP-DM analysis for Paris and student data."""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from webapp.train_datasets import ROOT, SPECS

OUT=ROOT/'outputs'

def adjusted_r2(r2,n,p):
    return 1-(1-r2)*(n-1)/(n-p-1) if n>p+1 else float('nan')

def intervals_for_pipeline(model,xtrain,ytrain,xtest):
    def design(x):
        z=model[:-1].transform(x) if len(model.steps)>1 else np.asarray(x)
        return sm.add_constant(np.asarray(z),has_constant='add')
    a,b=design(xtrain),design(xtest)
    fit=sm.OLS(np.asarray(ytrain),a).fit()
    frame=fit.get_prediction(b).summary_frame(alpha=.05)
    if not np.allclose(frame['mean'],model.predict(xtest),rtol=1e-7,atol=1e-7): raise ValueError('OLS implementations disagree')
    return frame

def analyze(key):
    spec=SPECS[key]; raw=pd.read_csv(ROOT/spec['path']); data=raw.dropna(subset=[spec['target']]); features=spec['features']; cats=spec['categorical']; nums=[x for x in features if x not in cats]
    X=data[features]; y=data[spec['target']]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    prep=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('category',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',drop='first'))]),cats)],verbose_feature_names_out=False)
    assert all(Xtr[c].nunique()==2 for c in cats), 'Feature comparison requires binary categorical variables'
    models={'dummy':Pipeline([('prep',clone(prep)),('model',DummyRegressor())]),'OLS all':Pipeline([('prep',clone(prep)),('model',LinearRegression())]),'KBest 14':Pipeline([('prep',clone(prep)),('select',SelectKBest(f_regression,k=14)),('model',LinearRegression())]),'RFE 14':Pipeline([('prep',clone(prep)),('select',RFE(LinearRegression(),n_features_to_select=14)),('model',LinearRegression())])}
    cv=KFold(n_splits=5,shuffle=True,random_state=42); records=[]; fitted={}; selection={}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for name,model in models.items():
            sc=cross_validate(model,Xtr,ytr,cv=cv,scoring={'rmse':'neg_root_mean_squared_error','mae':'neg_mean_absolute_error','r2':'r2'},return_train_score=True,n_jobs=1,error_score='raise')
            m=clone(model).fit(Xtr,ytr);fitted[name]=m
            if name in ('KBest 14','RFE 14'):
                columns=m.named_steps['prep'].get_feature_names_out(); selection[name]=columns[m.named_steps['select'].get_support()].tolist()
            for subset,xx,yy in [('train',Xtr,ytr),('holdout',Xte,yte)]:
                pred=m.predict(xx); mse=mean_squared_error(yy,pred); r2=r2_score(yy,pred); p=0 if name=='dummy' else (14 if '14' in name else len(features))
                records.append({'model':name,'set':subset,'n':len(yy),'p':p,'mae':mean_absolute_error(yy,pred),'mse':mse,'rmse':np.sqrt(mse),'r2':r2,'adj_r2':adjusted_r2(r2,len(yy),p) if p else None,'cv_rmse':-sc['test_rmse'].mean(),'cv_std':sc['test_rmse'].std(),'cv_mae':-sc['test_mae'].mean(),'cv_r2':sc['test_r2'].mean()})
    report=pd.DataFrame(records);report.to_csv(OUT/f'{key}_report_metrics.csv',index=False)
    model=fitted['OLS all'];pred=model.predict(Xte); resid=np.asarray(yte)-pred
    ztr=model.named_steps['prep'].transform(Xtr); zte=model.named_steps['prep'].transform(Xte); ztr=np.asarray(ztr); zte=np.asarray(zte)
    a=sm.add_constant(ztr,has_constant='add');fit=sm.OLS(np.asarray(ytr),a).fit(); ci=intervals_for_pipeline(model,Xtr,ytr,Xte)
    bp=het_breuschpagan(fit.resid,a); names=model.named_steps['prep'].get_feature_names_out()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        vif=[float(variance_inflation_factor(a,i+1)) for i in range(ztr.shape[1])]
    pd.DataFrame({'feature':names,'vif':vif,'coefficient':model.named_steps['model'].coef_}).to_csv(OUT/f'{key}_diagnostics.csv',index=False)
    order=np.argsort(pred); ix=order[::max(1,len(order)//150)]; bounds=ci.iloc[ix].reset_index(drop=True); t=np.arange(len(ix))
    fig,ax=plt.subplots(1,3,figsize=(12,3.4));ax[0].scatter(yte,pred,s=8,alpha=.4);lo=min(min(yte),min(pred));hi=max(max(yte),max(pred));ax[0].plot([lo,hi],[lo,hi],'r--');ax[0].set(xlabel='Actual',ylabel='Prediction',title='Holdout')
    ax[1].scatter(pred,resid,s=8,alpha=.4);ax[1].axhline(0,color='r');ax[1].set(xlabel='Prediction',ylabel='Residual',title='Residual vs fitted');ax[2].hist(resid,bins=30,color='#71977b');ax[2].set(title='Residual distribution');fig.tight_layout();fig.savefig(OUT/f'{key}_diagnostics.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.4));ax.fill_between(t,bounds.obs_ci_lower,bounds.obs_ci_upper,alpha=.13,label='95% prediction interval');ax.fill_between(t,bounds.mean_ci_lower,bounds.mean_ci_upper,alpha=.35,label='95% mean response CI');ax.plot(t,pred[ix],label='Predicted');ax.scatter(t,np.asarray(yte)[ix],s=8,alpha=.5,label='Actual');ax.set(xlabel='Holdout samples sorted by prediction (subsample)',ylabel=spec['target']);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/f'{key}_intervals.png',dpi=140);plt.close(fig)
    fig=sm.qqplot(resid,line='45',fit=True);plt.title('Holdout residual Q-Q');fig.tight_layout();fig.savefig(OUT/f'{key}_qq.png',dpi=140);plt.close(fig)
    metadata=json.loads((OUT/f'{key}_metadata.json').read_text(encoding='utf8')); metadata.update({'raw_rows':len(raw),'raw_columns':len(raw.columns),'missing_total':int(raw.isna().sum().sum()),'missing_target':int(raw[spec['target']].isna().sum()),'duplicates':int(raw.duplicated().sum()),'bp_lm':float(bp[0]),'bp_p':float(bp[1]),'max_vif':max(vif),'max_vif_feature':str(names[int(np.argmax(vif))]),'selection':selection,'design_rank':int(np.linalg.matrix_rank(a)),'design_columns':int(a.shape[1])})
    (OUT/f'{key}_report_analysis.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    print(key,report.query("set == 'holdout'")[['model','rmse','r2']].to_string(index=False)); print('BP p=',bp[1],'max VIF=',max(vif))

if __name__=='__main__':
    for name in SPECS: analyze(name)
