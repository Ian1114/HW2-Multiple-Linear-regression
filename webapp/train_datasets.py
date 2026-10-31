"""Train independent reproducible baseline regression models for web datasets."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parent.parent
SPECS={
 'paris': {'path':'dataset/Paris Housing Price Prediction/ParisHousing.csv','target':'price','features':['squareMeters','numberOfRooms','hasYard','hasPool','floors','cityCode','cityPartRange','numPrevOwners','made','isNewBuilt','hasStormProtector','basement','attic','garage','hasStorageRoom','hasGuestRoom'],'categorical':[]},
 'student': {'path':'dataset/Student Performance Data Set/student-por.csv','target':'G3','features':['sex','age','address','Medu','Fedu','traveltime','studytime','failures','schoolsup','famsup','paid','activities','higher','internet','famrel','goout','health','absences','G1','G2'],'categorical':['sex','address','schoolsup','famsup','paid','activities','higher','internet']},
}

def train_one(key):
    spec=SPECS[key]; path=ROOT/spec['path']; data=pd.read_csv(path); features=spec['features']; cats=spec['categorical']; nums=[x for x in features if x not in cats]
    data=data.dropna(subset=[spec['target']]).copy(); X=data[features]; y=pd.to_numeric(data[spec['target']],errors='raise')
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,random_state=42)
    prep=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('category',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',drop='first'))]),cats)])
    pipe=Pipeline([('prepare',prep),('model',LinearRegression())]); cv=KFold(n_splits=5,shuffle=True,random_state=42)
    scores=-cross_val_score(pipe,Xtrain,ytrain,cv=cv,scoring='neg_root_mean_squared_error'); pipe.fit(Xtrain,ytrain); pred=pipe.predict(Xtest)
    categories={c:sorted(str(x) for x in Xtrain[c].dropna().unique()) for c in cats}
    for c in ['hasYard','hasPool','isNewBuilt','hasStormProtector','hasStorageRoom']:
        if c in features: categories[c]=[0,1]
    if key == 'student':
        for name, lo, hi in [('Medu',0,4),('Fedu',0,4),('traveltime',1,4),('studytime',1,4),('failures',0,3),('famrel',1,5),('goout',1,5),('health',1,5),('G1',0,20),('G2',0,20)]:
            categories[name] = list(range(lo, hi + 1))
    info={'dataset':key,'target':spec['target'],'features':features,'categories':categories,'rows':len(data),'train_rows':len(Xtrain),'test_rows':len(Xtest),'rmse':float(np.sqrt(mean_squared_error(ytest,pred))),'mae':float(mean_absolute_error(ytest,pred)),'r2':float(r2_score(ytest,pred)),'cv_rmse':float(scores.mean()),'cv_std':float(scores.std()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'example':{k:(None if pd.isna(v) else v.item() if hasattr(v,'item') else v) for k,v in X.iloc[0].items()}}
    output=ROOT/'outputs';output.mkdir(exist_ok=True);dump({'pipeline':pipe,'info':info},output/f'{key}_model.joblib');(output/f'{key}_metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    print(key, json.dumps({k:info[k] for k in ['rows','rmse','mae','r2','cv_rmse']},ensure_ascii=False))

if __name__=='__main__':
    for key in SPECS: train_one(key)
