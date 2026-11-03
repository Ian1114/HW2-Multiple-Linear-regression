"""In-memory CSV inspection and reproducible multiple linear regression."""
import csv
import io

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _read(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError('請選擇非空的 UTF-8 CSV 檔案。')
    try:
        clean = content.lstrip('\ufeff')
        try: delimiter = csv.Sniffer().sniff(clean[:8192], delimiters=',;\t').delimiter
        except csv.Error: delimiter = ','
        header = next(csv.reader(io.StringIO(clean), delimiter=delimiter))
        if len(header) < 3 or any(not name.strip() for name in header) or len(set(header)) != len(header):
            raise ValueError('CSV 至少需要三個欄位，欄名不得空白或重複。')
        df = pd.read_csv(io.StringIO(clean), sep=delimiter)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration, csv.Error) as exc:
        raise ValueError('無法讀取 CSV，請確認格式與編碼。') from exc
    if len(df) > 100000 or len(df.columns) > 200:
        raise ValueError('資料最多支援 100,000 筆及 200 個欄位。')
    return df.replace([np.inf, -np.inf], np.nan)


def inspect_csv(content):
    df = _read(content)
    return {'rows': len(df), 'columns': [{'name': str(c), 'numeric': bool(pd.api.types.is_numeric_dtype(df[c])), 'missing': int(df[c].isna().sum()), 'unique': int(df[c].nunique())} for c in df.columns]}


def train_csv(content, target, features):
    df = _read(content)
    if not isinstance(features, list) or len(features) < 2 or len(features) != len(set(features)):
        raise ValueError('請選擇至少兩個不同的輸入特徵。')
    if target not in df.columns or target in features or any(c not in df.columns for c in features):
        raise ValueError('目標欄位與特徵欄位不正確，預測目標不能作為輸入。')
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError('預測目標必須是數值欄位。')
    original_rows = len(df)
    df = df.dropna(subset=[target])
    if len(df) < 20 or df[target].nunique() < 2:
        raise ValueError('預測目標至少需要 20 筆有效資料及兩個不同數值。')
    if any(df[c].isna().all() for c in features):
        raise ValueError('所選特徵不可整欄缺值。')
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    if any(df[c].nunique() > 100 for c in categorical):
        raise ValueError('類別欄位最多支援 100 種不同值；請移除高基數或識別碼欄位。')
    xtrain, xtest, ytrain, ytest = train_test_split(df[features], df[target], test_size=.2, random_state=42)
    if any(xtrain[c].isna().all() for c in features):
        raise ValueError('有特徵在訓練集全部缺值，請取消選取。')
    pre = ColumnTransformer([('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric), ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical)])
    model = Pipeline([('prepare', pre), ('regression', LinearRegression())])
    model.fit(xtrain, ytrain)
    def metrics(y, pred):
        mse = mean_squared_error(y, pred)
        return {'mae': float(mean_absolute_error(y, pred)), 'mse': float(mse), 'rmse': float(np.sqrt(mse)), 'r2': float(r2_score(y, pred))}
    baseline = DummyRegressor().fit(xtrain, ytrain)
    cv = cross_val_score(model, xtrain, ytrain, scoring='neg_mean_squared_error', cv=KFold(n_splits=5, shuffle=True, random_state=42), error_score='raise')
    return {'rows': len(df), 'dropped': original_rows-len(df), 'train_rows': len(xtrain), 'test_rows': len(xtest), 'target': target, 'features': features, 'numeric': numeric, 'categorical': categorical, 'train': metrics(ytrain, model.predict(xtrain)), 'test': metrics(ytest, model.predict(xtest)), 'baseline': metrics(ytest, baseline.predict(xtest)), 'cv_rmse': float(np.sqrt(-cv).mean())}
