"""WHO country-year data contract and country-disjoint split."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

TARGET = 'Life expectancy'


def validate_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError('Duplicate column names after trimming')
    for c in (TARGET, 'Country', 'Status', 'Year'):
        if c not in df:
            raise ValueError(f'Missing column: {c}')
    if df['Country'].isna().any() or df['Year'].isna().any():
        raise ValueError('Country and Year must be present')
    if df.duplicated(['Country', 'Year']).any():
        raise ValueError('Duplicate Country/Year records')
    if not df.Status.isin(['Developing', 'Developed']).all():
        raise ValueError('Unknown or missing Status')
    df['Status'] = df.Status.map({'Developing': 0, 'Developed': 1})
    features = [c for c in df.columns if c not in (TARGET, 'Country')]
    if len(features) != 20:
        raise ValueError(f'Expected 20 predictors, got {len(features)}')
    for c in features + [TARGET]:
        if not pd.api.types.is_numeric_dtype(df[c]) or np.isinf(df[c]).any():
            raise ValueError(f'{c} must contain finite numeric values or missing values')
    clean = df.dropna(subset=[TARGET]).copy()
    if clean[features].isna().all().any():
        raise ValueError('A feature is entirely missing')
    if clean.Country.nunique() < 6:
        raise ValueError('At least 6 countries are required')
    return clean, features


def split_data(df: pd.DataFrame):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    a, b = next(splitter.split(df, groups=df.Country))
    return df.iloc[a].copy(), df.iloc[b].copy()
