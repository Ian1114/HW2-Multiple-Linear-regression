"""Predict new country-year rows with the saved pipeline."""
import argparse
import numpy as np
import pandas as pd
from joblib import load


def prepare_features(df, features):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError('Duplicate column names')
    missing = sorted(set(features)-set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if not df.Status.isin(['Developing', 'Developed']).all():
        raise ValueError('Status must be Developing or Developed')
    df['Status'] = df.Status.map({'Developing': 0, 'Developed': 1})
    for col in features:
        if not pd.api.types.is_numeric_dtype(df[col]) or np.isinf(df[col]).any():
            raise ValueError(f'{col} must be numeric and finite or missing')
    return df[features]


def main():
    parser = argparse.ArgumentParser(description='Predict life expectancy from a CSV')
    parser.add_argument('input_csv'); parser.add_argument('output_csv')
    parser.add_argument('--model', default='outputs/life_expectancy_model.joblib')
    args = parser.parse_args()
    bundle = load(args.model)
    data = pd.read_csv(args.input_csv)
    x = prepare_features(data, bundle['features'])
    result = pd.DataFrame({'source_row': np.arange(len(data)), 'prediction_years': bundle['pipeline'].predict(x)})
    for col in ('Country','Year'):
        if col in data: result[col] = data[col].to_numpy()
    result.to_csv(args.output_csv, index=False)


if __name__ == '__main__': main()
