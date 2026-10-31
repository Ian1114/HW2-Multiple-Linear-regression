import pandas as pd
import pytest
from hw2lib.data import validate_data, split_data


def frame():
    data = {f'x{i}': list(range(40)) for i in range(18)}
    data['Year'] = range(40)
    data['Status'] = ['Developing', 'Developed'] * 20
    data['Life expectancy '] = [i + 40.0 for i in range(40)]
    data['Country'] = [f'c{i // 2}' for i in range(40)]
    return pd.DataFrame(data)


def test_validate_schema_and_encode_status():
    df, features = validate_data(frame())
    assert len(features) == 20 and 'Country' not in features
    assert set(df.Status) == {0, 1}
    assert 'Life expectancy' not in features


def test_rejects_unknown_status():
    df = frame(); df.loc[0, 'Status'] = 'unknown'
    with pytest.raises(ValueError, match='Status'):
        validate_data(df)


def test_split_has_no_shared_countries():
    df, _ = validate_data(frame())
    train, test = split_data(df)
    assert set(train.Country).isdisjoint(test.Country)
