import numpy as np
import pandas as pd
import pytest
from hw2lib.predict import prepare_features


def test_new_data_validates_and_encodes_status():
    df = pd.DataFrame({'Status':['Developing', 'Developed'], 'Year':[2000, 2001]})
    out = prepare_features(df, ['Year','Status'])
    assert out.columns.tolist() == ['Year','Status']
    assert out.Status.tolist() == [0,1]


def test_new_data_rejects_missing_or_invalid_fields():
    with pytest.raises(ValueError, match='Year'):
        prepare_features(pd.DataFrame({'Status':['Developed']}), ['Year','Status'])
    with pytest.raises(ValueError, match='Status'):
        prepare_features(pd.DataFrame({'Status':['unknown'],'Year':[2000]}), ['Year','Status'])
