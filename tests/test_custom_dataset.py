import pytest

from webapp.custom_dataset import inspect_csv, train_csv


CSV = 'x,z,group,y\n' + ''.join(f'{i},{i % 7},{"a" if i % 2 else "b"},{3*i + 2*(i % 7)}\n' for i in range(50))


def test_inspect_and_train_custom_regression():
    info = inspect_csv(CSV)
    assert info['rows'] == 50
    assert [col['name'] for col in info['columns']] == ['x', 'z', 'group', 'y']
    result = train_csv(CSV, 'y', ['x', 'z', 'group'])
    assert result['train_rows'] == 40 and result['test_rows'] == 10
    assert result['test']['rmse'] < 1e-8
    assert result['test']['r2'] > 0.999999


def test_reject_ambiguous_columns_and_target_leakage():
    with pytest.raises(ValueError):
        inspect_csv('x,x,y\n1,2,3\n')
    with pytest.raises(ValueError):
        train_csv(CSV, 'y', ['x', 'y'])


def test_reject_invalid_target_and_too_few_features():
    with pytest.raises(ValueError):
        train_csv(CSV, 'group', ['x', 'z'])
    with pytest.raises(ValueError):
        train_csv(CSV, 'y', ['x'])


def test_semicolon_csv_is_supported():
    assert inspect_csv(CSV.replace(',', ';'))['rows'] == 50


def test_inspection_reports_cardinality_to_avoid_unusable_default_features():
    info = inspect_csv(CSV)
    columns = {column['name']: column for column in info['columns']}
    assert columns['group']['unique'] == 2
    assert columns['x']['unique'] == 50
