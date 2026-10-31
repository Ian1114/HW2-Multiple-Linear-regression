import pytest
from webapp.server import predict_record, example_record


def test_example_produces_model_prediction():
    example = example_record()
    result = predict_record(example)
    assert 0 < result['prediction'] < 120


def test_missing_feature_is_reported():
    example = example_record()
    example.pop('Year')
    with pytest.raises(ValueError, match='Year'):
        predict_record(example)


def test_invalid_value_is_rejected():
    example = example_record()
    example['GDP'] = 'not a number'
    with pytest.raises(ValueError, match='GDP'):
        predict_record(example)


def test_example_is_json_serializable():
    import json
    json.dumps(example_record(), allow_nan=False)
