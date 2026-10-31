from webapp.datasets import dataset_info, predict_dataset
import pytest


def test_paris_and_student_have_separate_targets_and_predictions():
    for key, target in [('paris','price'), ('student','G3')]:
        info=dataset_info(key)
        assert info['target']==target
        assert len(info['features']) in (16,20)
        assert isinstance(predict_dataset(key,info['example'])['prediction'],float)


def test_student_category_and_invalid_numeric_are_rejected():
    info=dataset_info('student'); bad=info['example'].copy(); bad['sex']='invalid'
    with pytest.raises(ValueError): predict_dataset('student',bad)
    info=dataset_info('paris'); bad=info['example'].copy(); bad['squareMeters']='abc'
    with pytest.raises(ValueError): predict_dataset('paris',bad)
