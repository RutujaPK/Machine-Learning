import numpy as np
from .test_utils import make_fake_data

def test_accuracy():
    from sklearn.metrics import accuracy_score
    from src import compute_accuracy

    y_true, y_pred = make_fake_data()
    _actual = accuracy_score(y_true, y_pred)
    _est = compute_accuracy(y_true, y_pred)
    assert np.allclose(_actual, _est)

def test_f1_measure():
    from sklearn.metrics import f1_score
    from src import compute_f1_measure

    y_true, y_pred = make_fake_data()
    _actual = f1_score(y_true, y_pred)
    _est = compute_f1_measure(y_true, y_pred)
    assert np.allclose(_actual, _est)
    assert np.isnan(compute_f1_measure(np.zeros(3), np.zeros(3)))

def test_precision_and_recall():
    from sklearn.metrics import precision_score, recall_score
    from src import compute_precision_and_recall

    y_true, y_pred = make_fake_data()
    _actual = [precision_score(y_true, y_pred), recall_score(y_true, y_pred)]
    _est = compute_precision_and_recall(y_true, y_pred)
    assert np.allclose(_actual, _est)

def test_confusion_matrix():
    from sklearn.metrics import confusion_matrix as ref_confusion_matrix
    from src import compute_confusion_matrix

    y_true, y_pred = make_fake_data()
    _actual = ref_confusion_matrix(y_true, y_pred)
    _est = compute_confusion_matrix(y_true, y_pred)
    assert np.allclose(_actual, _est)
