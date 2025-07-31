from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from starter.starter.ml.model import compute_model_metrics, train_model, inference
from sklearn.datasets import make_classification


def test_train_model():
    """
    test if the train_model func is called with the correct classifier
    """
    X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == X.shape[1]
    preds = model.predict(X)
    assert preds.shape == y.shape


@patch("starter.starter.ml.model.fbeta_score")
@patch("starter.starter.ml.model.recall_score")
@patch("starter.starter.ml.model.precision_score")
def test_compute_model_metrics_calls_metrics(mock_precision, mock_recall, mock_fbeta):
    """
    tests if the ml metrics are calculated correctly

    """
    y = np.array([1, 0, 1])
    preds = np.array([1, 0, 0])

    mock_precision.return_value = 0.8
    mock_recall.return_value = 0.6
    mock_fbeta.return_value = 0.7

    precision, recall, fbeta = compute_model_metrics(y, preds)

    mock_precision.assert_called_once_with(y, preds, zero_division=1)
    mock_recall.assert_called_once_with(y, preds, zero_division=1)
    mock_fbeta.assert_called_once_with(y, preds, beta=1, zero_division=1)

    assert precision == 0.8
    assert recall == 0.6
    assert fbeta == 0.7


def test_three():
    """
    tests if the predict functionality was called correctly and returns the predictions
    """
    mock_model = MagicMock()
    # mock x
    X = np.array([[1, 2], [3, 4]])
    # mock return value
    expected_preds = np.array([0, 1])
    mock_model.predict.return_value = expected_preds

    preds = inference(mock_model, X)

    mock_model.predict.assert_called_once_with(X)
    assert np.array_equal(preds, expected_preds)
