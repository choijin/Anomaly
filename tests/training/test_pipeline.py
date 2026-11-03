from anomaly_detection.config import load_config
from anomaly_detection.training.pipeline import select_model


def test_validation_selection_is_deterministic_and_does_not_take_test_metrics():
    config = load_config()
    base = {
        "event_recall": 1.0,
        "unmatched_alerts_per_day": 0.2,
        "events_total": 3,
        "inference_ms_per_row": 1.0,
    }
    results = {
        name: {"validation_metrics": dict(base)}
        for name in ["if_raw", "if_temporal", "cnn_autoencoder"]
    }
    assert select_model(results, config) == "if_raw"
    results["if_raw"]["test_metrics"] = {"event_recall": 0.0}
    results["cnn_autoencoder"]["test_metrics"] = {"event_recall": 1.0}
    assert select_model(results, config) == "if_raw"
    results["if_temporal"]["validation_metrics"]["unmatched_alerts_per_day"] = 0.1
    assert select_model(results, config) == "if_temporal"


def test_training_requires_explicit_source_metadata(observations, tmp_path):
    import pytest

    from anomaly_detection.training.pipeline import train_and_evaluate

    with pytest.raises(ValueError, match="explicit data_source and source_sha256"):
        train_and_evaluate(observations, load_config(), tmp_path, source_metadata={})
