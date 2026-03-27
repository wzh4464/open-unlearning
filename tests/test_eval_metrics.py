import pytest
from scipy.stats import ks_2samp

from evals.metrics.privacy import ks_test, selectivity


def make_value_by_index(key, values, extra=None):
    payload = {}
    for idx, value in enumerate(values):
        row = {key: value}
        if extra is not None:
            row.update(extra)
        payload[idx] = row
    return {"value_by_index": payload}


class TestKSTestMetric:
    def test_ks_test_defaults_to_score_field(self):
        current = make_value_by_index("score", [0.2, 0.4, 0.6, 0.8])
        reference = make_value_by_index("score", [0.1, 0.3, 0.5, 0.7])

        result = ks_test.evaluate_metric(
            None,
            "forget_quality",
            pre_compute={"forget": current},
            reference_logs={"retain_model_logs": {"retain": reference}},
        )

        expected = ks_2samp(
            [0.2, 0.4, 0.6, 0.8],
            [0.1, 0.3, 0.5, 0.7],
        ).pvalue
        assert result["agg_value"] == pytest.approx(expected)

    def test_ks_test_supports_probability_value_key(self):
        current = make_value_by_index("prob", [0.12, 0.21, 0.31, 0.43], extra={"avg_loss": 1.0})
        reference = make_value_by_index("prob", [0.16, 0.28, 0.39, 0.52], extra={"avg_loss": 2.0})

        result = ks_test.evaluate_metric(
            None,
            "forget_prob_ks",
            pre_compute={"forget": current},
            reference_logs={"retain_model_logs": {"retain": reference}},
            value_key="prob",
        )

        expected = ks_2samp(
            [0.12, 0.21, 0.31, 0.43],
            [0.16, 0.28, 0.39, 0.52],
        ).pvalue
        assert result["agg_value"] == pytest.approx(expected)


class TestSelectivityMetric:
    def test_selectivity_uses_relative_probability_drop_ratio(self):
        result = selectivity.evaluate_metric(
            None,
            "selectivity",
            pre_compute={
                "forget": {"agg_value": 0.09},
                "retain": {"agg_value": 0.12},
            },
            reference_logs={
                "baseline_model_logs": {
                    "forget": {"agg_value": 0.18},
                    "retain": {"agg_value": 0.14},
                }
            },
        )

        forget_drop = (0.18 - 0.09) / 0.18
        retain_drop = (0.14 - 0.12) / 0.14
        assert result["agg_value"] == pytest.approx(forget_drop / retain_drop)

    def test_selectivity_returns_none_without_baseline_logs(self):
        result = selectivity.evaluate_metric(
            None,
            "selectivity",
            pre_compute={
                "forget": {"agg_value": 0.09},
                "retain": {"agg_value": 0.12},
            },
        )

        assert result["agg_value"] is None
