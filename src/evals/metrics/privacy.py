import numpy as np
from scipy.stats import ks_2samp
from evals.metrics.base import unlearning_metric, logger


def _get_reference_group(reference_logs, preferred_key):
    if not reference_logs:
        return None
    if preferred_key in reference_logs:
        return reference_logs[preferred_key]
    if len(reference_logs) == 1:
        return next(iter(reference_logs.values()))
    logger.warning(
        "Reference logs has multiple entries but none match '%s', returning None",
        preferred_key,
    )
    return None


def _extract_distribution(metric_result, value_key=None):
    values = []
    skipped = 0
    value_by_index = metric_result["value_by_index"]
    for evals in value_by_index.values():
        if evals is None:
            continue
        if value_key is not None:
            value = evals.get(value_key)
            if value is None and value_key not in evals:
                skipped += 1
        elif "score" in evals:
            value = evals["score"]
        else:
            numeric_items = [
                value
                for value in evals.values()
                if isinstance(value, (int, float, np.integer, np.floating))
            ]
            if len(numeric_items) != 1:
                raise KeyError(
                    "Could not infer which scalar field to compare. "
                    "Pass value_key explicitly."
                )
            value = numeric_items[0]
        if value is not None:
            values.append(value)
    if skipped > 0:
        logger.warning(
            "%d entries missing key '%s' were skipped in _extract_distribution",
            skipped,
            value_key,
        )
    return np.array(values)


@unlearning_metric(name="ks_test")
def ks_test(model, **kwargs):
    """Compare two scalar per-example distributions with a 2-sample KS-test.

    This is used for TOFU forget_quality when the compared statistic is truth_ratio,
    but it also supports probability-based comparisons when value_key is provided.
    """
    current_key = kwargs.get("current_key", "forget")
    reference_key = kwargs.get("reference_key", "retain")
    value_key = kwargs.get("value_key", None)

    current_stats = _extract_distribution(
        kwargs["pre_compute"][current_key], value_key=value_key
    )
    reference_group = _get_reference_group(
        kwargs.get("reference_logs", None), preferred_key="retain_model_logs"
    )
    if not reference_group:
        logger.warning(
            "reference_logs not provided for ks_test (expected 'retain_model_logs' "
            "containing '%s'), setting result to None",
            reference_key,
        )
        pvalue = None
    elif reference_key not in reference_group or reference_group[reference_key] is None:
        logger.warning(
            "reference_logs present but missing key '%s' in retain_model_logs, "
            "setting ks_test result to None",
            reference_key,
        )
        pvalue = None
    else:
        reference_stats = _extract_distribution(
            reference_group[reference_key], value_key=value_key
        )
        if len(current_stats) == 0 or len(reference_stats) == 0:
            logger.warning(
                "One or both distributions are empty for ks_test, setting result to None"
            )
            pvalue = None
        else:
            pvalue = ks_2samp(current_stats, reference_stats).pvalue
    return {"agg_value": pvalue}


@unlearning_metric(name="privleak")
def privleak(model, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic.
    To be used for MIA AUC scores in ensuring consistency and reproducibility of the MUSE benchmark.
    This function is similar to the rel_diff function below, but due to the MUSE benchmark reporting AUC
    scores as (1-x) when the more conventional way is x, we do adjustments here to our MIA AUC scores.
    calculations in the reverse way,"""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    try:
        ref = kwargs["reference_logs"]["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            f"retain_model_logs evals not provided for privleak, using default retain auc of {kwargs['ref_value']}"
        )
        ref = kwargs["ref_value"]
    score = 1 - score
    ref = 1 - ref
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}


@unlearning_metric(name="rel_diff")
def rel_diff(model, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic."""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    try:
        ref = kwargs["reference_logs"]["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            f"retain_model_logs evals not provided for privleak, using default retain auc of {kwargs['ref_value']}"
        )
        ref = kwargs["ref_value"]
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}


@unlearning_metric(name="selectivity")
def selectivity(model, **kwargs):
    """Measure how much more the forget probability drops than the retain probability."""
    forget_key = kwargs.get("forget_key", "forget")
    retain_key = kwargs.get("retain_key", "retain")
    eps = kwargs.get("eps", 1e-10)

    reference_group = _get_reference_group(
        kwargs.get("reference_logs", None), preferred_key="baseline_model_logs"
    )
    if reference_group is None:
        logger.warning(
            "baseline_model_logs not provided in reference_logs, setting selectivity to None"
        )
        return {"agg_value": None}

    try:
        current_forget = kwargs["pre_compute"][forget_key]["agg_value"]
        current_retain = kwargs["pre_compute"][retain_key]["agg_value"]
        reference_forget = reference_group[forget_key]["agg_value"]
        reference_retain = reference_group[retain_key]["agg_value"]
    except KeyError as exc:
        logger.warning(
            "Required metric '%s' missing in pre_compute or reference_logs, "
            "setting selectivity to None",
            exc,
        )
        return {"agg_value": None}

    raw_values = {
        "current_forget": current_forget,
        "current_retain": current_retain,
        "reference_forget": reference_forget,
        "reference_retain": reference_retain,
    }
    invalid = []
    for name, v in raw_values.items():
        if v is None:
            invalid.append(name)
        elif not np.isscalar(v):
            invalid.append(name)
        else:
            try:
                if np.isnan(v):
                    invalid.append(name)
            except (TypeError, ValueError):
                invalid.append(name)
    if invalid:
        logger.warning(
            "Required metrics contain None, NaN, or non-scalar (%s), setting selectivity to None",
            ", ".join(invalid),
        )
        return {"agg_value": None}

    current_forget = float(current_forget)
    current_retain = float(current_retain)
    reference_forget = float(reference_forget)
    reference_retain = float(reference_retain)

    forget_drop = (reference_forget - current_forget) / (abs(reference_forget) + eps)
    retain_drop = (reference_retain - current_retain) / (abs(reference_retain) + eps)

    if abs(retain_drop) <= eps:
        logger.warning("retain drop is too small to compute selectivity reliably")
        return {"agg_value": None}

    return {"agg_value": forget_drop / retain_drop}
