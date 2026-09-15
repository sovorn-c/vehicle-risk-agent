"""Contracts for the frozen e12 held-out comparative split."""

import pytest

from vehicle_risk_agent.evaluation.comparative import (
    E12EvaluationConfig,
    get_e12_held_out_scenarios,
    load_e12_evaluation_config,
    validate_e12_split,
)


def test_e12_config_freezes_held_out_split_and_overlays() -> None:
    config = load_e12_evaluation_config()
    validate_e12_split(config)
    held_out = get_e12_held_out_scenarios(config)

    assert len(held_out) == 30
    assert len({scenario.scenario_id for scenario in held_out}) == 30
    assert config.provider == "gemini"
    assert config.model == "gemini-3.1-flash-lite"
    assert config.pricing_input_usd_per_million == 0.25
    assert config.pricing_output_usd_per_million == 1.5
    assert config.repeats == 3
    assert config.live_run_count == 24
    assert config.live_unique_comparable_scenarios == 4
    assert {item.scenario_id for item in config.comparable_shared_inputs} == {
        "sc-clean-01",
        "sc-risk-compound-05",
        "sc-conflict-ppsr-01",
        "sc-temporal-multi-rev-02",
    }
    assert not any(
        "odometer discrepancy" in item.investigation_question.lower()
        or "required vehicle evidence remains unresolved" in item.investigation_question.lower()
        for item in config.comparable_shared_inputs
    )
    assert not set(config.held_out_scenario_ids) & set(config.tuning_excluded_scenario_ids)
    compound = next(
        item
        for item in config.comparable_shared_inputs
        if item.scenario_id == "sc-risk-compound-05"
    )
    assert compound.retrieval_query_id == "q-08"
    live_labels = compound.live_expected_labels
    assert live_labels is not None
    assert live_labels.risk_band is not None
    assert live_labels.risk_band.value == "CRITICAL"
    assert live_labels.min_risk_score == 100
    assert set(live_labels.required_factor_ids) == {
        "LISTED",
        "MATCH",
        "STATUTORY",
    }


def test_e12_config_rejects_duplicate_held_out_ids() -> None:
    config = load_e12_evaluation_config()
    duplicated = list(config.held_out_scenario_ids)
    duplicated[-1] = duplicated[0]
    payload = config.model_dump(by_alias=True)
    payload["held_out_scenario_ids"] = tuple(duplicated)

    with pytest.raises(ValueError, match="unique"):
        E12EvaluationConfig.model_validate(payload)


def test_comparable_overlays_use_the_same_labelled_vehicle_fixtures() -> None:
    config = load_e12_evaluation_config()
    scenarios = {item.scenario_id: item for item in get_e12_held_out_scenarios(config)}

    assert {
        scenario_id: scenarios[scenario_id].vin
        for scenario_id in (
            "sc-clean-01",
            "sc-risk-compound-05",
            "sc-conflict-ppsr-01",
            "sc-temporal-multi-rev-02",
        )
    } == {
        "sc-clean-01": "1HGCR2F85HA000000",
        "sc-risk-compound-05": "1FA6P8CF8H5000000",
        "sc-conflict-ppsr-01": "WAUZZZ8K7BA000000",
        "sc-temporal-multi-rev-02": "1HGCR2F85HA000000",
    }

    for overlay in config.comparable_shared_inputs:
        live_labels = overlay.live_expected_labels
        if live_labels is not None:
            assert scenarios[overlay.scenario_id].expected_labels == live_labels
