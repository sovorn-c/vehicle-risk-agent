"""Contracts for the frozen e12 held-out comparative split."""

from vehicle_risk_agent.evaluation.comparative import (
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
    assert config.repeats == 3
    assert config.live_run_count == 24
    assert config.live_unique_comparable_scenarios == 4
    assert {item.scenario_id for item in config.comparable_shared_inputs} == {
        "sc-clean-01",
        "sc-risk-statutory-04",
        "sc-conflict-ppsr-01",
        "sc-temporal-multi-rev-02",
    }
    assert not any(
        "odometer discrepancy" in item.investigation_question.lower()
        or "required vehicle evidence remains unresolved" in item.investigation_question.lower()
        for item in config.comparable_shared_inputs
    )
    assert not set(config.held_out_scenario_ids) & set(config.tuning_excluded_scenario_ids)
