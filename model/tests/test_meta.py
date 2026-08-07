"""Tests for the shared model contract.

These guard the boundary between training and the two deployed runtimes. If the contract
drifts, the app keeps running and quietly returns wrong numbers, so it is checked here.
"""

from __future__ import annotations

import json

import pytest

from platevision import meta as m


def test_contract_loads_and_validates():
    contract = m.load_meta()
    assert contract["schema_version"] == 1


def test_input_is_uint8_nhwc_rgb():
    """Clients hand over raw camera pixels. Any drift here breaks both runtimes at once."""
    spec = m.load_meta()["input"]
    assert spec["dtype"] == "uint8"
    assert spec["layout"] == "NHWC"
    assert spec["channel_order"] == "RGB"
    assert spec["value_range"] == [0, 255]


def test_preprocessing_is_declared_in_graph():
    """The whole point of the design: clients do no preprocessing at all."""
    assert m.load_meta()["preprocessing"]["location"] == "in_graph"


def test_normalization_is_well_formed():
    mean, std = m.normalization()
    assert len(mean) == len(std) == 3
    assert all(s > 0 for s in std)


def test_quantiles_are_ascending_and_in_range():
    qs = m.quantiles()
    assert qs == sorted(qs)
    assert all(0.0 < q < 1.0 for q in qs)


def test_quantiles_are_symmetric_around_the_median():
    """The UI presents these as a centred interval, so they must actually be centred."""
    qs = m.quantiles()
    assert len(qs) == 3
    assert qs[1] == pytest.approx(0.5)
    assert qs[0] + qs[2] == pytest.approx(1.0)


def test_output_shape_matches_declared_targets_and_quantiles():
    spec = m.load_meta()["outputs"]["nutrition_quantiles"]
    assert spec["shape"][1] == len(m.target_keys())
    assert spec["shape"][2] == len(m.quantiles())


def test_target_keys_are_unique():
    keys = m.target_keys()
    assert len(keys) == len(set(keys))


def test_every_target_declares_a_unit():
    targets = m.load_meta()["outputs"]["nutrition_quantiles"]["targets"]
    assert all(t.get("unit") for t in targets)


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": 99},
        {"preprocessing": {"normalize": {"mean": [0.5, 0.5], "std": [0.5, 0.5, 0.5]}}},
        {"preprocessing": {"normalize": {"mean": [0.5] * 3, "std": [0.0, 0.5, 0.5]}}},
    ],
    ids=["bad-schema-version", "wrong-channel-count", "zero-std"],
)
def test_invalid_contracts_are_rejected(tmp_path, mutation):
    """Validation has to actually fail, not just exist."""
    contract = json.loads(m.META_PATH.read_text(encoding="utf-8"))
    for key, value in mutation.items():
        if isinstance(value, dict):
            contract[key].update(value)
        else:
            contract[key] = value

    bad = tmp_path / "model_meta.json"
    bad.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError):
        m.load_meta(bad)


def test_unsorted_quantiles_are_rejected(tmp_path):
    contract = json.loads(m.META_PATH.read_text(encoding="utf-8"))
    contract["outputs"]["nutrition_quantiles"]["quantiles"] = [0.95, 0.5, 0.05]

    bad = tmp_path / "model_meta.json"
    bad.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="ascending"):
        m.load_meta(bad)
