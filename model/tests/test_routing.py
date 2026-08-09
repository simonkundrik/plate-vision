from __future__ import annotations

import pytest
import torch

from platevision import routing

KEYS = ["energy"]
QUANTILES = [0.05, 0.5, 0.95]


def estimates(truth: torch.Tensor, error: float, half_width: float) -> torch.Tensor:
    """Predictions offset from the truth by a fixed amount, shape (n, 1, 3)."""
    median = truth + error
    return torch.stack([median - half_width, median, median + half_width], dim=-1).unsqueeze(1)


class TestAssignRoutes:
    def test_picks_the_best_evidence_available(self):
        assigned = routing.assign_routes([{"absolute", "barcode"}, {"absolute"}])
        assert assigned == ["barcode", "absolute"]

    def test_a_meal_with_no_evidence_still_gets_an_answer(self):
        # The common case. Nothing scanned, no chain matched, no plate detected.
        assert routing.assign_routes([set()]) == ["absolute"]

    def test_chain_menu_beats_scale_reference(self):
        # Ordering is the routing policy, not a display preference: a published menu figure
        # is stated, a recovered scale is inferred.
        assigned = routing.assign_routes([{"scale_reference", "chain_menu"}])
        assert assigned == ["chain_menu"]

    def test_rejects_a_route_name_it_does_not_know(self):
        # A typo would otherwise fall through to the fallback silently, and a barcode result
        # would be counted as the vision model's. Failing loudly is the whole point.
        with pytest.raises(ValueError, match="unknown routes"):
            routing.assign_routes([{"barcodes"}])

    def test_rejects_a_fallback_outside_the_priority_order(self):
        with pytest.raises(ValueError, match="not one of the priority routes"):
            routing.assign_routes([set()], fallback="guess")


class TestRoutedReport:
    """A near-exact route and a mediocre one, evaluated together."""

    @pytest.fixture
    def split(self):
        truth = torch.arange(1.0, 101.0).unsqueeze(1) * 5.0
        # First 20 meals were scanned; the rest fell through to the photograph.
        routes = ["barcode"] * 20 + ["absolute"] * 80
        predictions = torch.cat(
            [estimates(truth[:20, 0], 2.0, 5.0), estimates(truth[20:, 0], 60.0, 40.0)]
        )
        return predictions, truth, routes

    def test_a_good_route_does_not_flatter_a_bad_one(self, split):
        # The reason this module exists. Blended, these are one number near 48 kcal; split,
        # the vision model's 60 is visible.
        report = routing.build_routed_report(*split, target_keys=KEYS, quantiles=QUANTILES)
        by_route = {outcome.route: outcome for outcome in report.routes}

        assert by_route["barcode"].report.mae["energy"] == pytest.approx(2.0)
        assert by_route["absolute"].report.mae["energy"] == pytest.approx(60.0)

    def test_shares_account_for_every_meal(self, split):
        report = routing.build_routed_report(*split, target_keys=KEYS, quantiles=QUANTILES)
        assert sum(outcome.used_share for outcome in report.routes) == pytest.approx(1.0)
        assert sum(outcome.used for outcome in report.routes) == report.total

    def test_the_blended_number_sits_between_the_routes(self, split):
        report = routing.build_routed_report(*split, target_keys=KEYS, quantiles=QUANTILES)
        blended = report.as_routed.mae["energy"]
        assert 2.0 < blended < 60.0
        # 20 meals at 2 kcal and 80 at 60 kcal.
        assert blended == pytest.approx(0.2 * 2.0 + 0.8 * 60.0)

    def test_the_headline_states_a_share_for_every_route(self, split):
        # Accuracy without coverage is the lie this module was written to prevent, so the
        # honesty property is asserted rather than left to whoever reads the output.
        report = routing.build_routed_report(*split, target_keys=KEYS, quantiles=QUANTILES)
        lines = report.headline()

        for outcome in report.routes:
            line = next(line for line in lines if line.strip().startswith(outcome.route))
            assert f"{outcome.used_share * 100:.1f}%" in line

    def test_the_headline_blends_only_when_there_is_something_to_blend(self, split):
        predictions, truth, _ = split
        one_route = routing.build_routed_report(
            predictions, truth, ["absolute"] * 100, target_keys=KEYS, quantiles=QUANTILES
        )
        # With a single route the blended row restates the line above it, and a repeated
        # number reads as corroboration rather than the same measurement twice.
        assert not any("all meals" in line for line in one_route.headline())
        assert any(
            "all meals" in line
            for line in routing.build_routed_report(
                *split, target_keys=KEYS, quantiles=QUANTILES
            ).headline()
        )

    def test_best_route_first_regardless_of_input_order(self, split):
        predictions, truth, routes = split
        order = torch.randperm(100, generator=torch.Generator().manual_seed(0))
        shuffled = routing.build_routed_report(
            predictions[order],
            truth[order],
            [routes[i] for i in order.tolist()],
            target_keys=KEYS,
            quantiles=QUANTILES,
        )
        assert [outcome.route for outcome in shuffled.routes] == ["barcode", "absolute"]

    def test_metrics_survive_shuffling(self, split):
        predictions, truth, routes = split
        order = torch.randperm(100, generator=torch.Generator().manual_seed(1))
        shuffled = routing.build_routed_report(
            predictions[order],
            truth[order],
            [routes[i] for i in order.tolist()],
            target_keys=KEYS,
            quantiles=QUANTILES,
        )
        by_route = {outcome.route: outcome for outcome in shuffled.routes}
        assert by_route["barcode"].report.mae["energy"] == pytest.approx(2.0)
        assert by_route["barcode"].used == 20

    def test_dish_ids_follow_their_own_rows(self, split):
        predictions, truth, routes = split
        ids = [f"dish_{i:03d}" for i in range(100)]
        report = routing.build_routed_report(
            predictions, truth, routes, target_keys=KEYS, quantiles=QUANTILES, dish_ids=ids
        )
        by_route = {outcome.route: outcome for outcome in report.routes}

        # The worst barcode failures must be barcode dishes. Getting this wrong would put
        # another route's images in front of someone debugging this one.
        worst = by_route["barcode"].report.worst
        assert worst
        assert all(int(case.dish_id.removeprefix("dish_")) < 20 for case in worst)


class TestCoverageVersusUse:
    def test_a_preempted_route_reports_what_it_could_have_covered(self):
        # Both fired on the same meal, barcode won. Scale reference was still available, and
        # "how often could we use it" is a different question from "how often did we".
        availability = [{"barcode", "scale_reference"}, {"scale_reference"}, set()]
        routes = routing.assign_routes(availability)
        truth = torch.tensor([[100.0], [200.0], [300.0]])

        report = routing.build_routed_report(
            estimates(truth[:, 0], 5.0, 10.0),
            truth,
            routes,
            target_keys=KEYS,
            quantiles=QUANTILES,
            availability=availability,
        )
        scale = next(o for o in report.routes if o.route == "scale_reference")

        assert scale.used == 1
        assert scale.available == 2
        assert scale.available_share == pytest.approx(2 / 3)


class TestReliability:
    def test_a_thin_route_is_flagged_rather_than_hidden(self):
        # Median percentage error over three dishes describes three dishes. Reporting it is
        # fine; quoting it as a measured route accuracy is not.
        truth = torch.tensor([[100.0], [200.0], [300.0], [400.0]])
        report = routing.build_routed_report(
            estimates(truth[:, 0], 1.0, 5.0),
            truth,
            ["barcode"] * 3 + ["absolute"],
            target_keys=KEYS,
            quantiles=QUANTILES,
        )
        assert all(not outcome.reliable for outcome in report.routes)
        assert any("too few to quote" in line for line in report.headline())

    def test_a_route_becomes_quotable_once_it_has_enough_meals(self):
        n = routing.MIN_RELIABLE_SAMPLES
        truth = torch.arange(1.0, n + 1.0).unsqueeze(1)
        report = routing.build_routed_report(
            estimates(truth[:, 0], 1.0, 5.0),
            truth,
            ["barcode"] * n,
            target_keys=KEYS,
            quantiles=QUANTILES,
        )
        assert report.routes[0].reliable


class TestGuards:
    def test_refuses_estimates_with_no_route_label(self):
        # An unlabelled estimate is exactly the thing this module forbids: a number whose
        # provenance nobody can state.
        truth = torch.tensor([[100.0], [200.0]])
        with pytest.raises(ValueError, match="every meal needs one"):
            routing.build_routed_report(
                estimates(truth[:, 0], 1.0, 5.0),
                truth,
                ["barcode"],
                target_keys=KEYS,
                quantiles=QUANTILES,
            )

    def test_refuses_mismatched_availability(self):
        truth = torch.tensor([[100.0], [200.0]])
        with pytest.raises(ValueError, match="availability sets"):
            routing.build_routed_report(
                estimates(truth[:, 0], 1.0, 5.0),
                truth,
                ["barcode", "absolute"],
                target_keys=KEYS,
                quantiles=QUANTILES,
                availability=[{"barcode"}],
            )

    def test_an_unrecognised_route_is_reported_rather_than_dropped(self):
        # Priority ordering does not know about it, so it sorts last, but a meal handled by
        # something unexpected must not vanish from the accounting.
        truth = torch.tensor([[100.0], [200.0]])
        report = routing.build_routed_report(
            estimates(truth[:, 0], 1.0, 5.0),
            truth,
            ["absolute", "experimental"],
            target_keys=KEYS,
            quantiles=QUANTILES,
        )
        assert [outcome.route for outcome in report.routes] == ["absolute", "experimental"]
        assert report.total == 2
