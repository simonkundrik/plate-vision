"""Per-route accounting for nutrition estimates.

A photograph is not always the best evidence available. When the calories are printed on a
wrapper, reading the wrapper beats estimating from pixels, and the resulting error is nothing
like the error of the vision model. Averaging the two into one accuracy figure would let a
barcode route at 2% hide a vision model at 19%, so this module makes that averaging
impossible to do silently.

Two rules, both enforced here rather than left to the caller's discretion:

1. **Every estimate declares the route that produced it.** ``build_routed_report`` takes a
   route label per sample and refuses to run without one.
2. **Accuracy is never reported without coverage.** ``RoutedReport.headline`` prints the
   share of meals each route handled next to how well it did on them, because "near exact"
   is only interesting alongside "on 4% of meals".

The blended figure is still computed, under the name ``as_routed``, because it does answer a
real question: what a user experiences across a whole day of meals. It is the *headline that
omits the shares* that is dishonest, not the number itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from .evaluation import NutritionReport, build_report

# Ordered best evidence first. A meal is handled by the first route that is available for it,
# so this ordering is the routing policy, not a display preference. It follows the expected
# accuracy of each route: a stated composition beats an inferred one, and a recovered scale
# beats no scale at all.
ROUTE_PRIORITY: tuple[str, ...] = (
    "barcode",
    "chain_menu",
    "scale_reference",
    "absolute",
)

# The route that needs no evidence beyond the photograph, and therefore always applies.
FALLBACK_ROUTE = "absolute"

# Below this many meals, a route's median percentage error is describing a handful of dishes
# rather than the route. Reported anyway, because hiding a thin result is its own dishonesty,
# but flagged so it cannot be quoted as though it were measured.
MIN_RELIABLE_SAMPLES = 30


def assign_routes(
    availability: list[set[str]],
    priority: tuple[str, ...] = ROUTE_PRIORITY,
    fallback: str = FALLBACK_ROUTE,
) -> list[str]:
    """Pick the best available route for each meal.

    ``availability[i]`` is the set of routes that have the evidence they need for meal i: a
    barcode was scanned, GPS matched a chain, a plate was detected. The fallback covers the
    meal where none of them fired, which is the common case.
    """
    if fallback not in priority:
        raise ValueError(f"fallback {fallback!r} is not one of the priority routes {priority}")

    unknown = {route for routes in availability for route in routes} - set(priority)
    if unknown:
        raise ValueError(f"unknown routes {sorted(unknown)}; expected some of {list(priority)}")

    assigned: list[str] = []
    for routes in availability:
        assigned.append(next((route for route in priority if route in routes), fallback))
    return assigned


@dataclass(slots=True)
class RouteOutcome:
    """How one route performed, and how much of the problem it actually covered."""

    route: str
    used: int
    used_share: float
    # How often the route *could* have fired, which differs from how often it was used when a
    # higher-priority route preempted it. "How often could we use the good route" is a
    # separate question from "how good is it when we can", and both belong in the answer.
    available: int
    available_share: float
    reliable: bool
    report: NutritionReport


@dataclass(slots=True)
class RoutedReport:
    total: int
    routes: list[RouteOutcome] = field(default_factory=list)
    as_routed: NutritionReport | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def headline(self, key: str = "energy") -> list[str]:
        """The report as lines, shares first, so accuracy cannot be quoted without them.

        Returns lines rather than a single string because the shares are the point: a caller
        that wants one number has to reach past this into ``as_routed`` deliberately.
        """
        lines = [f"{key}, as routed over {self.total:,} meals"]
        for outcome in self.routes:
            note = "" if outcome.reliable else f"  (only {outcome.used}, too few to quote)"
            lines.append(
                f"  {outcome.route:<16} {outcome.used_share * 100:5.1f}% of meals   "
                f"MAE {outcome.report.mae[key]:7.1f}   "
                f"median APE {outcome.report.median_ape[key] * 100:5.1f}%   "
                f"coverage {outcome.report.coverage[key] * 100:5.1f}%{note}"
            )
        # The blended row is only worth printing when there is something to blend. With one
        # route it restates the line above it, and a repeated number reads as corroboration.
        if self.as_routed is not None and len(self.routes) > 1:
            lines.append(
                f"  {'all meals':<16} {'':18}"
                f"MAE {self.as_routed.mae[key]:7.1f}   "
                f"median APE {self.as_routed.median_ape[key] * 100:5.1f}%   "
                f"coverage {self.as_routed.coverage[key] * 100:5.1f}%"
            )
        return lines


def build_routed_report(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    routes: list[str],
    *,
    target_keys: list[str],
    quantiles: list[float],
    dish_ids: list[str] | None = None,
    availability: list[set[str]] | None = None,
    crossing_rate: float = 0.0,
    min_samples: int = MIN_RELIABLE_SAMPLES,
    top_n: int = 10,
) -> RoutedReport:
    """Break a set of estimates down by the route that produced each one.

    ``predictions`` are already in real units, one row per meal, produced by whichever route
    ``routes[i]`` names. Routes that predict nothing for a meal simply do not appear in that
    row, so a route covering 4% of meals contributes 4% of the rows rather than a column of
    padding that would have to be masked out later.
    """
    total = int(targets.shape[0])
    if len(routes) != total:
        raise ValueError(f"{len(routes)} route labels for {total} meals; every meal needs one")
    if predictions.shape[0] != total:
        raise ValueError(f"{predictions.shape[0]} predictions for {total} targets")
    if availability is not None and len(availability) != total:
        raise ValueError(f"{len(availability)} availability sets for {total} meals")

    seen = _ordered_routes(routes)
    position_of = {route: index for index, route in enumerate(seen)}
    labels = torch.tensor([position_of[route] for route in routes], dtype=torch.long)

    outcomes: list[RouteOutcome] = []
    for position, route in enumerate(seen):
        mask = labels == position
        count = int(mask.sum().item())
        if count == 0:
            continue

        available = (
            sum(1 for routes_for_meal in availability if route in routes_for_meal)
            if availability is not None
            else count
        )
        selected = mask.nonzero(as_tuple=True)[0].tolist()

        outcomes.append(
            RouteOutcome(
                route=route,
                used=count,
                used_share=count / total if total else 0.0,
                available=available,
                available_share=available / total if total else 0.0,
                reliable=count >= min_samples,
                report=build_report(
                    predictions[mask],
                    targets[mask],
                    target_keys=target_keys,
                    quantiles=quantiles,
                    dish_ids=[dish_ids[i] for i in selected] if dish_ids else None,
                    crossing_rate=crossing_rate,
                    top_n=top_n,
                ),
            )
        )

    return RoutedReport(
        total=total,
        routes=outcomes,
        as_routed=build_report(
            predictions,
            targets,
            target_keys=target_keys,
            quantiles=quantiles,
            dish_ids=dish_ids,
            crossing_rate=crossing_rate,
            top_n=top_n,
        )
        if total
        else None,
    )


def _ordered_routes(routes: list[str]) -> list[str]:
    """Routes present, best evidence first, with any unrecognised ones after."""
    present = set(routes)
    known = [route for route in ROUTE_PRIORITY if route in present]
    return known + sorted(present - set(known))
