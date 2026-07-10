"""Synthetic ranking data for development when real datasets are unavailable.

The generator is RecTour-like, not RecTour-derived: it uses only public field
descriptions and hand-written behavioral assumptions. It is suitable for local
pipeline development, schema tests, and latency/quality plumbing, but not for
claims about Expedia's real traffic distribution.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from learning_to_rank_distillation.schema import RankingExample

RECTOUR_LIKE_FEATURE_COLUMNS: tuple[str, ...] = (
    "checkin_date",
    "checkout_date",
    "adult_count",
    "child_count",
    "length_of_stay",
    "booking_window",
    "destination_id",
    "point_of_sale",
    "geo_location_country",
    "is_mobile",
    "sort_type",
    "applied_filters",
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "srch_room_count",
    "orig_destination_distance",
    "prop_country_id",
    "prop_id",
    "prop_brand_bool",
    "star_rating",
    "review_rating",
    "review_count",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "price_bucket",
    "is_free_cancellation",
    "is_travel_ad",
    "is_drr",
    "promotion_flag",
    "competitor_price_delta",
    "competitor_availability_rate",
    "marketplace_supply_segment",
    "is_cold_start_supply",
    "exposure_skew_score",
    "price_rank",
    "star_rank",
    "location_score_rank",
    "price_diff_from_query_median",
    "historical_price_gap",
    "number_of_amenities",
    "free_parking",
    "parking",
    "gym",
    "swimming_pool",
    "washer_dryer",
    "spa_services",
    "free_airport_transportation",
)

_AMENITY_COLUMNS = (
    "free_parking",
    "parking",
    "gym",
    "swimming_pool",
    "washer_dryer",
    "spa_services",
    "free_airport_transportation",
)
_SORT_TYPES = ("recommended", "price", "star_rating", "review_rating")
_POINTS_OF_SALE = ("US", "CA", "GB", "DE", "FR", "CH")
_COUNTRIES = ("US", "CA", "GB", "DE", "FR", "CH", "IN", "AU")


@dataclass(frozen=True, slots=True)
class SyntheticMarketplaceConfig:
    """Controls for stress-testing marketplace ranking behavior."""

    num_queries: int = 24
    items_per_query: int = 8
    seed: int = 13
    num_destinations: int = 5
    num_properties: int | None = None
    supply_concentration: float = 0.0
    cold_start_rate: float = 0.0
    exposure_skew: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("supply_concentration", self.supply_concentration),
            ("cold_start_rate", self.cold_start_rate),
            ("exposure_skew", self.exposure_skew),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def make_synthetic_rectour_rows(
    *,
    num_queries: int = 24,
    items_per_query: int = 8,
    seed: int = 13,
    num_destinations: int = 5,
    num_properties: int | None = None,
) -> list[dict[str, Any]]:
    """Create deterministic, source-shaped RecTour-like rows.

    The rows intentionally include both raw behavioral columns (`num_clicks`,
    `is_trans`) and canonical adapter columns (`query_id`, `item_id`, `label`,
    `is_unbiased`, `position`) so they can be inspected directly or converted
    into `RankingExample` without guessing a real Expedia schema.
    """

    if num_queries < 2:
        raise ValueError("num_queries must be at least 2")
    if items_per_query < 2:
        raise ValueError("items_per_query must be at least 2")
    if num_destinations < 1:
        raise ValueError("num_destinations must be at least 1")

    property_count = num_properties or max(num_destinations * items_per_query, items_per_query * 2)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    base_date = date(2021, 4, 1)

    for query_index in range(num_queries):
        query_rng = random.Random(rng.randint(0, 2**31 - 1))
        destination_id = f"dest-{query_index % num_destinations:03d}"
        point_of_sale = _POINTS_OF_SALE[query_index % len(_POINTS_OF_SALE)]
        geo_country = _COUNTRIES[(query_index * 3) % len(_COUNTRIES)]
        adult_count = 1 + query_rng.randrange(4)
        child_count = query_rng.choices((0, 1, 2), weights=(0.75, 0.18, 0.07), k=1)[0]
        length_of_stay = 1 + query_rng.randrange(8)
        booking_window = int(query_rng.expovariate(1 / 18))
        booking_window = min(90, booking_window)
        checkin_date = base_date + timedelta(days=query_index % 56 + booking_window)
        checkout_date = checkin_date + timedelta(days=length_of_stay)
        is_mobile = query_rng.random() < 0.42
        sort_type = query_rng.choices(_SORT_TYPES, weights=(0.70, 0.12, 0.10, 0.08), k=1)[0]
        requested_filters = _sample_filters(query_rng)
        is_unbiased = query_index % 3 == 0
        query_budget = 2 + ((query_index + adult_count + child_count) % 4)
        room_count = 1 + (1 if adult_count + child_count >= 5 and query_rng.random() < 0.45 else 0)
        visitor_hist_starrating = (
            round(max(1.0, min(5.0, query_rng.gauss(3.6, 0.7))), 2)
            if query_rng.random() < 0.42
            else None
        )
        visitor_hist_adr_usd = (
            round(query_rng.lognormvariate(4.65, 0.38), 2) if query_rng.random() < 0.45 else None
        )
        orig_destination_distance = round(query_rng.lognormvariate(6.2, 1.0), 2)

        query_rows = _make_query_rows(
            query_id=f"q-{query_index:05d}",
            query_index=query_index,
            items_per_query=items_per_query,
            property_count=property_count,
            destination_id=destination_id,
            point_of_sale=point_of_sale,
            geo_country=geo_country,
            adult_count=adult_count,
            child_count=child_count,
            length_of_stay=length_of_stay,
            booking_window=booking_window,
            checkin_date=checkin_date.isoformat(),
            checkout_date=checkout_date.isoformat(),
            is_mobile=is_mobile,
            sort_type=sort_type,
            requested_filters=requested_filters,
            query_budget=query_budget,
            room_count=room_count,
            visitor_hist_starrating=visitor_hist_starrating,
            visitor_hist_adr_usd=visitor_hist_adr_usd,
            orig_destination_distance=orig_destination_distance,
            is_unbiased=is_unbiased,
            rng=query_rng,
        )
        rows.extend(query_rows)

    return rows


def write_synthetic_rectour_csv(
    output_path: Path,
    *,
    num_queries: int = 24,
    items_per_query: int = 8,
    seed: int = 13,
) -> Path:
    """Write a RecTour-like synthetic CSV for manual inspection or demos."""

    rows = make_synthetic_rectour_rows(
        num_queries=num_queries,
        items_per_query=items_per_query,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def make_synthetic_ranking_data(
    *,
    num_queries: int = 24,
    items_per_query: int = 8,
    seed: int = 13,
) -> list[RankingExample]:
    """Create deterministic examples matching the RankingExample contract.

    Implements the AGENTS.md fallback path for an empty data/rectour directory.
    """

    rows = make_synthetic_rectour_rows(
        num_queries=num_queries,
        items_per_query=items_per_query,
        seed=seed,
    )
    return [_row_to_ranking_example(row) for row in rows]


def make_configurable_marketplace_rows(
    config: SyntheticMarketplaceConfig,
) -> list[dict[str, Any]]:
    """Create synthetic rows with explicit marketplace stress-test controls."""

    rows = make_synthetic_rectour_rows(
        num_queries=config.num_queries,
        items_per_query=config.items_per_query,
        seed=config.seed,
        num_destinations=config.num_destinations,
        num_properties=config.num_properties,
    )
    _apply_marketplace_controls(rows, config)
    return rows


def make_configurable_marketplace_ranking_data(
    config: SyntheticMarketplaceConfig,
) -> list[RankingExample]:
    """Create `RankingExample` rows from configurable marketplace controls."""

    return [_row_to_ranking_example(row) for row in make_configurable_marketplace_rows(config)]


def _make_query_rows(
    *,
    query_id: str,
    query_index: int,
    items_per_query: int,
    property_count: int,
    destination_id: str,
    point_of_sale: str,
    geo_country: str,
    adult_count: int,
    child_count: int,
    length_of_stay: int,
    booking_window: int,
    checkin_date: str,
    checkout_date: str,
    is_mobile: bool,
    sort_type: str,
    requested_filters: tuple[str, ...],
    query_budget: int,
    room_count: int,
    visitor_hist_starrating: float | None,
    visitor_hist_adr_usd: float | None,
    orig_destination_distance: float,
    is_unbiased: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for rank_index in range(items_per_query):
        prop_offset = (query_index * 7 + rank_index * 3) % property_count
        prop_id = f"property-{prop_offset:06d}"
        prop_country_id = f"country-{(prop_offset * 11 + query_index) % 172:03d}"
        star_rating = 1 + ((prop_offset + query_index) % 5)
        review_rating = round(max(1.0, min(5.0, rng.gauss(3.2 + 0.28 * star_rating, 0.45))), 2)
        review_count = int(max(1, rng.lognormvariate(5.2 + 0.18 * star_rating, 0.9)))
        location_score1 = round(max(0.05, rng.lognormvariate(0.6 + 0.06 * star_rating, 0.5)), 4)
        location_score2 = round(
            max(0.01, min(1.0, rng.betavariate(2.0 + star_rating / 2.0, 3.2))),
            4,
        )
        brand_bool = rng.random() < 0.46
        price_bucket = int(max(1, min(5, round(1 + star_rating * 0.65 + rng.gauss(0.0, 0.85)))))
        base_price = 54 + 28 * star_rating + 16 * price_bucket + rng.gauss(0.0, 18.0)
        if brand_bool:
            base_price *= 1.08
        if visitor_hist_adr_usd:
            base_price = 0.82 * base_price + 0.18 * visitor_hist_adr_usd
        price_usd = round(max(38.0, base_price), 2)
        historical_price_usd = max(35.0, price_usd * rng.uniform(0.78, 1.24))
        prop_log_historical_price = round(math.log(historical_price_usd), 4)
        promotion_flag = rng.random() < 0.18
        competitor_price_delta = round(rng.gauss(0.0, 0.12) - (0.06 if promotion_flag else 0.0), 4)
        competitor_availability_rate = round(rng.betavariate(6.0, 2.2), 4)
        amenities = _sample_amenities(rng, star_rating)
        number_of_amenities = sum(amenities.values())
        is_free_cancellation = rng.random() < 0.75 - 0.08 * max(0, price_bucket - 2)
        is_travel_ad = rank_index % 7 == 0 and rng.random() < 0.55
        is_drr = rng.random() < 0.18
        utility = _latent_utility(
            star_rating=star_rating,
            review_rating=review_rating,
            review_count=review_count,
            location_score1=location_score1,
            location_score2=location_score2,
            price_usd=price_usd,
            price_bucket=price_bucket,
            query_budget=query_budget,
            visitor_hist_starrating=visitor_hist_starrating,
            visitor_hist_adr_usd=visitor_hist_adr_usd,
            orig_destination_distance=orig_destination_distance,
            amenities=amenities,
            requested_filters=requested_filters,
            is_free_cancellation=is_free_cancellation,
            is_drr=is_drr,
            promotion_flag=promotion_flag,
            competitor_price_delta=competitor_price_delta,
            competitor_availability_rate=competitor_availability_rate,
            rng=rng,
        )
        policy_score = (
            utility
            + (0.10 if is_travel_ad else 0.0)
            + (0.05 if is_drr else 0.0)
            + (0.04 if promotion_flag else 0.0)
        )
        candidates.append(
            {
                "query_id": query_id,
                "item_id": prop_id,
                "group_id": prop_id,
                "prop_id": prop_id,
                "random_bool": is_unbiased,
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
                "adult_count": adult_count,
                "child_count": child_count,
                "length_of_stay": length_of_stay,
                "booking_window": booking_window,
                "destination_id": destination_id,
                "point_of_sale": point_of_sale,
                "geo_location_country": geo_country,
                "is_mobile": is_mobile,
                "sort_type": sort_type,
                "applied_filters": "|".join(requested_filters),
                "visitor_hist_starrating": visitor_hist_starrating,
                "visitor_hist_adr_usd": visitor_hist_adr_usd,
                "srch_room_count": room_count,
                "orig_destination_distance": orig_destination_distance,
                "prop_country_id": prop_country_id,
                "prop_brand_bool": brand_bool,
                "star_rating": star_rating,
                "review_rating": review_rating,
                "review_count": review_count,
                "prop_location_score1": location_score1,
                "prop_location_score2": location_score2,
                "prop_log_historical_price": prop_log_historical_price,
                "price_usd": price_usd,
                "price_bucket": price_bucket,
                "is_free_cancellation": is_free_cancellation,
                "is_travel_ad": is_travel_ad,
                "is_drr": is_drr,
                "promotion_flag": promotion_flag,
                "competitor_price_delta": competitor_price_delta,
                "competitor_availability_rate": competitor_availability_rate,
                "marketplace_supply_segment": "baseline",
                "is_cold_start_supply": False,
                "exposure_skew_score": 0.0,
                "number_of_amenities": number_of_amenities,
                **amenities,
                "_utility": utility,
                "_policy_score": policy_score,
            }
        )

    _add_listwise_features(candidates)

    if is_unbiased:
        display_order = list(candidates)
        rng.shuffle(display_order)
    else:
        display_order = sorted(candidates, key=lambda row: row["_policy_score"], reverse=True)

    has_positive_label = False
    for position, row in enumerate(display_order, start=1):
        clicked, transacted = _sample_outcome(
            utility=row["_utility"],
            position=position,
            is_free_cancellation=bool(row["is_free_cancellation"]),
            rng=rng,
        )
        row["position"] = position
        row["is_unbiased"] = is_unbiased
        row["num_clicks"] = clicked
        row["is_trans"] = transacted
        row["label"] = 2.0 if transacted else 1.0 if clicked else 0.0
        has_positive_label = has_positive_label or row["label"] > 0.0
        row.pop("_utility")
        row.pop("_policy_score")

    if not has_positive_label:
        best_row = max(display_order, key=lambda row: (row["review_rating"], row["review_count"]))
        best_row["num_clicks"] = 1
        best_row["is_trans"] = False
        best_row["label"] = 1.0

    return sorted(display_order, key=lambda row: int(row["position"]))


def _apply_marketplace_controls(
    rows: list[dict[str, Any]],
    config: SyntheticMarketplaceConfig,
) -> None:
    if not rows:
        return
    rng = random.Random(config.seed + 10_000)
    property_ids = sorted({str(row["prop_id"]) for row in rows})
    popular_pool_size = max(
        1,
        round(len(property_ids) * (1.0 - config.supply_concentration)),
    )
    popular_pool = property_ids[:popular_pool_size]

    cold_index = 0
    for row in rows:
        segment = "long_tail"
        if rng.random() < config.cold_start_rate:
            cold_id = f"cold-start-{cold_index:06d}"
            cold_index += 1
            _replace_property_id(row, cold_id)
            segment = "cold_start"
        elif rng.random() < config.supply_concentration:
            _replace_property_id(row, rng.choice(popular_pool))
            segment = "popular"
        row["marketplace_supply_segment"] = segment
        row["is_cold_start_supply"] = segment == "cold_start"

    if config.exposure_skew > 0.0:
        _apply_exposure_skew(rows, config.exposure_skew, rng)
    rows.sort(key=lambda row: (str(row["query_id"]), int(row["position"])))


def _replace_property_id(row: dict[str, Any], property_id: str) -> None:
    row["prop_id"] = property_id
    row["item_id"] = property_id
    row["group_id"] = property_id


def _apply_exposure_skew(
    rows: list[dict[str, Any]],
    exposure_skew: float,
    rng: random.Random,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query_id"]), []).append(row)

    for query_rows in grouped.values():
        if query_rows and bool(query_rows[0]["is_unbiased"]):
            continue
        query_size = len(query_rows)
        for row in query_rows:
            segment = str(row["marketplace_supply_segment"])
            segment_bonus = (
                1.0 if segment == "popular" else -0.5 if segment == "cold_start" else 0.0
            )
            score = (
                float(row["position"])
                - exposure_skew * query_size * segment_bonus
                + rng.random() * 0.001
            )
            row["exposure_skew_score"] = round(score, 6)
        query_rows.sort(key=lambda row: float(row["exposure_skew_score"]))
        for position, row in enumerate(query_rows, start=1):
            row["position"] = position


def _row_to_ranking_example(row: dict[str, Any]) -> RankingExample:
    return RankingExample(
        query_id=str(row["query_id"]),
        item_id=str(row["item_id"]),
        group_id=str(row["group_id"]),
        label=float(row["label"]),
        is_unbiased=bool(row["is_unbiased"]),
        position=int(row["position"]) if row["is_unbiased"] else None,
        features={column: row[column] for column in RECTOUR_LIKE_FEATURE_COLUMNS},
    )


def _sample_filters(rng: random.Random) -> tuple[str, ...]:
    filters = []
    if rng.random() < 0.34:
        filters.append("free_cancellation")
    if rng.random() < 0.25:
        filters.append("free_parking")
    if rng.random() < 0.18:
        filters.append("swimming_pool")
    if rng.random() < 0.12:
        filters.append("gym")
    return tuple(filters)


def _sample_amenities(rng: random.Random, star_rating: int) -> dict[str, bool]:
    quality_boost = 0.05 * max(0, star_rating - 2)
    return {
        "free_parking": rng.random() < 0.34,
        "parking": rng.random() < 0.62,
        "gym": rng.random() < 0.28 + quality_boost,
        "swimming_pool": rng.random() < 0.24 + quality_boost,
        "washer_dryer": rng.random() < 0.16,
        "spa_services": rng.random() < 0.12 + quality_boost,
        "free_airport_transportation": rng.random() < 0.10,
    }


def _latent_utility(
    *,
    star_rating: int,
    review_rating: float,
    review_count: int,
    location_score1: float,
    location_score2: float,
    price_usd: float,
    price_bucket: int,
    query_budget: int,
    visitor_hist_starrating: float | None,
    visitor_hist_adr_usd: float | None,
    orig_destination_distance: float,
    amenities: dict[str, bool],
    requested_filters: tuple[str, ...],
    is_free_cancellation: bool,
    is_drr: bool,
    promotion_flag: bool,
    competitor_price_delta: float,
    competitor_availability_rate: float,
    rng: random.Random,
) -> float:
    price_fit = 1.0 - abs(price_bucket - query_budget) / 4.0
    review_volume = math.log1p(review_count) / math.log1p(5000)
    location_fit = min(1.0, math.log1p(location_score1) / math.log1p(6.0)) * 0.45
    location_fit += location_score2 * 0.55
    visitor_star_fit = (
        1.0 - abs(star_rating - visitor_hist_starrating) / 4.0
        if visitor_hist_starrating is not None
        else 0.55
    )
    visitor_price_fit = (
        1.0 - min(1.0, abs(price_usd - visitor_hist_adr_usd) / max(visitor_hist_adr_usd, 1.0))
        if visitor_hist_adr_usd is not None
        else price_fit
    )
    distance_penalty = min(0.08, math.log1p(orig_destination_distance) / 140.0)
    requested_set = set(requested_filters)
    amenity_match = 0.0
    if "free_parking" in requested_set:
        amenity_match += 0.08 if amenities["free_parking"] else -0.05
    if "swimming_pool" in requested_set:
        amenity_match += 0.08 if amenities["swimming_pool"] else -0.04
    if "gym" in requested_set:
        amenity_match += 0.05 if amenities["gym"] else -0.03
    if "free_cancellation" in requested_set:
        amenity_match += 0.08 if is_free_cancellation else -0.06
    utility = (
        0.16 * (star_rating / 5.0)
        + 0.20 * (review_rating / 5.0)
        + 0.13 * review_volume
        + 0.12 * price_fit
        + 0.10 * visitor_star_fit
        + 0.10 * visitor_price_fit
        + 0.10 * location_fit
        + 0.08 * (sum(amenities.values()) / len(_AMENITY_COLUMNS))
        + amenity_match
        + (0.05 if is_free_cancellation else 0.0)
        + (0.03 if is_drr else 0.0)
        + (0.04 if promotion_flag else 0.0)
        + 0.05 * max(0.0, -competitor_price_delta)
        + 0.03 * competitor_availability_rate
        - distance_penalty
        + rng.gauss(0.0, 0.045)
    )
    return max(0.0, min(1.0, utility))


def _add_listwise_features(candidates: list[dict[str, Any]]) -> None:
    prices = [float(row["price_usd"]) for row in candidates]
    median_price = sorted(prices)[len(prices) // 2]
    historical_prices = {
        row["prop_id"]: math.exp(float(row["prop_log_historical_price"])) for row in candidates
    }
    _assign_rank(candidates, output_column="price_rank", sort_column="price_usd", reverse=False)
    _assign_rank(candidates, output_column="star_rank", sort_column="star_rating", reverse=True)
    _assign_rank(
        candidates,
        output_column="location_score_rank",
        sort_column="prop_location_score2",
        reverse=True,
    )
    for row in candidates:
        row["price_diff_from_query_median"] = round(float(row["price_usd"]) - median_price, 2)
        row["historical_price_gap"] = round(
            float(row["price_usd"]) - historical_prices[str(row["prop_id"])],
            2,
        )


def _assign_rank(
    candidates: list[dict[str, Any]],
    *,
    output_column: str,
    sort_column: str,
    reverse: bool,
) -> None:
    ranked = sorted(candidates, key=lambda row: float(row[sort_column]), reverse=reverse)
    for rank, row in enumerate(ranked, start=1):
        row[output_column] = rank


def _sample_outcome(
    *,
    utility: float,
    position: int,
    is_free_cancellation: bool,
    rng: random.Random,
) -> tuple[int, bool]:
    examination = 1.0 / math.log2(position + 1.6)
    click_probability = min(0.92, max(0.02, (utility**1.8) * examination * 1.35))
    clicked = rng.random() < click_probability
    if not clicked:
        return 0, False
    extra_click = rng.random() < max(0.02, min(0.18, click_probability * 0.18))
    transaction_probability = min(
        0.45,
        max(0.01, (utility**2.5) * 0.36 + (0.04 if is_free_cancellation else 0.0)),
    )
    transacted = rng.random() < transaction_probability
    return 2 if extra_click else 1, transacted
