import math
import pytest
from datetime import datetime, timezone, timedelta

from backend.similarity import (
    CONFIG,
    l2_normalize,
    cosine_similarity,
    rescale_semantic,
    haversine_distance,
    location_score,
    temporal_score,
    category_similarity,
    discipline_similarity,
    score_pair,
    generate_reasons,
    plan_cluster_update,
    build_embedding_text,
)


def test_weights_sum_to_one():
    """Verify that all scoring weights sum exactly to 1.0."""
    total_weights = sum(CONFIG["weights"].values())
    assert math.isclose(total_weights, 1.0, abs_tol=1e-6)


def test_l2_normalize_unit_length():
    """Verify that L2 normalization produces vectors with norm 1.0."""
    vec = [3.0, 4.0]
    normed = l2_normalize(vec)
    norm = math.sqrt(sum(x * x for x in normed))
    assert math.isclose(norm, 1.0, abs_tol=1e-6)
    assert math.isclose(normed[0], 0.6, abs_tol=1e-6)
    assert math.isclose(normed[1], 0.8, abs_tol=1e-6)


def test_cosine_similarity_unit_vectors():
    """Verify cosine similarity between known unit vectors."""
    v1 = l2_normalize([1.0, 0.0, 0.0])
    v2 = l2_normalize([1.0, 0.0, 0.0])
    v3 = l2_normalize([0.0, 1.0, 0.0])
    assert math.isclose(cosine_similarity(v1, v2), 1.0, abs_tol=1e-6)
    assert math.isclose(cosine_similarity(v1, v3), 0.0, abs_tol=1e-6)


def test_rescale_semantic_floor():
    """Verify raw cosine to rescaled semantic range with semantic_floor 0.50."""
    assert rescale_semantic(0.50, floor=0.50) == 0.0
    assert rescale_semantic(0.75, floor=0.50) == 0.5
    assert rescale_semantic(1.00, floor=0.50) == 1.0
    assert rescale_semantic(0.30, floor=0.50) == 0.0  # Clamped to 0


def test_haversine_known_distance():
    """Verify Haversine against a known real-world distance (approx 60m)."""
    # Points 60 meters apart
    lat1, lon1 = 28.6139, 77.2090
    lat2, lon2 = 28.6144, 77.2090
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    assert 50.0 < dist < 65.0


def test_location_score_distance_bands():
    """Verify distance mapping to location score bands."""
    assert location_score(50.0) == 1.0
    assert location_score(250.0) == 0.8
    assert location_score(1200.0) == 0.5
    assert location_score(5000.0) == 0.2
    assert location_score(15000.0) == 0.0


def test_temporal_decay():
    """Verify half-life exponential decay (90 days = 0.5 score)."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t_same = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t_90d = t0 + timedelta(days=90)
    t_180d = t0 + timedelta(days=180)

    assert math.isclose(temporal_score(t0, t_same), 1.0, abs_tol=1e-4)
    assert math.isclose(temporal_score(t0, t_90d), 0.5, abs_tol=1e-4)
    assert math.isclose(temporal_score(t0, t_180d), 0.25, abs_tol=1e-4)


def test_worked_example_1_same_issue_near_school():
    """
    Worked Example 1: Same issue, same school, 60m apart, same day, same category & discipline.
    raw cosine 0.95 -> rescaled semantic = (0.95 - 0.50)/0.50 = 0.90
    Formula: 0.55*0.90 + 0.15*1 + 0.10*1 + 0.15*1 + 0.05*1 = 0.945 -> POSSIBLE_DUPLICATE
    """
    now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    p1 = {
        "id": "1",
        "category": "Roads & Transport",
        "technical_discipline": ["Civil Engineering"],
        "latitude": 28.6139,
        "longitude": 77.2090,
        "created_at": now,
    }
    p2 = {
        "id": "2",
        "category": "Roads & Transport",
        "technical_discipline": ["Civil Engineering"],
        "latitude": 28.6144,
        "longitude": 77.2090,
        "created_at": now,
    }
    result = score_pair(p1, p2, raw_cosine=0.95)
    assert math.isclose(result["combined_score"], 0.945, abs_tol=1e-3)
    assert result["match_level"] == "POSSIBLE_DUPLICATE"


def test_worked_example_2_jaipur_vs_delhi():
    """
    Worked Example 2: Same wording but Jaipur vs Delhi (~240 km apart; location score 0).
    0.55*0.90 + 0.15*1 + 0.10*1 + 0.15*0 + 0.05*1 = 0.795 -> RELATED_PROBLEM, NOT a duplicate.
    """
    now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    p_delhi = {
        "id": "1",
        "category": "Roads & Transport",
        "technical_discipline": ["Civil Engineering"],
        "latitude": 28.6139,
        "longitude": 77.2090,
        "created_at": now,
    }
    p_jaipur = {
        "id": "2",
        "category": "Roads & Transport",
        "technical_discipline": ["Civil Engineering"],
        "latitude": 26.9124,
        "longitude": 75.7873,
        "created_at": now,
    }
    result = score_pair(p_delhi, p_jaipur, raw_cosine=0.95)
    assert math.isclose(result["combined_score"], 0.795, abs_tol=1e-3)
    assert result["match_level"] == "RELATED_PROBLEM"


def test_worked_example_3_gps_denied_no_penalty():
    """
    Worked Example 3: Same as example 1 but GPS denied.
    Weights re-normalized over 0.85: (0.495 + 0.15 + 0.10 + 0.05) / 0.85 = 0.935 -> POSSIBLE_DUPLICATE
    """
    now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    p1 = {
        "id": "1",
        "category": "Roads & Transport",
        "technical_discipline": ["Civil Engineering"],
        "latitude": 28.6139,
        "longitude": 77.2090,
        "created_at": now,
    }
    p2 = {
        "id": "2",
        "category": "Roads & Transport",
        "technical_discipline": ["Civil Engineering"],
        "latitude": None,
        "longitude": None,
        "created_at": now,
    }
    result = score_pair(p1, p2, raw_cosine=0.95)
    expected = (0.55 * 0.90 + 0.15 + 0.10 + 0.05) / 0.85
    assert math.isclose(result["combined_score"], expected, abs_tol=1e-3)
    assert result["match_level"] == "POSSIBLE_DUPLICATE"


def test_worked_example_4_unrelated_topics():
    """
    Worked Example 4: Unrelated topics (water leakage vs garbage), raw cosine 0.55, different domain.
    """
    now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    p_water = {
        "id": "1",
        "category": "Water & Sanitation",
        "technical_discipline": ["Hydrology"],
        "latitude": 28.6139,
        "longitude": 77.2090,
        "created_at": now,
    }
    p_garbage = {
        "id": "2",
        "category": "Waste Management",
        "technical_discipline": ["Environmental Engineering"],
        "latitude": 28.6139,
        "longitude": 77.2090,
        "created_at": now,
    }
    result = score_pair(p_water, p_garbage, raw_cosine=0.55)
    assert result["combined_score"] < 0.65
    assert result["match_level"] == "NO_STRONG_MATCH"


def test_safety_guard_against_false_duplicates():
    """
    Safety Guard (Section 8.4):
    Even if category + location + time are 1.0 (combined high),
    if semantic < duplicate_min_semantic (0.60), it can NEVER be POSSIBLE_DUPLICATE.
    """
    now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    p1 = {
        "id": "1",
        "category": "Roads & Transport",
        "technical_discipline": ["Civil Engineering"],
        "latitude": 28.6139,
        "longitude": 77.2090,
        "created_at": now,
    }
    p2 = {
        "id": "2",
        "category": "Roads & Transport",
        "technical_discipline": ["Civil Engineering"],
        "latitude": 28.6139,
        "longitude": 77.2090,
        "created_at": now,
    }
    # raw_cosine = 0.75 -> rescaled semantic = 0.50 (below 0.60 min)
    # signals: 0.55*0.50 + 0.15 + 0.10 + 0.15 + 0.05 = 0.725
    result = score_pair(p1, p2, raw_cosine=0.75)
    assert result["signals"]["semantic"] < 0.60
    assert result["match_level"] != "POSSIBLE_DUPLICATE"


def test_cluster_planning_create_and_merge():
    """Verify connected component cluster planning logic."""
    # 1. New cluster creation when candidates have no existing cluster
    candidates = [{"problem_id": "P2", "combined_score": 0.85}]
    plan1 = plan_cluster_update(
        new_report_id="P1",
        candidate_scores=candidates,
        existing_memberships={},
        existing_clusters={},
    )
    assert plan1["action"] == "CREATE_NEW"
    assert "P1" in plan1["member_ids_to_add"]
    assert "P2" in plan1["member_ids_to_add"]

    # 2. Add to existing cluster
    memberships = {"P2": "C1"}
    clusters = {"C1": {"id": "C1", "created_at": "2026-01-01T00:00:00Z"}}
    plan2 = plan_cluster_update(
        new_report_id="P3",
        candidate_scores=[{"problem_id": "P2", "combined_score": 0.88}],
        existing_memberships=memberships,
        existing_clusters=clusters,
    )
    assert plan2["action"] == "ADD_TO_EXISTING"
    assert plan2["target_cluster_id"] == "C1"
    assert "P3" in plan2["member_ids_to_add"]

    # 3. Merge two clusters into the oldest
    memberships_multi = {"P2": "C_NEW", "P4": "C_OLD"}
    clusters_multi = {
        "C_NEW": {"id": "C_NEW", "created_at": "2026-05-01T00:00:00Z"},
        "C_OLD": {"id": "C_OLD", "created_at": "2026-01-01T00:00:00Z"},
    }
    plan3 = plan_cluster_update(
        new_report_id="P5",
        candidate_scores=[
            {"problem_id": "P2", "combined_score": 0.82},
            {"problem_id": "P4", "combined_score": 0.83},
        ],
        existing_memberships=memberships_multi,
        existing_clusters=clusters_multi,
    )
    assert plan3["action"] == "MERGE_CLUSTERS"
    assert plan3["target_cluster_id"] == "C_OLD"
    assert "C_NEW" in plan3["absorbed_cluster_ids"]
