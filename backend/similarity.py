import math
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

CONFIG = {
    "weights": {
        "semantic": 0.55,
        "category": 0.15,
        "discipline": 0.10,
        "location": 0.15,
        "temporal": 0.05,
    },
    "semantic_floor": 0.50,
    "thresholds": {
        "no_strong_match": 0.65,
        "possible_duplicate": 0.82,
    },
    "duplicate_min_semantic": 0.60,
    "cluster_edge_threshold": 0.75,
    "cluster_size_review_threshold": 25,
    "cluster_min_score_gap_review": 0.25,
    "distance_bands_m": [
        (100.0, 1.0),
        (500.0, 0.8),
        (2000.0, 0.5),
        (10000.0, 0.2),
        (float("inf"), 0.0),
    ],
    "temporal_half_life_days": 90.0,
    "top_k": 10,
    "recent_days": 30,
    "neighboring_categories": {
        frozenset(["Roads & Transport", "Infrastructure"]),
        frozenset(["Water & Sanitation", "Infrastructure"]),
        frozenset(["Electricity & Energy", "Infrastructure"]),
        frozenset(["Waste Management", "Environment"]),
        frozenset(["Water & Sanitation", "Environment"]),
        frozenset(["Housing", "Infrastructure"]),
    },
}


def build_embedding_text(
    problem_text: str,
    summary: str,
    category: str,
    technical_discipline: List[str],
    impact_areas: List[str],
    keywords: List[str],
    image_observations: Optional[List[str]] = None,
    manual_location: Optional[str] = None,
    sub_category: Optional[str] = None,
    voice_transcript: Optional[str] = None,
) -> str:
    """Builds a normalized, bounded document representation for embedding."""
    lines = [
        f"Problem: {problem_text.strip()}",
    ]
    if voice_transcript and voice_transcript.strip():
        lines.append(f"Voice transcript: {voice_transcript.strip()}")
    lines.append(f"Summary: {summary.strip()}")
    lines.append(f"Category: {category.strip()}")
    if sub_category and sub_category.strip():
        lines.append(f"Sub-category: {sub_category.strip()}")
    lines.append(f"Technical discipline: {', '.join(technical_discipline[:4])}")
    lines.append(f"Impact areas: {', '.join(impact_areas[:5])}")
    lines.append(f"Keywords: {', '.join(keywords[:6])}")
    if image_observations:
        clean_obs = [obs.strip() for obs in image_observations if obs.strip()]
        if clean_obs:
            lines.append(f"Visual evidence: {'; '.join(clean_obs[:3])}")
    if manual_location and manual_location.strip():
        lines.append(f"Location context: {manual_location.strip()[:150]}")

    return "\n".join(lines)



def l2_normalize(vec: List[float]) -> List[float]:
    """L2 normalizes a vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculates dot product of two L2-normalized vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector dimensions must match")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(-1.0, min(1.0, dot))


def rescale_semantic(raw_cosine: float, floor: float = CONFIG["semantic_floor"]) -> float:
    """Rescales cosine similarity using a floor threshold."""
    if raw_cosine <= floor:
        return 0.0
    return max(0.0, min(1.0, (raw_cosine - floor) / (1.0 - floor)))


def category_similarity(cat_a: str, cat_b: str) -> float:
    """Scores domain category alignment."""
    if not cat_a or not cat_b:
        return 0.0
    if cat_a.strip().lower() == cat_b.strip().lower():
        return 1.0
    pair = frozenset([cat_a.strip(), cat_b.strip()])
    if pair in CONFIG["neighboring_categories"]:
        return 0.5
    return 0.0


def discipline_similarity(disc_a: List[str], disc_b: List[str]) -> float:
    """Computes Jaccard similarity across technical disciplines."""
    set_a = {d.strip().lower() for d in disc_a if d.strip()}
    set_b = {d.strip().lower() for d in disc_b if d.strip()}
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates geodesic distance between two points in meters."""
    R = 6371000.0  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def location_score(distance_m: Optional[float]) -> Optional[float]:
    """Maps distance in meters to a location score between 0.0 and 1.0."""
    if distance_m is None:
        return None
    for max_dist, score in CONFIG["distance_bands_m"]:
        if distance_m <= max_dist:
            return score
    return 0.0


def format_distance(distance_m: Optional[float]) -> str:
    """Formats distance for public display without exposing exact coordinates."""
    if distance_m is None:
        return "Not available"
    if distance_m < 1000:
        rounded = round(distance_m / 10.0) * 10
        return f"~{int(rounded)} m"
    else:
        km = distance_m / 1000.0
        return f"{km:.1f} km"


def temporal_score(
    time_a: Any,
    time_b: Any,
    half_life_days: float = CONFIG["temporal_half_life_days"],
) -> float:
    """Computes time relevance using exponential decay."""
    if isinstance(time_a, str):
        time_a = datetime.fromisoformat(time_a.replace("Z", "+00:00"))
    if isinstance(time_b, str):
        time_b = datetime.fromisoformat(time_b.replace("Z", "+00:00"))
    if time_a.tzinfo is None:
        time_a = time_a.replace(tzinfo=timezone.utc)
    if time_b.tzinfo is None:
        time_b = time_b.replace(tzinfo=timezone.utc)
    delta = abs((time_a - time_b).total_seconds()) / 86400.0
    return 0.5 ** (delta / half_life_days)


def temporal_label(score: float) -> str:
    if score >= 0.66:
        return "High"
    if score >= 0.33:
        return "Medium"
    return "Low"


def score_pair(
    cand_a: Dict[str, Any],
    cand_b: Dict[str, Any],
    raw_cosine: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes multi-signal score between two problem records.
    Returns composite score, individual signals, classification level, and reasons.
    """
    cfg = config or CONFIG
    weights = cfg["weights"]

    # 1. Semantic score
    if raw_cosine is None:
        raw_cosine = cosine_similarity(cand_a["embedding"], cand_b["embedding"])
    semantic = rescale_semantic(raw_cosine, cfg["semantic_floor"])

    # 2. Category score
    cat_score = category_similarity(cand_a.get("category", ""), cand_b.get("category", ""))

    # 3. Discipline score
    disc_score = discipline_similarity(
        cand_a.get("technical_discipline", []), cand_b.get("technical_discipline", [])
    )

    # 4. Location score
    loc_score = None
    dist_m = None
    if (
        cand_a.get("latitude") is not None
        and cand_a.get("longitude") is not None
        and cand_b.get("latitude") is not None
        and cand_b.get("longitude") is not None
    ):
        dist_m = haversine_distance(
            cand_a["latitude"], cand_a["longitude"], cand_b["latitude"], cand_b["longitude"]
        )
        loc_score = location_score(dist_m)

    # 5. Temporal score
    t_a = cand_a.get("created_at") or datetime.now(timezone.utc)
    t_b = cand_b.get("created_at") or datetime.now(timezone.utc)
    temp_score = temporal_score(t_a, t_b, cfg["temporal_half_life_days"])

    # Weight re-normalization if location is missing
    available_weights = [
        ("semantic", weights["semantic"], semantic),
        ("category", weights["category"], cat_score),
        ("discipline", weights["discipline"], disc_score),
        ("temporal", weights["temporal"], temp_score),
    ]
    if loc_score is not None:
        available_weights.append(("location", weights["location"], loc_score))

    total_weight = sum(w for _, w, _ in available_weights)
    weighted_sum = sum(w * val for _, w, val in available_weights)
    combined = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Decision Level
    dup_thresh = cfg["thresholds"]["possible_duplicate"]
    rel_thresh = cfg["thresholds"]["no_strong_match"]
    min_sem = cfg["duplicate_min_semantic"]

    if combined >= dup_thresh and semantic >= min_sem:
        level = "LEVEL_3"
        label = "POSSIBLE DUPLICATE"
    elif combined >= rel_thresh:
        level = "LEVEL_2"
        label = "RELATED REPORT"
    else:
        level = "LEVEL_1"
        label = "NO_STRONG_MATCH"

    # Evidence reasons
    evidence_reasons = generate_reasons(
        semantic=semantic,
        cat_score=cat_score,
        disc_score=disc_score,
        loc_score=loc_score,
        dist_m=dist_m,
        temp_score=temp_score,
    )

    return {
        "combined_score": round(combined, 4),
        "raw_cosine": round(raw_cosine, 4),
        "semantic_similarity": round(semantic, 4),
        "category_similarity": round(cat_score, 4),
        "discipline_similarity": round(disc_score, 4),
        "location_score": round(loc_score, 4) if loc_score is not None else None,
        "distance_meters": round(dist_m, 1) if dist_m is not None else None,
        "distance_display": format_distance(dist_m),
        "temporal_score": round(temp_score, 4),
        "temporal_relevance": temporal_label(temp_score),
        "level": level,
        "label": label,
        "match_level": "POSSIBLE_DUPLICATE" if level == "LEVEL_3" else ("RELATED_PROBLEM" if level == "LEVEL_2" else "NO_STRONG_MATCH"),
        "signals": {
            "semantic": semantic,
            "category": cat_score,
            "discipline": disc_score,
            "location": loc_score,
            "distance_m": dist_m,
            "temporal": temp_score,
        },
        "reasons": evidence_reasons,
    }


def generate_reasons(
    pair_res_or_semantic: Any = None,
    cat_score: Optional[float] = None,
    disc_score: Optional[float] = None,
    loc_score: Optional[float] = None,
    dist_m: Optional[float] = None,
    temp_score: Optional[float] = None,
    **kwargs,
) -> List[str]:
    """Generates human-readable explainable reasons for why the pair is linked."""
    if isinstance(pair_res_or_semantic, dict):
        signals = pair_res_or_semantic.get("signals", {})
        semantic = signals.get("semantic", pair_res_or_semantic.get("semantic_similarity", 0.0))
        cat_score = signals.get("category", pair_res_or_semantic.get("category_similarity", 0.0))
        disc_score = signals.get("discipline", pair_res_or_semantic.get("discipline_similarity", 0.0))
        loc_score = signals.get("location", pair_res_or_semantic.get("location_score"))
        dist_m = signals.get("distance_m", pair_res_or_semantic.get("distance_meters"))
        temp_score = signals.get("temporal", pair_res_or_semantic.get("temporal_score", 1.0))
    else:
        semantic = float(pair_res_or_semantic) if pair_res_or_semantic is not None else 0.0
        cat_score = cat_score if cat_score is not None else 0.0
        disc_score = disc_score if disc_score is not None else 0.0
        temp_score = temp_score if temp_score is not None else 1.0

    reasons_list = []
    if cat_score == 1.0:
        reasons_list.append("Same public-service domain")
    elif cat_score > 0.0:
        reasons_list.append("Closely related service domain")

    if semantic >= 0.60:
        reasons_list.append("Highly similar problem description")
    elif semantic >= 0.40:
        reasons_list.append("Related problem description")

    if disc_score > 0.0:
        reasons_list.append("Overlapping technical disciplines required")

    if loc_score is not None and loc_score >= 0.5 and dist_m is not None:
        reasons_list.append(f"Located {format_distance(dist_m)} away")

    if temp_score >= 0.50:
        reasons_list.append("Reported recently")

    return reasons_list


def plan_cluster_update(
    new_report_id: str,
    candidate_scores: List[Dict[str, Any]],
    existing_memberships: Dict[str, str],  # problem_id -> cluster_id
    existing_clusters: Dict[str, Dict[str, Any]],  # cluster_id -> cluster_meta
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministic connected component cluster planner.
    Evaluates links >= cluster_edge_threshold.
    """
    cfg = config or CONFIG
    edge_thresh = cfg["cluster_edge_threshold"]

    qualified_links = [
        c for c in candidate_scores if c["combined_score"] >= edge_thresh
    ]

    if not qualified_links:
        return {"action": "NONE", "cluster_id": None}

    connected_cluster_ids = set()
    for link in qualified_links:
        other_id = link["problem_id"]
        if other_id in existing_memberships:
            connected_cluster_ids.add(existing_memberships[other_id])

    # If the current report is already in a cluster (e.g. on re-analysis)
    if new_report_id in existing_memberships:
        connected_cluster_ids.add(existing_memberships[new_report_id])

    if len(connected_cluster_ids) == 0:
        # Create brand new cluster
        new_cluster_id = str(uuid.uuid4())
        return {
            "action": "CREATE_NEW",
            "target_cluster_id": new_cluster_id,
            "member_ids_to_add": [new_report_id] + [l["problem_id"] for l in qualified_links],
            "links": qualified_links,
        }
    elif len(connected_cluster_ids) == 1:
        target_id = list(connected_cluster_ids)[0]
        return {
            "action": "ADD_TO_EXISTING",
            "target_cluster_id": target_id,
            "member_ids_to_add": [new_report_id] + [
                l["problem_id"] for l in qualified_links if l["problem_id"] not in existing_memberships
            ],
            "links": qualified_links,
        }
    else:
        # Multiple clusters to merge - merge into the OLDEST cluster
        sorted_clusters = sorted(
            connected_cluster_ids,
            key=lambda cid: existing_clusters.get(cid, {}).get("created_at", "")
        )
        oldest_target = sorted_clusters[0]
        clusters_to_merge = sorted_clusters[1:]
        return {
            "action": "MERGE_CLUSTERS",
            "target_cluster_id": oldest_target,
            "absorbed_cluster_ids": clusters_to_merge,
            "member_ids_to_add": [new_report_id],
            "links": qualified_links,
        }
