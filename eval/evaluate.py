import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from backend.similarity import (
    CONFIG,
    cosine_similarity,
    score_pair,
    build_embedding_text,
)
from backend.gemini_service import embed_text

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PAIRS_FILE = BASE_DIR / "eval" / "pairs.json"


def evaluate_dataset(config_override=None):
    cfg = config_override or CONFIG
    if not PAIRS_FILE.exists():
        print(f"Dataset file {PAIRS_FILE} not found.")
        return

    with open(PAIRS_FILE, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} labeled test pairs from eval/pairs.json.")
    print("=" * 75)
    print(f"EVALUATING WITH CONFIG:")
    print(f"  Weights: {cfg['weights']}")
    print(f"  Semantic floor: {cfg['semantic_floor']}")
    print(f"  Thresholds: {cfg['thresholds']}")
    print(f"  Duplicate min semantic: {cfg['duplicate_min_semantic']}")
    print("=" * 75)

    now = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    
    # Metrics counters
    # Duplicate level
    dup_tp = 0
    dup_fp = 0
    dup_fn = 0
    dup_tn = 0

    # Related level (duplicate OR related vs unrelated)
    rel_tp = 0
    rel_fp = 0
    rel_fn = 0
    rel_tn = 0

    false_positives = []
    false_negatives = []

    for pair in pairs:
        pid = pair["id"]
        true_label = pair["label"]  # 'duplicate', 'related', 'unrelated'

        # Build embedding texts
        doc_a = build_embedding_text(
            problem_text=pair["text_a"],
            summary=pair["text_a"],
            category=pair["cat_a"],
            technical_discipline=pair["disc_a"],
            impact_areas=["Civic Infrastructure"],
            keywords=[w for w in pair["text_a"].lower().split() if len(w) > 4][:5],
        )
        doc_b = build_embedding_text(
            problem_text=pair["text_b"],
            summary=pair["text_b"],
            category=pair["cat_b"],
            technical_discipline=pair["disc_b"],
            impact_areas=["Civic Infrastructure"],
            keywords=[w for w in pair["text_b"].lower().split() if len(w) > 4][:5],
        )

        vec_a = embed_text(doc_a)
        vec_b = embed_text(doc_b)
        cos = cosine_similarity(vec_a, vec_b)

        t_a = now
        t_b = now - timedelta(days=pair.get("days_apart", 0))

        prob_a = {
            "id": f"{pid}_a",
            "category": pair["cat_a"],
            "technical_discipline": pair["disc_a"],
            "latitude": pair.get("lat_a"),
            "longitude": pair.get("lon_a"),
            "created_at": t_a,
        }
        prob_b = {
            "id": f"{pid}_b",
            "category": pair["cat_b"],
            "technical_discipline": pair["disc_b"],
            "latitude": pair.get("lat_b"),
            "longitude": pair.get("lon_b"),
            "created_at": t_b,
        }

        res = score_pair(prob_a, prob_b, raw_cosine=cos, config=cfg)
        predicted_level = res["match_level"]  # 'POSSIBLE_DUPLICATE', 'RELATED_PROBLEM', 'NO_STRONG_MATCH'
        comb_score = res["combined_score"]

        # Duplicate classification evaluation
        is_true_dup = (true_label == "duplicate")
        is_pred_dup = (predicted_level == "POSSIBLE_DUPLICATE")

        if is_true_dup and is_pred_dup:
            dup_tp += 1
        elif not is_true_dup and is_pred_dup:
            dup_fp += 1
            false_positives.append((pid, true_label, predicted_level, comb_score, pair["text_a"], pair["text_b"]))
        elif is_true_dup and not is_pred_dup:
            dup_fn += 1
            false_negatives.append((pid, true_label, predicted_level, comb_score, pair["text_a"], pair["text_b"]))
        else:
            dup_tn += 1

        # Related/Match classification evaluation (Duplicate OR Related)
        is_true_rel = (true_label in ("duplicate", "related"))
        is_pred_rel = (predicted_level in ("POSSIBLE_DUPLICATE", "RELATED_PROBLEM"))

        if is_true_rel and is_pred_rel:
            rel_tp += 1
        elif not is_true_rel and is_pred_rel:
            rel_fp += 1
        elif is_true_rel and not is_pred_rel:
            rel_fn += 1
        else:
            rel_tn += 1

    def calc_metrics(tp, fp, fn, tn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        return prec, rec, f1, acc

    d_prec, d_rec, d_f1, d_acc = calc_metrics(dup_tp, dup_fp, dup_fn, dup_tn)
    r_prec, r_rec, r_f1, r_acc = calc_metrics(rel_tp, rel_fp, rel_fn, rel_tn)

    print("\nEVALUATION RESULTS:")
    print("-" * 75)
    print(f"LEVEL 3: POSSIBLE DUPLICATES")
    print(f"  TP: {dup_tp:2d} | FP: {dup_fp:2d} | FN: {dup_fn:2d} | TN: {dup_tn:2d}")
    print(f"  Precision: {d_prec*100:6.2f}%")
    print(f"  Recall:    {d_rec*100:6.2f}%")
    print(f"  F1 Score:  {d_f1*100:6.2f}%")
    print(f"  Accuracy:  {d_acc*100:6.2f}%")
    print("-" * 75)
    print(f"LEVEL 2: RELATED PROBLEMS (Duplicate OR Related vs Unrelated)")
    print(f"  TP: {rel_tp:2d} | FP: {rel_fp:2d} | FN: {rel_fn:2d} | TN: {rel_tn:2d}")
    print(f"  Precision: {r_prec*100:6.2f}%")
    print(f"  Recall:    {r_rec*100:6.2f}%")
    print(f"  F1 Score:  {r_f1*100:6.2f}%")
    print(f"  Accuracy:  {r_acc*100:6.2f}%")
    print("=" * 75)

    if false_positives:
        print("\nFALSE POSITIVES (Duplicate False Alarms):")
        for fp_item in false_positives:
            print(f"  - Pair #{fp_item[0]} (True: {fp_item[1]}, Predicted: {fp_item[2]}, Score: {fp_item[3]}):")
            print(f"      Text A: {fp_item[4][:60]}...")
            print(f"      Text B: {fp_item[5][:60]}...")

    if false_negatives:
        print("\nFALSE NEGATIVES (Missed Duplicates):")
        for fn_item in false_negatives:
            print(f"  - Pair #{fn_item[0]} (True: {fn_item[1]}, Predicted: {fn_item[2]}, Score: {fn_item[3]}):")
            print(f"      Text A: {fn_item[4][:60]}...")
            print(f"      Text B: {fn_item[5][:60]}...")


if __name__ == "__main__":
    evaluate_dataset()
