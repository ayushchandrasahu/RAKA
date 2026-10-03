import os
import json
import uuid
from typing import Dict, List, Any
from dotenv import load_dotenv

from backend import database as db
from backend.similarity import CONFIG, score_pair, plan_cluster_update

load_dotenv()


def rebuild_all_clusters() -> Dict[str, Any]:
    """
    Reconstructs all problem clusters from scratch deterministically.
    Scans all analyzed problem pairs and builds connected components.
    """
    conn = db.get_connection()
    c = conn.cursor()

    # Clear current clusters and memberships
    if db.USE_POSTGRES:
        c.execute("DELETE FROM problem_cluster_members;")
        c.execute("DELETE FROM problem_clusters;")
    else:
        c.execute("DELETE FROM problem_cluster_members;")
        c.execute("DELETE FROM problem_clusters;")
    conn.commit()

    # Fetch all analyzed problems sorted by created_at ascending
    if db.USE_POSTGRES:
        c.execute("""
            SELECT p.id, p.latitude, p.longitude, p.created_at, p.status, p.is_demo,
                   a.category, a.problem_summary, a.technical_discipline,
                   e.embedding
            FROM problems p
            JOIN problem_ai_analysis a ON a.problem_id = p.id
            JOIN problem_embeddings e ON e.problem_id = p.id
            WHERE p.status = 'AI_ANALYZED'
            ORDER BY p.created_at ASC;
        """)
        rows = c.fetchall()
    else:
        c.execute("""
            SELECT p.id, p.latitude, p.longitude, p.created_at, p.status, p.is_demo,
                   a.category, a.problem_summary, a.technical_discipline,
                   e.embedding
            FROM problems p
            JOIN problem_ai_analysis a ON a.problem_id = p.id
            JOIN problem_embeddings e ON e.problem_id = p.id
            WHERE p.status = 'AI_ANALYZED'
            ORDER BY p.created_at ASC;
        """)
        rows = c.fetchall()

    conn.close()

    reports = []
    for r in rows:
        pid = r["id"] if db.USE_POSTGRES else r[0]
        lat = r["latitude"] if db.USE_POSTGRES else r[1]
        lng = r["longitude"] if db.USE_POSTGRES else r[2]
        created = r["created_at"] if db.USE_POSTGRES else r[3]
        status = r["status"] if db.USE_POSTGRES else r[4]
        is_demo = bool(r["is_demo"] if db.USE_POSTGRES else r[5])
        cat = r["category"] if db.USE_POSTGRES else r[6]
        summary = r["problem_summary"] if db.USE_POSTGRES else r[7]
        td_raw = r["technical_discipline"] if db.USE_POSTGRES else r[8]
        td = json.loads(td_raw) if isinstance(td_raw, str) else (td_raw or [])
        emb_raw = r["embedding"] if db.USE_POSTGRES else r[9]
        emb = emb_raw if db.USE_POSTGRES else json.loads(emb_raw)

        reports.append({
            "problem_id": pid,
            "latitude": lat,
            "longitude": lng,
            "created_at": created,
            "status": status,
            "is_demo": is_demo,
            "category": cat,
            "problem_summary": summary,
            "technical_discipline": td,
            "embedding": emb,
        })

    # Progressively process each report as if arriving incrementally
    processed_count = 0
    for idx, rep in enumerate(reports):
        processed_count += 1
        # Candidates are previously processed reports with same is_demo
        prev_reports = [r for r in reports[:idx] if r["is_demo"] == rep["is_demo"]]
        if not prev_reports:
            continue

        scored_candidates = []
        for prev in prev_reports:
            from backend.similarity import cosine_similarity
            cos = cosine_similarity(rep["embedding"], prev["embedding"])
            pair_score = score_pair(rep, prev, cos)
            scored_candidates.append({
                "problem_id": prev["problem_id"],
                "combined_score": pair_score["combined_score"],
                "match_level": pair_score["match_level"],
            })

        existing_memberships, existing_clusters = db.get_cluster_state()
        plan = plan_cluster_update(
            new_report_id=rep["problem_id"],
            candidate_scores=scored_candidates,
            existing_memberships=existing_memberships,
            existing_clusters=existing_clusters,
        )
        db.apply_cluster_plan(plan, rep["problem_id"], rep["category"], rep["problem_summary"])

    return {"status": "SUCCESS", "processed_reports": processed_count}


if __name__ == "__main__":
    result = rebuild_all_clusters()
    print(f"Rebuild completed successfully: {result}")
