import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from backend import database as db
from backend.gemini_service import analyze_problem_with_gemini, embed_text
from backend.similarity import build_embedding_text, CONFIG

load_dotenv()

# Predefined realistic seed scenarios (Section 32)
SEED_PROBLEMS = [
    {
        "text": "Deep and dangerous potholes have opened up right outside the government primary school gate. School children and cycles keep tripping.",
        "category": "Roads & Transport",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "days_ago": 1,
    },
    {
        "text": "Severe road surface damage with a huge pothole outside the school entrance. Two-wheelers are slipping during morning hours.",
        "category": "Roads & Transport",
        "latitude": 28.6143,
        "longitude": 77.2092,
        "days_ago": 2,
    },
    {
        "text": "Large pothole near the school gate creating road hazard for passing buses and students.",
        "category": "Roads & Transport",
        "latitude": 28.6141,
        "longitude": 77.2088,
        "days_ago": 4,
    },
    {
        "text": "Huge heap of overflowing solid waste and plastic bags dumping near the vegetable market area, emitting foul stench and attracting stray cattle.",
        "category": "Waste Management",
        "latitude": 28.6250,
        "longitude": 77.2200,
        "days_ago": 5,
    },
    {
        "text": "Major drinking water pipeline breach leaking thousands of liters of clean water into the residential street. Pressure in home taps is zero.",
        "category": "Water & Sanitation",
        "latitude": 28.6300,
        "longitude": 77.2150,
        "days_ago": 8,
    },
    {
        "text": "High mast streetlight broken and dark near the main bus depot junction, creating safety concerns for women commuters at night.",
        "category": "Electricity & Energy",
        "latitude": 28.6180,
        "longitude": 77.2050,
        "days_ago": 12,
    },
    {
        "text": "Agricultural irrigation feeder canal broken and clogged with silt, preventing water from reaching downstream wheat fields.",
        "category": "Agriculture",
        "latitude": 28.5800,
        "longitude": 77.1500,
        "days_ago": 20,
    }
]


def seed_demo_data(clear_only: bool = False):
    conn = db.get_connection()
    c = conn.cursor()

    print("[Seed] Clearing existing demo rows...")
    if db.USE_POSTGRES:
        c.execute("""
            DELETE FROM problem_cluster_members WHERE problem_id IN (SELECT id FROM problems WHERE is_demo = true);
            DELETE FROM problem_clusters WHERE id NOT IN (SELECT DISTINCT cluster_id FROM problem_cluster_members);
            DELETE FROM problem_embeddings WHERE problem_id IN (SELECT id FROM problems WHERE is_demo = true);
            DELETE FROM problem_ai_analysis WHERE problem_id IN (SELECT id FROM problems WHERE is_demo = true);
            DELETE FROM problems WHERE is_demo = true;
        """)
    else:
        c.execute("DELETE FROM problem_cluster_members WHERE problem_id IN (SELECT id FROM problems WHERE is_demo = 1);")
        c.execute("DELETE FROM problem_clusters WHERE id NOT IN (SELECT DISTINCT cluster_id FROM problem_cluster_members);")
        c.execute("DELETE FROM problem_embeddings WHERE problem_id IN (SELECT id FROM problems WHERE is_demo = 1);")
        c.execute("DELETE FROM problem_ai_analysis WHERE problem_id IN (SELECT id FROM problems WHERE is_demo = 1);")
        c.execute("DELETE FROM problems WHERE is_demo = 1;")
    conn.commit()
    conn.close()

    if clear_only:
        print("[Seed] Demo data cleared.")
        return

    print(f"[Seed] Seeding {len(SEED_PROBLEMS)} realistic demonstration records...")
    now = datetime.now(timezone.utc)

    for idx, item in enumerate(SEED_PROBLEMS):
        prob_id = str(uuid.uuid4())
        created_time = now - timedelta(days=item["days_ago"], hours=idx * 2)

        # 1. Save problem
        db.save_problem(
            problem_id=prob_id,
            problem_text=item["text"],
            image_url=None,
            has_location=True,
            latitude=item["latitude"],
            longitude=item["longitude"],
            location_accuracy=10.0,
            manual_location="Near local landmark",
            is_demo=True,
        )

        # Update creation timestamp
        conn = db.get_connection()
        c = conn.cursor()
        if db.USE_POSTGRES:
            c.execute("UPDATE problems SET created_at = %s WHERE id = %s", (created_time, prob_id))
        else:
            c.execute("UPDATE problems SET created_at = ? WHERE id = ?", (created_time.isoformat(), prob_id))
        conn.commit()
        conn.close()

        # 2. Analyze
        print(f"  -> [{idx+1}/{len(SEED_PROBLEMS)}] Analyzing: {item['text'][:45]}...")
        analysis = analyze_problem_with_gemini(item["text"])
        db.save_analysis(prob_id, analysis)

        # 3. Embed & Cluster
        doc_text = build_embedding_text(
            problem_text=item["text"],
            summary=analysis.problem_summary,
            category=analysis.category,
            technical_discipline=analysis.technical_discipline,
            impact_areas=analysis.impact_areas,
            keywords=analysis.keywords,
        )
        vec = embed_text(doc_text)
        db.save_embedding(prob_id, vec, doc_text)

        # Candidate check & clustering
        from backend.main import run_similarity_pipeline
        from backend.matching_service import match_stakeholders_for_problem
        try:
            run_similarity_pipeline(prob_id, is_demo=True)
            
            # Step 4: Generate and persist stakeholder recommendations
            s4_recs = match_stakeholders_for_problem(
                category=analysis.category,
                sub_category=getattr(analysis.classification, "sub_category", "") if hasattr(analysis, "classification") else "",
                technical_disciplines=analysis.technical_discipline,
                urgency=analysis.urgency,
                has_image=False,
                cluster_member_count=1,
                city="New Delhi"
            )
            db.save_step4_recommendations(prob_id, s4_recs)
        except Exception as e:
            print(f"     Clustering / Step 4 note: {e}")

    print("[Seed] Done! Demo records with Step 4 Intelligent Matching successfully generated.")


if __name__ == "__main__":
    import sys
    clear_flag = "--clear" in sys.argv
    seed_demo_data(clear_only=clear_flag)
