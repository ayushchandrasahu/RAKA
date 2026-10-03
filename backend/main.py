import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AnalysisOutput, SimilarityPayload, MatchDetail, ClusterPayload, ClusterMemberSummary,
    MultimodalAnalysisOutput, Step4MatchingPayload,
    ChallengeReadiness, ChallengeRecord, ChatRequest, ChatResponse,
    SolutionCreate, SolutionRecord, ImplementationUpdateCreate, ImpactRecordCreate,
    FeedbackCreate, AppealCreate, AppealResolve, StakeholderReviewCreate, CollaborationRequestCreate, CollaborationResponse, CollaborationProgressUpdate, CollaborationTaskCreate, CollaborationMessageCreate
)
from backend.similarity import (
    CONFIG, build_embedding_text, score_pair, generate_reasons, plan_cluster_update
)
from backend.gemini_service import (
    analyze_problem_multimodal, analyze_problem_with_gemini, embed_text,
    transcribe_audio_file, detect_language_simple
)
from backend.matching_service import match_stakeholders_for_problem
from backend.challenge_service import calculate_challenge_readiness, generate_challenge_for_cluster
from backend.chat_service import process_chat_message, process_voice_chat
from backend import database as db

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

RAKA_DEMO_MODE = os.getenv("RAKA_DEMO_MODE", "0") == "1"


def run_similarity_pipeline(problem_id: str, is_demo: bool = False) -> Dict[str, Any]:
    """
    Executes Step 3:
    1. Fetches problem and AI analysis.
    2. Builds embedding document and generates 768d L2-normalized vector.
    3. Finds top candidates using pgvector / vector search.
    4. Evaluates multi-signal scores.
    5. Updates problem clusters deterministically.
    6. Updates similarity_status.
    """
    conn = db.get_connection()
    c = conn.cursor()

    if db.USE_POSTGRES:
        c.execute("SELECT id, problem_text, latitude, longitude, manual_location, created_at, status, voice_transcript FROM problems WHERE id = %s", (problem_id,))
        p_row = c.fetchone()
        c.execute("SELECT category, problem_summary, technical_discipline, impact_areas, keywords, image_observations, sub_category FROM problem_ai_analysis WHERE problem_id = %s", (problem_id,))
        a_row = c.fetchone()
    else:
        c.execute("SELECT id, problem_text, latitude, longitude, manual_location, created_at, status, voice_transcript FROM problems WHERE id = ?", (problem_id,))
        p_row = c.fetchone()
        c.execute("SELECT category, problem_summary, technical_discipline, impact_areas, keywords, image_observations, sub_category FROM problem_ai_analysis WHERE problem_id = ?", (problem_id,))
        a_row = c.fetchone()

    conn.close()

    if not p_row or not a_row:
        raise ValueError(f"Problem or analysis not found for ID {problem_id}")

    # Parse fields
    p_text = p_row["problem_text"] if db.USE_POSTGRES else p_row[1]
    p_lat = p_row["latitude"] if db.USE_POSTGRES else p_row[2]
    p_lng = p_row["longitude"] if db.USE_POSTGRES else p_row[3]
    p_manual = p_row["manual_location"] if db.USE_POSTGRES else p_row[4]
    p_created = p_row["created_at"] if db.USE_POSTGRES else p_row[5]
    p_voice = p_row.get("voice_transcript") if db.USE_POSTGRES else (p_row[7] if len(p_row) > 7 else "")

    cat = a_row["category"] if db.USE_POSTGRES else a_row[0]
    summary = a_row["problem_summary"] if db.USE_POSTGRES else a_row[1]
    
    td_raw = a_row["technical_discipline"] if db.USE_POSTGRES else a_row[2]
    td = json.loads(td_raw) if isinstance(td_raw, str) else (td_raw or [])
    
    imp_raw = a_row["impact_areas"] if db.USE_POSTGRES else a_row[3]
    impact = json.loads(imp_raw) if isinstance(imp_raw, str) else (imp_raw or [])

    kw_raw = a_row["keywords"] if db.USE_POSTGRES else a_row[4]
    kws = json.loads(kw_raw) if isinstance(kw_raw, str) else (kw_raw or [])

    obs_raw = a_row["image_observations"] if db.USE_POSTGRES else a_row[5]
    obs = json.loads(obs_raw) if isinstance(obs_raw, str) else (obs_raw or [])

    sub_cat = (a_row.get("sub_category") if db.USE_POSTGRES else (a_row[6] if len(a_row) > 6 else "")) or ""

    # 1. Build embedding document
    doc_text = build_embedding_text(
        problem_text=p_text,
        summary=summary,
        category=cat,
        technical_discipline=td,
        impact_areas=impact,
        keywords=kws,
        image_observations=obs,
        manual_location=p_manual,
        sub_category=sub_cat,
        voice_transcript=p_voice,
    )


    # 2. Generate and store embedding
    try:
        vec = embed_text(doc_text)
        db.save_embedding(problem_id, vec, doc_text)
    except Exception as e:
        db.update_similarity_status(problem_id, "FAILED")
        raise e

    # 3. Retrieve top K candidates via vector search
    candidates = db.get_candidates(problem_id, vec, is_demo=is_demo, limit=CONFIG["top_k"])

    # 4. Multi-signal scoring for candidates
    scored_candidates = []
    base_prob = {
        "id": problem_id,
        "category": cat,
        "technical_discipline": td,
        "latitude": p_lat,
        "longitude": p_lng,
        "created_at": p_created,
    }

    for cand in candidates:
        pair_score = score_pair(base_prob, cand, cand["cosine"])
        reasons = generate_reasons(pair_score)
        
        # Round distance
        dist_m = pair_score["signals"]["distance_m"]
        if dist_m is None:
            dist_str = "Not available"
        elif dist_m < 1000:
            dist_str = f"{round(dist_m, -1):.0f} m"
        else:
            dist_str = f"{(dist_m / 1000.0):.1f} km"

        # Temporal display
        temp_val = pair_score["signals"]["temporal"]
        if temp_val >= 0.66:
            temp_str = "High"
        elif temp_val >= 0.33:
            temp_str = "Medium"
        else:
            temp_str = "Low"

        # Discipline and Category display
        cat_str = "Same" if pair_score["signals"]["category"] == 1.0 else ("Related" if pair_score["signals"]["category"] > 0 else "Different")
        disc_val = pair_score["signals"]["discipline"]
        disc_str = "Same" if disc_val == 1.0 else ("Overlap" if disc_val > 0 else "Different")

        created_str = str(cand["created_at"]).split()[0] if cand["created_at"] else "Recently"

        scored_candidates.append({
            "problem_id": cand["problem_id"],
            "combined_score": pair_score["combined_score"],
            "match_level": pair_score["match_level"],
            "percentage": int(round(pair_score["combined_score"] * 100)),
            "other_summary": cand["problem_summary"] or cand["category"],
            "other_category": cand["category"],
            "other_date": created_str,
            "other_status": cand["status"],
            "semantic_similarity": round(pair_score["signals"]["semantic"], 2),
            "category_match": cat_str,
            "discipline_match": disc_str,
            "distance_display": dist_str,
            "temporal_display": temp_str,
            "reasons": reasons,
        })

    # 5. Cluster updates
    existing_memberships, existing_clusters = db.get_cluster_state()
    plan = plan_cluster_update(
        new_report_id=problem_id,
        candidate_scores=scored_candidates,
        existing_memberships=existing_memberships,
        existing_clusters=existing_clusters,
    )
    db.apply_cluster_plan(plan, problem_id, cat, summary)

    # 6. Set similarity status to DONE
    db.update_similarity_status(problem_id, "DONE")
    return {"candidates": scored_candidates, "plan": plan}


def build_similarity_payload(problem_id: str) -> Dict[str, Any]:
    """Builds clean, privacy-preserving similarity result payload."""
    conn = db.get_connection()
    c = conn.cursor()

    if db.USE_POSTGRES:
        c.execute("SELECT similarity_status, is_demo FROM problems WHERE id = %s", (problem_id,))
        p_row = c.fetchone()
    else:
        c.execute("SELECT similarity_status, is_demo FROM problems WHERE id = ?", (problem_id,))
        p_row = c.fetchone()

    if not p_row:
        conn.close()
        return {"status": "FAILED", "headline_message": "Problem not found", "related_count": 0, "matches": []}

    sim_status = p_row["similarity_status"] if db.USE_POSTGRES else p_row[0]
    is_demo = bool(p_row["is_demo"] if db.USE_POSTGRES else p_row[1])

    if sim_status == "FAILED":
        conn.close()
        return {
            "status": "FAILED",
            "headline_message": "Similarity check could not be completed right now.",
            "related_count": 0,
            "matches": []
        }

    # Find cluster membership for this problem
    if db.USE_POSTGRES:
        c.execute("""
            SELECT m.cluster_id, c.cluster_title, c.problem_count
            FROM problem_cluster_members m
            JOIN problem_clusters c ON c.id = m.cluster_id
            WHERE m.problem_id = %s;
        """, (problem_id,))
        cl_row = c.fetchone()
    else:
        c.execute("""
            SELECT m.cluster_id, c.cluster_title, c.problem_count
            FROM problem_cluster_members m
            JOIN problem_clusters c ON c.id = m.cluster_id
            WHERE m.problem_id = ?;
        """, (problem_id,))
        cl_row = c.fetchone()

    cluster_id = None
    cluster_title = None
    cluster_report_count = None
    if cl_row:
        cluster_id = str(cl_row["cluster_id"] if db.USE_POSTGRES else cl_row[0])
        cluster_title = cl_row["cluster_title"] if db.USE_POSTGRES else cl_row[1]
        cluster_report_count = cl_row["problem_count"] if db.USE_POSTGRES else cl_row[2]

    # Fetch stored embedding to compute top candidates
    if db.USE_POSTGRES:
        c.execute("SELECT embedding FROM problem_embeddings WHERE problem_id = %s", (problem_id,))
        emb_row = c.fetchone()
    else:
        c.execute("SELECT embedding FROM problem_embeddings WHERE problem_id = ?", (problem_id,))
        emb_row = c.fetchone()

    conn.close()

    if not emb_row:
        return {
            "status": "PENDING",
            "headline_message": "Analyzing related reports...",
            "related_count": 0,
            "matches": []
        }

    vec = emb_row["embedding"] if db.USE_POSTGRES else json.loads(emb_row[0])
    candidates = db.get_candidates(problem_id, vec, is_demo=is_demo, limit=CONFIG["top_k"])

    # Score and filter
    matches = []
    has_possible_duplicate = False

    # Get this problem details
    conn = db.get_connection()
    c = conn.cursor()
    if db.USE_POSTGRES:
        c.execute("SELECT p.latitude, p.longitude, p.created_at, a.category, a.technical_discipline FROM problems p JOIN problem_ai_analysis a ON a.problem_id = p.id WHERE p.id = %s", (problem_id,))
        curr = c.fetchone()
    else:
        c.execute("SELECT p.latitude, p.longitude, p.created_at, a.category, a.technical_discipline FROM problems p JOIN problem_ai_analysis a ON a.problem_id = p.id WHERE p.id = ?", (problem_id,))
        curr = c.fetchone()
    conn.close()

    if curr:
        p_lat = curr["latitude"] if db.USE_POSTGRES else curr[0]
        p_lng = curr["longitude"] if db.USE_POSTGRES else curr[1]
        p_created = curr["created_at"] if db.USE_POSTGRES else curr[2]
        cat = curr["category"] if db.USE_POSTGRES else curr[3]
        td_raw = curr["technical_discipline"] if db.USE_POSTGRES else curr[4]
        td = json.loads(td_raw) if isinstance(td_raw, str) else (td_raw or [])

        base_prob = {
            "id": problem_id, "category": cat, "technical_discipline": td,
            "latitude": p_lat, "longitude": p_lng, "created_at": p_created
        }

        for cand in candidates:
            res = score_pair(base_prob, cand, cand["cosine"])
            if res["match_level"] in ("POSSIBLE_DUPLICATE", "RELATED_PROBLEM"):
                if res["match_level"] == "POSSIBLE_DUPLICATE":
                    has_possible_duplicate = True

                reasons = generate_reasons(res)
                dist_m = res["signals"]["distance_m"]
                dist_str = "Not available" if dist_m is None else (f"{round(dist_m, -1):.0f} m" if dist_m < 1000 else f"{(dist_m/1000):.1f} km")
                temp_str = "High" if res["signals"]["temporal"] >= 0.66 else ("Medium" if res["signals"]["temporal"] >= 0.33 else "Low")
                cat_str = "Same" if res["signals"]["category"] == 1.0 else ("Related" if res["signals"]["category"] > 0 else "Different")
                disc_str = "Same" if res["signals"]["discipline"] == 1.0 else ("Overlap" if res["signals"]["discipline"] > 0 else "Different")
                date_str = str(cand["created_at"]).split()[0] if cand["created_at"] else "Recently"

                matches.append(MatchDetail(
                    match_level=res["match_level"],
                    combined_score=round(res["combined_score"], 3),
                    percentage=int(round(res["combined_score"] * 100)),
                    other_summary=cand["problem_summary"] or cand["category"],
                    other_category=cand["category"],
                    other_date=date_str,
                    other_status=cand["status"],
                    semantic_similarity=round(res["signals"]["semantic"], 2),
                    category_match=cat_str,
                    discipline_match=disc_str,
                    distance_display=dist_str,
                    temporal_display=temp_str,
                    reasons=reasons
                ))

    # Sort matches by percentage descending
    matches.sort(key=lambda m: m.percentage, reverse=True)
    top_matches = matches[:3]

    # Smart message logic (Section 20)
    if has_possible_duplicate:
        headline = "RAKA found an existing report that may describe the same underlying issue."
    elif len(matches) > 0:
        headline = f"RAKA found {len(matches)} reports describing similar problems in the area."
    elif len(candidates) > 0:
        headline = "No strong related report was found."
    else:
        headline = "No related reports found yet."

    return {
        "status": "DONE",
        "headline_message": headline,
        "related_count": len(matches),
        "cluster_id": cluster_id,
        "cluster_title": cluster_title,
        "cluster_report_count": cluster_report_count,
        "matches": [m.model_dump() for m in top_matches]
    }


def build_cluster_payload(cluster_id: str) -> Optional[Dict[str, Any]]:
    """Builds cluster projection for /clusters/{id} (Section 17)."""
    conn = db.get_connection()
    c = conn.cursor()

    if db.USE_POSTGRES:
        c.execute("""
            SELECT id, cluster_title, category, primary_problem_id, problem_count, status, needs_review, created_at
            FROM problem_clusters WHERE id = %s;
        """, (cluster_id,))
        cl = c.fetchone()
    else:
        c.execute("""
            SELECT id, cluster_title, category, primary_problem_id, problem_count, status, needs_review, created_at
            FROM problem_clusters WHERE id = ?;
        """, (cluster_id,))
        cl = c.fetchone()

    if not cl:
        conn.close()
        return None

    # Fetch members with analysis details
    if db.USE_POSTGRES:
        c.execute("""
            SELECT m.problem_id, m.membership_score, m.created_at as joined_at,
                   p.latitude, p.longitude, p.created_at as reported_at, p.status as p_status,
                   a.category, a.problem_summary, a.technical_discipline, a.impact_areas
            FROM problem_cluster_members m
            JOIN problems p ON p.id = m.problem_id
            LEFT JOIN problem_ai_analysis a ON a.problem_id = p.id
            WHERE m.cluster_id = %s
            ORDER BY p.created_at ASC;
        """, (cluster_id,))
        member_rows = c.fetchall()
    else:
        c.execute("""
            SELECT m.problem_id, m.membership_score, m.created_at as joined_at,
                   p.latitude, p.longitude, p.created_at as reported_at, p.status as p_status,
                   a.category, a.problem_summary, a.technical_discipline, a.impact_areas
            FROM problem_cluster_members m
            JOIN problems p ON p.id = m.problem_id
            LEFT JOIN problem_ai_analysis a ON a.problem_id = p.id
            WHERE m.cluster_id = ?
            ORDER BY p.created_at ASC;
        """, (cluster_id,))
        member_rows = c.fetchall()

    conn.close()

    # Aggregate metadata
    impact_counter: Dict[str, int] = {}
    discipline_counter: Dict[str, int] = {}
    lats = []
    lngs = []
    scores = []
    recent_30d_count = 0
    now = datetime.datetime.now(datetime.timezone.utc)
    members_list = []

    for r in member_rows:
        score = float(r["membership_score"] if db.USE_POSTGRES else r[1])
        scores.append(score)
        
        lat = r["latitude"] if db.USE_POSTGRES else r[3]
        lng = r["longitude"] if db.USE_POSTGRES else r[4]
        if lat is not None and lng is not None:
            lats.append(lat)
            lngs.append(lng)

        rep_at = r["reported_at"] if db.USE_POSTGRES else r[5]
        if rep_at:
            try:
                # Handle ISO format or sqlite timestamp
                dt = datetime.datetime.fromisoformat(str(rep_at).replace("Z", "+00:00"))
                if (now - dt.replace(tzinfo=datetime.timezone.utc if dt.tzinfo is None else dt.tzinfo)).days <= 30:
                    recent_30d_count += 1
            except Exception:
                recent_30d_count += 1

        cat = r["category"] if db.USE_POSTGRES else r[7]
        summary = r["problem_summary"] if db.USE_POSTGRES else r[8]
        p_status = r["p_status"] if db.USE_POSTGRES else r[6]
        
        td_raw = r["technical_discipline"] if db.USE_POSTGRES else r[9]
        td = json.loads(td_raw) if isinstance(td_raw, str) else (td_raw or [])
        for item in td:
            discipline_counter[item] = discipline_counter.get(item, 0) + 1

        imp_raw = r["impact_areas"] if db.USE_POSTGRES else r[10]
        imp = json.loads(imp_raw) if isinstance(imp_raw, str) else (imp_raw or [])
        for item in imp:
            impact_counter[item] = impact_counter.get(item, 0) + 1

        members_list.append(ClusterMemberSummary(
            problem_summary=summary or f"Report regarding {cat}",
            category=cat or "Civic Infrastructure",
            reported_date=str(rep_at).split()[0] if rep_at else "Recent",
            status=p_status or "AI_ANALYZED",
            link_percentage=int(round(score * 100)) if score <= 1.0 else int(round(score))
        ))

    # Generalized area (centroid rounded to 2 decimal places ~1km)
    if lats and lngs:
        avg_lat = sum(lats) / len(lats)
        avg_lng = sum(lngs) / len(lngs)
        gen_area = f"Zone approx {avg_lat:.2f}°N, {avg_lng:.2f}°E"
    else:
        gen_area = "Regional area (GPS coordinates not provided)"

    avg_sim = int(round((sum(scores) / len(scores)) * 100)) if scores else 85
    needs_review = bool(cl["needs_review"] if db.USE_POSTGRES else cl[6])

    why_grouped = [
        {"signal": "Semantic similarity", "evidence": "High - Reports describe related public conditions"},
        {"signal": "Category match", "evidence": f"Same domain ({cl['category'] if db.USE_POSTGRES else cl[2]})"},
        {"signal": "Technical discipline", "evidence": ", ".join(list(discipline_counter.keys())[:2]) or "Civic Engineering"},
        {"signal": "Location proximity", "evidence": gen_area},
        {"signal": "Temporal relevance", "evidence": f"{recent_30d_count} reports submitted in the last 30 days"}
    ]

    return {
        "id": str(cl["id"] if db.USE_POSTGRES else cl[0]),
        "title": cl["cluster_title"] if db.USE_POSTGRES else cl[1],
        "category": cl["category"] if db.USE_POSTGRES else cl[2],
        "problem_count": len(member_rows),
        "recent_count_30d": recent_30d_count,
        "status": "Under validation, not yet reviewed by any authority",
        "needs_review": needs_review,
        "review_notice": "This group is large or mixed and may contain more than one issue." if needs_review else None,
        "generalized_area": gen_area,
        "average_similarity_pct": avg_sim,
        "impact_areas": sorted(impact_counter.keys(), key=lambda k: impact_counter[k], reverse=True)[:4],
        "technical_disciplines": sorted(discipline_counter.keys(), key=lambda k: discipline_counter[k], reverse=True)[:3],
        "why_grouped": why_grouped,
        "members": [m.model_dump() for m in members_list]
    }


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield

app = FastAPI(title="RAKA Domain AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# Web SPA Routes
@app.get("/")
async def serve_index():
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_file))


@app.get("/hero-bg.jpg")
async def serve_hero_bg():
    bg_file = BASE_DIR / "hero-bg.jpg"
    if bg_file.exists():
        return FileResponse(str(bg_file))
    raise HTTPException(status_code=404, detail="hero-bg.jpg not found")


@app.get("/problems")
@app.get("/problems/{problem_id}/analysis")
async def serve_spa_result(problem_id: Optional[str] = None):
    index_file = BASE_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/clusters")
@app.get("/clusters/{cluster_id}")
async def serve_spa_cluster(cluster_id: Optional[str] = None):
    index_file = BASE_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/report")
@app.get("/judge")
async def serve_spa_other():
    index_file = BASE_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/challenges")
async def serve_spa_challenges():
    index_file = BASE_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/challenges/{challenge_id}")
async def serve_spa_challenge_detail(challenge_id: str):
    index_file = BASE_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/solutions")
@app.get("/solutions/{solution_id}")
async def serve_spa_solutions(solution_id: Optional[str] = None):
    index_file = BASE_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/admin")
async def serve_spa_admin():
    index_file = BASE_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/analytics")
async def serve_spa_analytics():
    index_file = BASE_DIR / "index.html"
    return FileResponse(str(index_file))


# API Endpoints
@app.get("/api/health")
@app.get("/health")
async def health_check():
    """Service health check returning server and pipeline operational status."""
    return {
        "status": "healthy",
        "service": "RAKA Domain AI",
        "version": "4.0.0",
        "capabilities": ["multimodal_analysis", "multilingual_voice", "image_processing", "mandatory_location", "step3_clustering", "step4_intelligent_routing"],
        "database": "postgres" if db.USE_POSTGRES else "sqlite"
    }


@app.post("/api/voice/transcribe")
async def transcribe_voice_endpoint(
    audio: UploadFile = File(...),
    preferredLanguage: Optional[str] = Form("Hindi")
):
    """
    Step 1 Voice Feature: Real audio upload and multilingual transcription.
    Supports Hindi, Marathi, English, and code-mixed speech.
    """
    audio_id = str(uuid.uuid4())
    ext = Path(audio.filename or "recording.webm").suffix or ".webm"
    audio_filename = f"voice_{audio_id}{ext}"
    audio_path = UPLOADS_DIR / audio_filename
    
    content = await audio.read()
    with open(audio_path, "wb") as f:
        f.write(content)

    transcription, detected_lang, confidence = transcribe_audio_file(audio_path, preferred_language=preferredLanguage)
    
    return {
        "success": True,
        "audioUrl": f"/uploads/{audio_filename}",
        "transcription": transcription,
        "detectedLanguage": detected_lang,
        "confidence": confidence
    }


@app.post("/api/location/reverse")
async def reverse_geocode_endpoint(
    request: Request,
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None)
):
    """
    Part C: Reverse geocodes coordinates to a human-readable generalized location
    (e.g., 'Local area, Pune') without exposing exact GPS coordinates.
    """
    if latitude is None or longitude is None:
        try:
            body = await request.json()
            latitude = float(body.get("latitude"))
            longitude = float(body.get("longitude"))
        except Exception:
            pass

    if latitude is None or longitude is None:
        return {
            "success": True,
            "city": "Mumbai",
            "district": "Mumbai",
            "state": "Maharashtra",
            "locality": "Local Area",
            "displayName": "Local area, Mumbai"
        }

    import urllib.request
    city = "Pune"
    district = "Pune"
    state = "Maharashtra"
    locality = "Local Area"
    display_name = f"Local area, {city}"

    try:
        req = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}&zoom=14&addressdetails=1",
            headers={"User-Agent": "RAKA-Multimodal-Civic-AI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode())
            addr = data.get("address", {})
            locality = addr.get("suburb") or addr.get("neighbourhood") or addr.get("residential") or addr.get("village") or "Local Area"
            city = addr.get("city") or addr.get("town") or addr.get("municipality") or addr.get("county") or "Pune"
            district = addr.get("state_district") or addr.get("county") or city
            state = addr.get("state") or "Maharashtra"
            display_name = f"{locality}, {city}" if locality != "Local Area" else f"{city}, {state}"
    except Exception:
        # Fallback bounding box logic for Indian cities
        if 18.0 <= latitude <= 19.5 and 73.0 <= longitude <= 74.5:
            city, district, state = "Pune", "Pune", "Maharashtra"
            display_name = "Local area, Pune"
        elif 18.8 <= latitude <= 19.4 and 72.7 <= longitude <= 73.1:
            city, district, state = "Mumbai", "Mumbai", "Maharashtra"
            display_name = "Local area, Mumbai"
        elif 28.4 <= latitude <= 28.9 and 76.9 <= longitude <= 77.4:
            city, district, state = "New Delhi", "Central Delhi", "Delhi"
            display_name = "Local area, Delhi"
        elif 12.8 <= latitude <= 13.2 and 77.4 <= longitude <= 77.8:
            city, district, state = "Bengaluru", "Bengaluru Urban", "Karnataka"
            display_name = "Local area, Bengaluru"
        else:
            display_name = f"Local area ({round(latitude, 2)}, {round(longitude, 2)})"

    return {
        "success": True,
        "city": city,
        "district": district,
        "state": state,
        "locality": locality,
        "displayName": display_name
    }


@app.post("/api/problems/analyze")
async def analyze_problem(
    problemText: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    images: List[UploadFile] = File(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    locationAccuracy: Optional[float] = Form(None),
    manualLocation: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    district: Optional[str] = Form(None),
    state: Optional[str] = Form(None),
    locality: Optional[str] = Form(None),
    preferredLanguage: Optional[str] = Form("en"),
    editedTranscript: Optional[str] = Form(None),
):
    """
    Step 1 -> Step 2 -> Step 3 -> Step 4 Multimodal Pipeline.
    Combines Text + Voice + Vision + Mandatory Location into structured evidence.
    """
    # 1. Validate Mandatory Location (Part C, D, AA)
    has_coords = (latitude is not None and longitude is not None)
    has_manual = bool(manualLocation and manualLocation.strip()) or bool(city and city.strip()) or bool(state and state.strip())
    if not (has_coords or has_manual):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Location is required to analyze this problem. Please use GPS or enter your location manually."
            }
        )

    # 2. Validate Multimodal Problem Content (Part B)
    clean_text = (problemText or "").strip()
    clean_edited_voice = (editedTranscript or "").strip()
    has_audio_file = bool(audio and audio.filename)
    
    if not clean_text and not clean_edited_voice and not has_audio_file:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Please describe your problem by typing or speaking."
            }
        )

    problem_id = str(uuid.uuid4())
    audio_url = None
    voice_transcript = clean_edited_voice
    detected_lang = "English"

    # 3. Process Spoken Audio if supplied (Part F, G, H)
    if has_audio_file:
        audio_bytes = await audio.read()
        if len(audio_bytes) > 25 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Audio recording exceeds maximum allowable size of 25MB."}
            )
        audio_ext = Path(audio.filename).suffix.lower() or ".webm"
        audio_name = f"{problem_id}_voice{audio_ext}"
        saved_audio_path = UPLOADS_DIR / audio_name
        with open(saved_audio_path, "wb") as f:
            f.write(audio_bytes)
        audio_url = f"/uploads/{audio_name}"

        # If citizen did not provide an edited transcript, transcribe now
        if not voice_transcript:
            trans, det_l, _ = transcribe_audio_file(saved_audio_path, preferred_language=preferredLanguage)
            voice_transcript = trans
            detected_lang = det_l
        else:
            detected_lang = preferredLanguage or "Hindi"
    elif voice_transcript:
        detected_lang, _ = detect_language_simple(voice_transcript)
    elif clean_text:
        detected_lang, _ = detect_language_simple(clean_text)

    # 4. Process Images (Part I, J, AQ)
    uploaded_files = []
    if images and len(images) > 0 and images[0].filename:
        uploaded_files.extend(images[:5])
    elif image and image.filename:
        uploaded_files.append(image)

    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    saved_image_paths: List[Path] = []
    primary_image_url = None

    for idx, img in enumerate(uploaded_files):
        ext = Path(img.filename or "").suffix.lower()
        if not ext or ext not in allowed_exts:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Unsupported image format ({ext or 'unknown'}). Allowed formats: JPG, JPEG, PNG, WEBP, HEIC."}
            )
        content = await img.read()
        if len(content) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Image file exceeds maximum allowable size of 10MB."}
            )
        img_filename = f"{problem_id}_{idx}{ext}"
        img_path = UPLOADS_DIR / img_filename
        with open(img_path, "wb") as f:
            f.write(content)
        saved_image_paths.append(img_path)
        if not primary_image_url:
            primary_image_url = f"/uploads/{img_filename}"

    # 5. Save Initial Problem Record (Part AL)
    db.save_problem(
        problem_id=problem_id,
        problem_text=clean_text or voice_transcript,
        image_url=primary_image_url,
        has_location=has_coords,
        latitude=latitude,
        longitude=longitude,
        location_accuracy=locationAccuracy,
        manual_location=manualLocation or f"{locality or ''}, {city or ''}, {state or ''}".strip(", "),
        is_demo=RAKA_DEMO_MODE,
        preferred_language=preferredLanguage or "en",
        audio_url=audio_url,
        voice_transcript=voice_transcript,
        detected_language=detected_lang,
        city=city or "",
        district=district or "",
        state=state or "",
        locality=locality or "",
        location_status="CAPTURED"
    )

    try:
        # 6. Step 2 Multimodal AI Analysis (Part K, L, M)
        analysis: MultimodalAnalysisOutput = analyze_problem_multimodal(
            text=clean_text,
            voice_transcript=voice_transcript,
            detected_lang=detected_lang,
            has_image=bool(saved_image_paths),
            image_paths=saved_image_paths,
            city=city or "",
            district=district or "",
            state=state or "",
            locality=locality or ""
        )
        db.save_analysis(problem_id, analysis)

        # 7. Step 3 Similarity & Problem Clustering (Part W)
        cluster_count = 1
        try:
            sim_res = run_similarity_pipeline(problem_id, is_demo=RAKA_DEMO_MODE)
            plan = sim_res.get("plan", {})
            if plan.get("action") in ("ADD_TO_EXISTING", "MERGE_CLUSTERS", "CREATE_NEW"):
                cluster_count = len(plan.get("member_ids_to_add", [])) + 1
        except Exception as sim_err:
            print(f"[Warning] Step 3 Similarity pipeline deferred: {sim_err}")
            db.update_similarity_status(problem_id, "FAILED")

        # 8. Step 4 Intelligent Stakeholder & Action Area Matching (Part X, Y)
        step4_payload = match_stakeholders_for_problem(
            category=analysis.classification.category,
            sub_category=analysis.classification.sub_category,
            technical_disciplines=analysis.classification.technical_disciplines,
            urgency=analysis.classification.urgency,
            has_image=bool(saved_image_paths),
            cluster_member_count=cluster_count,
            city=city or ""
        )
        db.save_step4_recommendations(problem_id, step4_payload)

        return {
            "success": True,
            "problemId": problem_id,
            "primaryIssue": analysis.problem_understanding.primary_issue,
            "category": analysis.classification.category,
            "subCategory": analysis.classification.sub_category
        }
    except Exception as e:
        print(f"[Analysis Error] {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "problemId": problem_id, "error": f"Analysis failed: {str(e)}"}
        )


@app.post("/api/problems/{problem_id}/analyze")
async def retry_analyze_problem(problem_id: str):
    conn = db.get_connection()
    c = conn.cursor()
    if db.USE_POSTGRES:
        c.execute("""
            SELECT id, problem_text, image_url, is_demo, voice_transcript, detected_language,
                   city, district, state, locality
            FROM problems WHERE id = %s
        """, (problem_id,))
        row = c.fetchone()
    else:
        c.execute("""
            SELECT id, problem_text, image_url, is_demo, voice_transcript, detected_language,
                   city, district, state, locality
            FROM problems WHERE id = ?
        """, (problem_id,))
        row = c.fetchone()
    conn.close()

    if not row:
        return JSONResponse(status_code=404, content={"success": False, "error": "Problem not found."})

    p_text = row["problem_text"] if db.USE_POSTGRES else row[1]
    img_url = row["image_url"] if db.USE_POSTGRES else row[2]
    is_demo = bool(row["is_demo"] if db.USE_POSTGRES else row[3])
    voice_tr = (row["voice_transcript"] if db.USE_POSTGRES else row[4]) or ""
    det_lang = (row["detected_language"] if db.USE_POSTGRES else row[5]) or "English"
    city = (row["city"] if db.USE_POSTGRES else row[6]) or ""
    district = (row["district"] if db.USE_POSTGRES else row[7]) or ""
    state = (row["state"] if db.USE_POSTGRES else row[8]) or ""
    locality = (row["locality"] if db.USE_POSTGRES else row[9]) or ""

    saved_img_paths = []
    if img_url:
        img_p = BASE_DIR / img_url.lstrip("/")
        if img_p.exists():
            saved_img_paths.append(img_p)

    try:
        analysis = analyze_problem_multimodal(
            text=p_text,
            voice_transcript=voice_tr,
            detected_lang=det_lang,
            has_image=bool(saved_img_paths),
            image_paths=saved_img_paths,
            city=city,
            district=district,
            state=state,
            locality=locality
        )
        db.save_analysis(problem_id, analysis)

        cluster_count = 1
        try:
            sim_res = run_similarity_pipeline(problem_id, is_demo=is_demo)
            plan = sim_res.get("plan", {})
            if plan.get("action") in ("ADD_TO_EXISTING", "MERGE_CLUSTERS", "CREATE_NEW"):
                cluster_count = len(plan.get("member_ids_to_add", [])) + 1
        except Exception as sim_err:
            print(f"[Warning] Step 3 Similarity retry error: {sim_err}")
            db.update_similarity_status(problem_id, "FAILED")

        step4_payload = match_stakeholders_for_problem(
            category=analysis.classification.category,
            sub_category=analysis.classification.sub_category,
            technical_disciplines=analysis.classification.technical_disciplines,
            urgency=analysis.classification.urgency,
            has_image=bool(saved_img_paths),
            cluster_member_count=cluster_count,
            city=city
        )
        db.save_step4_recommendations(problem_id, step4_payload)

        return {"success": True, "problemId": problem_id}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "problemId": problem_id, "error": f"Retry failed: {str(e)}"}
        )


@app.post("/api/problems/{problem_id}/similarity")
async def retry_similarity_endpoint(problem_id: str):
    """Retries ONLY Step 3 similarity pipeline for a problem."""
    conn = db.get_connection()
    c = conn.cursor()
    if db.USE_POSTGRES:
        c.execute("SELECT is_demo FROM problems WHERE id = %s", (problem_id,))
        row = c.fetchone()
    else:
        c.execute("SELECT is_demo FROM problems WHERE id = ?", (problem_id,))
        row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Problem not found.")

    is_demo = bool(row["is_demo"] if db.USE_POSTGRES else row[0])

    try:
        run_similarity_pipeline(problem_id, is_demo=is_demo)
        return {"success": True, "problemId": problem_id}
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"success": False, "code": "SIMILARITY_FAILED", "error": f"Similarity check failed: {str(e)}"}
        )


@app.get("/api/problems/{problem_id}/analysis")
async def get_problem_analysis(problem_id: str):
    conn = db.get_connection()
    c = conn.cursor()

    if db.USE_POSTGRES:
        c.execute("""
            SELECT id, problem_text, image_url, has_location, latitude, longitude,
                   location_accuracy, manual_location, status, similarity_status,
                   preferred_language, audio_url, voice_transcript, detected_language,
                   city, district, state, locality
            FROM problems WHERE id = %s
        """, (problem_id,))
        p_row = c.fetchone()
    else:
        c.execute("""
            SELECT id, problem_text, image_url, has_location, latitude, longitude,
                   location_accuracy, manual_location, status, similarity_status,
                   preferred_language, audio_url, voice_transcript, detected_language,
                   city, district, state, locality
            FROM problems WHERE id = ?
        """, (problem_id,))
        p_row = c.fetchone()

    if not p_row:
        conn.close()
        return JSONResponse(status_code=404, content={"success": False, "error": "Report not found."})

    problem_data = {
        "id": p_row["id"] if db.USE_POSTGRES else p_row[0],
        "problemText": p_row["problem_text"] if db.USE_POSTGRES else p_row[1],
        "imageUrl": p_row["image_url"] if db.USE_POSTGRES else p_row[2],
        "hasLocation": bool(p_row["has_location"] if db.USE_POSTGRES else p_row[3]),
        "latitude": p_row["latitude"] if db.USE_POSTGRES else p_row[4],
        "longitude": p_row["longitude"] if db.USE_POSTGRES else p_row[5],
        "locationAccuracy": p_row["location_accuracy"] if db.USE_POSTGRES else p_row[6],
        "manualLocation": p_row["manual_location"] if db.USE_POSTGRES else p_row[7],
        "status": p_row["status"] if db.USE_POSTGRES else p_row[8],
        "similarityStatus": p_row["similarity_status"] if db.USE_POSTGRES else p_row[9],
        "preferredLanguage": (p_row["preferred_language"] if db.USE_POSTGRES else (p_row[10] if len(p_row) > 10 else "en")) or "en",
        "audioUrl": p_row["audio_url"] if db.USE_POSTGRES else (p_row[11] if len(p_row) > 11 else None),
        "voiceTranscript": p_row["voice_transcript"] if db.USE_POSTGRES else (p_row[12] if len(p_row) > 12 else None),
        "detectedLanguage": p_row["detected_language"] if db.USE_POSTGRES else (p_row[13] if len(p_row) > 13 else None),
        "city": p_row["city"] if db.USE_POSTGRES else (p_row[14] if len(p_row) > 14 else ""),
        "district": p_row["district"] if db.USE_POSTGRES else (p_row[15] if len(p_row) > 15 else ""),
        "state": p_row["state"] if db.USE_POSTGRES else (p_row[16] if len(p_row) > 16 else ""),
        "locality": p_row["locality"] if db.USE_POSTGRES else (p_row[17] if len(p_row) > 17 else ""),
    }

    if db.USE_POSTGRES:
        c.execute("""
            SELECT category, problem_summary, urgency, complexity, technical_discipline, confidence,
                   suggested_departments, image_observations, impact_areas, required_skills,
                   possible_solution_types, estimated_project_duration, keywords, reasoning_summary,
                   problem_title, sub_category, raw_analysis_json, evidence, classification_confidence,
                   image_confidence, language_confidence
            FROM problem_ai_analysis WHERE problem_id = %s
        """, (problem_id,))
        a_row = c.fetchone()
    else:
        c.execute("""
            SELECT category, problem_summary, urgency, complexity, technical_discipline, confidence,
                   suggested_departments, image_observations, impact_areas, required_skills,
                   possible_solution_types, estimated_project_duration, keywords, reasoning_summary,
                   problem_title, sub_category, raw_analysis_json, evidence, classification_confidence,
                   image_confidence, language_confidence
            FROM problem_ai_analysis WHERE problem_id = ?
        """, (problem_id,))
        a_row = c.fetchone()

    conn.close()

    if not a_row:
        return {"success": True, "problem": problem_data, "analysis": None, "similarity": None, "step4": None}

    def _parse_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return []
        return []

    raw_json_str = (a_row["raw_analysis_json"] if db.USE_POSTGRES else (a_row[16] if len(a_row) > 16 else "")) or ""
    multimodal_obj = {}
    if raw_json_str:
        try:
            multimodal_obj = json.loads(raw_json_str)
        except Exception:
            pass

    analysis_data = {
        "problem_title": (a_row["problem_title"] if db.USE_POSTGRES else (a_row[14] if len(a_row) > 14 else "")) or (multimodal_obj.get("problem_title") or "Reported Civic Problem"),
        "sub_category": (a_row["sub_category"] if db.USE_POSTGRES else (a_row[15] if len(a_row) > 15 else "")) or (multimodal_obj.get("classification", {}).get("sub_category", "")),
        "category": a_row["category"] if db.USE_POSTGRES else a_row[0],
        "problem_summary": a_row["problem_summary"] if db.USE_POSTGRES else a_row[1],
        "urgency": a_row["urgency"] if db.USE_POSTGRES else a_row[2],
        "complexity": a_row["complexity"] if db.USE_POSTGRES else a_row[3],
        "technical_discipline": _parse_list(a_row["technical_discipline"] if db.USE_POSTGRES else a_row[4]),
        "confidence": a_row["confidence"] if db.USE_POSTGRES else a_row[5],
        "suggested_departments": _parse_list(a_row["suggested_departments"] if db.USE_POSTGRES else a_row[6]),
        "image_observations": _parse_list(a_row["image_observations"] if db.USE_POSTGRES else a_row[7]),
        "impact_areas": _parse_list(a_row["impact_areas"] if db.USE_POSTGRES else a_row[8]),
        "required_skills": _parse_list(a_row["required_skills"] if db.USE_POSTGRES else a_row[9]),
        "possible_solution_types": _parse_list(a_row["possible_solution_types"] if db.USE_POSTGRES else a_row[10]),
        "estimated_project_duration": a_row["estimated_project_duration"] if db.USE_POSTGRES else a_row[11],
        "keywords": _parse_list(a_row["keywords"] if db.USE_POSTGRES else a_row[12]),
        "reasoning_summary": a_row["reasoning_summary"] if db.USE_POSTGRES else a_row[13],
        "classification_confidence": a_row["classification_confidence"] if db.USE_POSTGRES else (a_row[18] if len(a_row) > 18 else 0.92),
        "image_confidence": a_row["image_confidence"] if db.USE_POSTGRES else (a_row[19] if len(a_row) > 19 else 0.88),
        "language_confidence": a_row["language_confidence"] if db.USE_POSTGRES else (a_row[20] if len(a_row) > 20 else 0.95),
    }

    similarity_data = build_similarity_payload(problem_id)

    step4_data = db.get_step4_recommendations(problem_id)
    if not step4_data:
        step4_payload = match_stakeholders_for_problem(
            category=analysis_data["category"],
            sub_category=analysis_data["sub_category"],
            technical_disciplines=analysis_data["technical_discipline"],
            urgency=analysis_data["urgency"],
            has_image=bool(problem_data["imageUrl"]),
            cluster_member_count=similarity_data.get("related_count", 0) + 1,
            city=problem_data.get("city", "")
        )
        db.save_step4_recommendations(problem_id, step4_payload)
        step4_data = step4_payload.model_dump()

    evidence_list = multimodal_obj.get("evidence", [])
    if not evidence_list:
        evidence_list = [
            {"source": "TEXT", "statement": f"Citizen statement recorded: {problem_data['problemText'][:80]}", "confidence": 0.95},
            {"source": "LOCATION", "statement": f"Geolocated to {problem_data['locality'] or problem_data['city'] or 'Pune, Maharashtra'}", "confidence": 0.99}
        ]
        if problem_data["voiceTranscript"]:
            evidence_list.append({"source": "VOICE", "statement": f"Spoken input ({problem_data['detectedLanguage'] or 'Hindi'}): {problem_data['voiceTranscript'][:80]}", "confidence": 0.92})
        if analysis_data["image_observations"]:
            evidence_list.append({"source": "IMAGE", "statement": f"Visual observation: {'; '.join(analysis_data['image_observations'][:2])}", "confidence": 0.89})

    return {
        "success": True,
        "problem": problem_data,
        "analysis": analysis_data,
        "similarity": similarity_data,
        "step4": step4_data,
        "multimodal": multimodal_obj or {
            "problem_title": analysis_data["problem_title"],
            "problem_summary": analysis_data["problem_summary"],
            "citizen_statement": {
                "text": problem_data["problemText"],
                "voice_transcript": problem_data["voiceTranscript"] or "",
                "detected_language": problem_data["detectedLanguage"] or "English"
            },
            "location_context": {
                "city": problem_data["city"],
                "district": problem_data["district"],
                "state": problem_data["state"],
                "locality": problem_data["locality"],
                "location_available": problem_data["hasLocation"]
            },
            "visual_analysis": {
                "image_available": bool(problem_data["imageUrl"]),
                "visible_problem_evidence": analysis_data["image_observations"],
                "image_confidence": analysis_data["image_confidence"]
            },
            "classification": {
                "category": analysis_data["category"],
                "sub_category": analysis_data["sub_category"],
                "technical_disciplines": analysis_data["technical_discipline"],
                "urgency": analysis_data["urgency"],
                "complexity": analysis_data["complexity"]
            },
            "impact": {
                "impact_areas": analysis_data["impact_areas"],
                "severity_reason": f"Impact observed across {', '.join(analysis_data['impact_areas'][:2])}"
            },
            "routing_context": {
                "suggested_departments": analysis_data["suggested_departments"]
            },
            "evidence": evidence_list,
            "uncertainty": {
                "missing_information": [],
                "uncertain_fields": []
            },
            "confidence": analysis_data["confidence"]
        }
    }



@app.get("/api/clusters/{cluster_id}")
async def get_cluster_endpoint(cluster_id: str):
    cluster = build_cluster_payload(cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail="Problem cluster not found.")
    return {"success": True, "cluster": cluster}


@app.get("/api/stats")
async def get_stats():
    conn = db.get_connection()
    c = conn.cursor()

    demo_filter_sql = "" if RAKA_DEMO_MODE else "WHERE is_demo = 0"
    demo_filter_sql_pg = "" if RAKA_DEMO_MODE else "WHERE is_demo = false"

    if db.USE_POSTGRES:
        c.execute(f"SELECT COUNT(*) as cnt FROM problems {demo_filter_sql_pg};")
        total_reports = c.fetchone()["cnt"]

        c.execute(f"SELECT COUNT(*) as cnt FROM problems WHERE status = 'AI_ANALYZED' {'AND is_demo = false' if not RAKA_DEMO_MODE else ''};")
        analyzed = c.fetchone()["cnt"]

        c.execute("SELECT COUNT(*) as cnt FROM problem_clusters;")
        total_clusters = c.fetchone()["cnt"]

        c.execute("SELECT COUNT(*) as cnt FROM problem_cluster_members;")
        clustered_reports = c.fetchone()["cnt"]

        c.execute("""
            SELECT a.category, COUNT(*) as cnt
            FROM problem_ai_analysis a
            JOIN problems p ON p.id = a.problem_id
            WHERE p.status = 'AI_ANALYZED'
            GROUP BY a.category ORDER BY cnt DESC;
        """)
        cat_rows = c.fetchall()
        by_category = [{"category": row["category"], "count": row["cnt"]} for row in cat_rows]
    else:
        c.execute(f"SELECT COUNT(*) FROM problems {demo_filter_sql};")
        total_reports = c.fetchone()[0]

        c.execute(f"SELECT COUNT(*) FROM problems WHERE status = 'AI_ANALYZED' {'AND is_demo = 0' if not RAKA_DEMO_MODE else ''};")
        analyzed = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM problem_clusters;")
        total_clusters = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM problem_cluster_members;")
        clustered_reports = c.fetchone()[0]

        c.execute("""
            SELECT a.category, COUNT(*)
            FROM problem_ai_analysis a
            JOIN problems p ON p.id = a.problem_id
            WHERE p.status = 'AI_ANALYZED'
            GROUP BY a.category ORDER BY COUNT(*) DESC;
        """)
        cat_rows = c.fetchall()
        by_category = [{"category": row[0], "count": row[1]} for row in cat_rows]

    conn.close()

    return {
        "totalReports": total_reports,
        "analyzed": analyzed,
        "clustersCount": total_clusters,
        "clusteredReports": clustered_reports,
        "byCategory": by_category
    }


# ===========================================================================
# STEP 5: INTELLIGENT CHALLENGES & MULTILINGUAL CHATBOT ENDPOINTS
# ===========================================================================

@app.get("/api/clusters/{cluster_id}/challenge-readiness")
async def get_cluster_readiness_endpoint(cluster_id: str):
    """
    Evaluates whether a problem cluster has sufficient structured evidence
    to be converted into an innovation challenge.
    """
    try:
        readiness = calculate_challenge_readiness(cluster_id)
        return {
            "success": True,
            "clusterId": cluster_id,
            "readiness": readiness.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/challenges/generate")
async def generate_challenge_endpoint(payload: Dict[str, Any]):
    """
    Module A: Synthesizes a verified problem cluster into an actionable challenge.
    Enforces strict anti-hallucination guardrails and Pydantic validation.
    """
    cluster_id = payload.get("clusterId")
    if not cluster_id:
        raise HTTPException(status_code=400, detail="clusterId is required to generate a challenge.")

    try:
        challenge_record = generate_challenge_for_cluster(cluster_id)
        return {
            "success": True,
            "challenge": challenge_record.model_dump()
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Challenge generation failed: {str(e)}")


@app.get("/api/challenges")
async def list_challenges_endpoint(
    search: Optional[str] = None,
    category: Optional[str] = None,
    subCategory: Optional[str] = None,
    status: Optional[str] = None,
    urgency: Optional[str] = None,
    discipline: Optional[str] = None,
    readiness: Optional[str] = None
):
    """
    Module A: Challenge Dashboard endpoint returning live filtered challenges
    and real database statistics without hardcoded metrics.
    """
    try:
        challenges, stats = db.list_challenges(
            search=search or "",
            category=category or "",
            sub_category=subCategory or "",
            status=status or "",
            urgency=urgency or "",
            discipline=discipline or "",
            readiness=readiness or ""
        )
        return {
            "success": True,
            "challenges": challenges,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/challenges/{challenge_id}")
async def get_challenge_detail_endpoint(challenge_id: str):
    """
    Module A: Detailed challenge view including evidence breakdown, cluster context,
    and contributing citizen reports.
    """
    ch = db.get_challenge(challenge_id)
    if not ch:
        raise HTTPException(status_code=404, detail=f"Challenge with ID '{challenge_id}' not found.")

    evidence = db.get_challenge_evidence(challenge_id)
    ch["evidence"] = evidence

    cluster_info = None
    if ch.get("cluster_id"):
        cluster_info = build_cluster_payload(ch["cluster_id"])

    return {
        "success": True,
        "challenge": ch,
        "cluster": cluster_info
    }


@app.patch("/api/challenges/{challenge_id}")
async def update_challenge_endpoint(challenge_id: str, updates: Dict[str, Any]):
    """
    Module A: Allows authorized editing of challenge description, statement, or urgency.
    """
    ok = db.update_challenge(challenge_id, updates)
    if not ok:
        raise HTTPException(status_code=400, detail="No valid editable fields provided or challenge not found.")
    updated_ch = db.get_challenge(challenge_id)
    return {
        "success": True,
        "challenge": updated_ch
    }


@app.post("/api/challenges/{challenge_id}/validate")
async def validate_challenge_endpoint(challenge_id: str):
    """
    Transitions challenge status from DRAFT / VALIDATION_REQUIRED to VALIDATED.
    """
    ch = db.get_challenge(challenge_id)
    if not ch:
        raise HTTPException(status_code=404, detail="Challenge not found.")
    db.update_challenge_status(challenge_id, "VALIDATED")
    return {
        "success": True,
        "challengeId": challenge_id,
        "challenge_status": "VALIDATED"
    }


@app.post("/api/challenges/{challenge_id}/publish")
async def publish_challenge_endpoint(challenge_id: str):
    """
    Transitions challenge status to PUBLISHED, exposing it to the open innovation ecosystem.
    """
    ch = db.get_challenge(challenge_id)
    if not ch:
        raise HTTPException(status_code=404, detail="Challenge not found.")
    db.update_challenge_status(challenge_id, "PUBLISHED")
    return {
        "success": True,
        "challengeId": challenge_id,
        "challenge_status": "PUBLISHED"
    }


@app.post("/api/challenges/{challenge_id}/retry")
async def retry_challenge_endpoint(challenge_id: str):
    """
    Retries structured challenge generation for the associated cluster.
    """
    ch = db.get_challenge(challenge_id)
    if not ch or not ch.get("cluster_id"):
        raise HTTPException(status_code=404, detail="Challenge or associated cluster not found.")
    new_ch = generate_challenge_for_cluster(ch["cluster_id"])
    return {
        "success": True,
        "challenge": new_ch.model_dump()
    }


@app.post("/api/chat")
async def chat_message_endpoint(req: ChatRequest):
    """
    Module B: Multilingual, context-aware RAKA AI Assistant message handler.
    Integrates verified Problem, Cluster, or Challenge context.
    """
    try:
        resp = process_chat_message(req)
        return resp.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/api/chat/voice")
async def chat_voice_endpoint(
    audio: UploadFile = File(...),
    problemId: Optional[str] = Form(None),
    clusterId: Optional[str] = Form(None),
    challengeId: Optional[str] = Form(None),
    language: Optional[str] = Form("Hindi"),
    conversationId: Optional[str] = Form(None)
):
    """
    Module B: Real voice chat input handler.
    Performs speech-to-text, detects language, processes through contextual RAKA AI,
    and returns both the transcription and assistant answer.
    """
    try:
        audio_id = str(uuid.uuid4())
        ext = Path(audio.filename or "voice_chat.webm").suffix or ".webm"
        audio_filename = f"voice_chat_{audio_id}{ext}"
        audio_path = UPLOADS_DIR / audio_filename

        content = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(content)

        transcription, chat_resp = process_voice_chat(
            audio_path=audio_path,
            problem_id=problemId,
            cluster_id=clusterId,
            challenge_id=challengeId,
            language=language,
            conversation_id=conversationId
        )

        return {
            "success": True,
            "transcription": transcription,
            "chat": chat_resp.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice chat failed: {str(e)}")


@app.get("/api/chat/conversations/{conversation_id}")
async def get_conversation_endpoint(conversation_id: str):
    """
    Retrieves message history for a chat conversation.
    """
    conv = db.get_chat_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = db.get_chat_messages(conversation_id, limit=30)
    return {
        "success": True,
        "conversation": conv,
        "messages": messages
    }




# ===========================================================================
# MODULE 7 & 8: SOLUTIONS, IMPLEMENTATION & IMPACT TRACKING ENDPOINTS
# ===========================================================================

@app.get("/api/solutions")
async def list_solutions_endpoint(challenge_id: Optional[str] = None, status: Optional[str] = None):
    try:
        solutions = db.list_solutions(challenge_id=challenge_id, status=status)
        return {"success": True, "solutions": solutions, "total": len(solutions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/solutions")
@app.post("/api/challenges/{challenge_id}/solutions")
async def create_solution_endpoint(payload: SolutionCreate, challenge_id: Optional[str] = None):
    try:
        data = payload.model_dump()
        if challenge_id:
            data["challenge_id"] = challenge_id
        record = db.create_solution(data)
        return {"success": True, "solution": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit solution proposal: {str(e)}")


@app.get("/api/solutions/{solution_id}")
async def get_solution_endpoint(solution_id: str):
    sol = db.get_solution(solution_id)
    if not sol:
        raise HTTPException(status_code=404, detail="Solution not found.")
    return {"success": True, "solution": sol}


@app.patch("/api/solutions/{solution_id}")
async def update_solution_endpoint(solution_id: str, updates: Dict[str, Any]):
    new_status = updates.get("status")
    review_notes = updates.get("review_notes")
    if not new_status:
        raise HTTPException(status_code=400, detail="Status is required for solution update.")
    ok = db.update_solution_status(solution_id, new_status, review_notes)
    if not ok:
        raise HTTPException(status_code=404, detail="Solution not found.")
    updated = db.get_solution(solution_id)
    return {"success": True, "solution": updated}


@app.post("/api/solutions/{solution_id}/updates")
async def add_solution_update_endpoint(solution_id: str, payload: ImplementationUpdateCreate):
    try:
        data = payload.model_dump()
        data["solution_id"] = solution_id
        res = db.add_implementation_update(data)
        return {"success": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/solutions/{solution_id}/impact")
@app.post("/api/impact")
async def create_impact_endpoint(payload: ImpactRecordCreate, solution_id: Optional[str] = None):
    try:
        data = payload.model_dump()
        if solution_id:
            data["solution_id"] = solution_id
        res = db.create_impact_record(data)
        return {"success": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/impact")
async def list_impact_endpoint(challenge_id: Optional[str] = None):
    try:
        records = db.list_impact_records(challenge_id=challenge_id)
        return {"success": True, "impact_records": records, "total": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# MODULE 9: CLOSED-LOOP FEEDBACK, APPEALS & STAKEHOLDER REVIEWS
# ===========================================================================

@app.post("/api/feedback")
async def submit_feedback_endpoint(payload: FeedbackCreate):
    try:
        res = db.submit_feedback(payload.model_dump())
        return {"success": True, "feedback": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/problems/{problem_id}/appeal")
async def submit_appeal_endpoint(problem_id: str, payload: AppealCreate):
    try:
        res = db.submit_appeal(
            problem_id=problem_id,
            appeal_type=payload.appeal_type,
            reason=payload.reason,
            citizen_notes=payload.citizen_notes,
            user_id=payload.user_id
        )
        return {"success": True, "appeal": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit appeal: {str(e)}")


@app.get("/api/appeals")
async def list_appeals_endpoint(status: Optional[str] = None, problem_id: Optional[str] = None):
    try:
        appeals = db.list_appeals(status=status, problem_id=problem_id)
        return {"success": True, "appeals": appeals, "total": len(appeals)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/appeals/{appeal_id}")
async def resolve_appeal_endpoint(appeal_id: str, payload: AppealResolve):
    try:
        ok = db.resolve_appeal(
            appeal_id=appeal_id,
            new_status=payload.status,
            reviewer_notes=payload.reviewer_notes,
            resolution_action=payload.resolution_action
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Appeal not found.")
        return {"success": True, "appealId": appeal_id, "status": payload.status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stakeholders/review")
async def record_stakeholder_review_endpoint(payload: StakeholderReviewCreate):
    try:
        res = db.record_stakeholder_review(
            problem_id=payload.problem_id,
            stakeholder_profile_id=payload.stakeholder_profile_id,
            reviewer_name=payload.reviewer_name,
            action=payload.action,
            reason=payload.reason
        )
        return {"success": True, "review": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# MODULE 12, 14, 31: ADMIN REVIEW QUEUE, ANALYTICS & ROOT-CAUSE LEARNING
# ===========================================================================

@app.get("/api/admin/review-queue")
async def get_admin_review_queue_endpoint():
    try:
        queue = db.get_admin_review_queue()
        return {"success": True, "queue": queue}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/intelligence-stats")
async def get_analytics_stats_endpoint():
    try:
        stats = db.get_analytics_intelligence_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/intelligence/root-cause")
async def get_root_cause_endpoint():
    try:
        insights = db.get_root_cause_insights()
        return {"success": True, "root_cause_insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audit-trail")
async def get_audit_trail_endpoint(entity_type: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 50):
    try:
        events = db.list_audit_events(entity_type=entity_type, entity_id=entity_id, limit=limit)
        return {"success": True, "events": events, "total": len(events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# ===========================================================================
# UNIVERSITY, INDUSTRY & COLLABORATION API ENDPOINTS (STEP 6 EXTENSION)
# ===========================================================================

@app.get("/citizen")
@app.get("/university")
@app.get("/industry")
@app.get("/collaborations/{collab_id}")
async def serve_spa_portals(collab_id: Optional[str] = None):
    index_file = BASE_DIR / "index.html"
    return FileResponse(str(index_file))


@app.get("/api/universities/profile")
async def get_university_profile_endpoint(univ_id: str = "univ_coep"):
    prof = db.get_university_profile(univ_id)
    if not prof:
        prof = {
            "id": univ_id,
            "institution_name": "COEP Technological University, Pune",
            "department": "Civil & Environmental Engineering",
            "location": "Shivajinagar, Pune, MH",
            "contact_email": "research@coep.ac.in",
            "website": "https://coep.org.in",
            "areas_of_expertise": ["Civil Engineering", "Hydrology & Drainage", "Environmental Engineering", "Materials Science"],
            "faculty_leads": ["Dr. A. Verma (Professor, Water Resources)", "Dr. S. Kulkarni (Head, Geotech Lab)"],
            "labs_facilities": "Advanced Fluid Mechanics Lab, Soil Mechanics Lab, Environmental Quality Testing Facility",
            "student_capabilities": "Final-Year B.Tech & M.Tech thesis candidates ready for field instrumentation and rapid prototyping"
        }
    return {"success": True, "profile": prof}


@app.get("/api/universities")
async def list_universities_endpoint():
    univs = db.list_universities()
    return {"success": True, "universities": univs}


@app.get("/api/universities/solutions")
async def list_university_solutions_endpoint(university_id: Optional[str] = None):
    sols = db.list_university_solutions(university_id)
    return {"success": True, "solutions": sols, "total": len(sols)}


@app.get("/api/industry/profile")
async def get_industry_profile_endpoint(ind_id: str = "ind_ecoroads"):
    prof = db.get_industry_profile(ind_id)
    if not prof:
        prof = {
            "id": ind_id,
            "company_name": "EcoRoads InfraTech Innovations",
            "industry_type": "Startup",
            "sector": "Smart Infrastructure & CleanTech",
            "technical_capabilities": ["Cold-Mix Bio-Asphalt Paving", "Rapid Road Repair", "Field Pilot Deployment", "Pothole Telemetry"],
            "location": "Dadar West, Mumbai, MH",
            "team_size": "18 engineers & field crew",
            "website": "https://ecoroadsinfra.com",
            "collaboration_interests": "Seeking university partners for lab durability validation, moisture-resistance testing, and municipal pilot deployment"
        }
    return {"success": True, "profile": prof}


@app.get("/api/industry")
async def list_industries_endpoint():
    inds = db.list_industries()
    return {"success": True, "industries": inds}


@app.get("/api/industry/solutions")
async def discover_solutions_for_industry(industry_id: str = "ind_ecoroads"):
    sols = db.list_university_solutions()
    # Add match explanation based on capabilities
    ind = db.get_industry_profile(industry_id) or {}
    caps = ind.get("technical_capabilities", [])
    
    enriched = []
    for s in sols:
        s_dict = dict(s)
        disc = s_dict.get("technical_disciplines", [])
        
        # Calculate matching rationale
        reasons = []
        if any("civil" in str(d).lower() or "material" in str(d).lower() for d in disc):
            reasons.append("Matches startup's Cold-Mix & Pavement expertise")
        if any("iot" in str(d).lower() or "sensor" in str(d).lower() or "hydro" in str(d).lower() for d in disc):
            reasons.append("Synergy with telemetry & drainage pilot capabilities")
        if s_dict.get("prototype_required"):
            reasons.append("University explicitly seeks Industry Rapid Prototyping support")
        if not reasons:
            reasons.append("Cross-domain civic technology innovation synergy")

        s_dict["match_rationale"] = reasons
        s_dict["match_score"] = min(96, 75 + len(reasons) * 7)
        enriched.append(s_dict)

    return {"success": True, "solutions": enriched}


@app.post("/api/collaborations/requests")
async def create_collaboration_request_endpoint(payload: CollaborationRequestCreate):
    try:
        res = db.create_collaboration_request(
            solution_id=payload.solution_id,
            challenge_id=payload.challenge_id,
            industry_id=payload.industry_id,
            company_name=payload.company_name,
            why_interested=payload.why_interested,
            contribution=payload.contribution,
            technical_capability=payload.technical_capability,
            resources=payload.resources,
            commercialization_capability=payload.commercialization_capability,
            pilot_capability=payload.pilot_capability,
            message=payload.message
        )
        return {"success": True, "collaboration_request": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collaborations/requests")
async def list_collaboration_requests_endpoint(solution_id: Optional[str] = None, industry_id: Optional[str] = None):
    reqs = db.list_collaboration_requests(solution_id=solution_id, industry_id=industry_id)
    return {"success": True, "requests": reqs, "total": len(reqs)}


@app.post("/api/collaborations/requests/{req_id}/respond")
async def respond_collaboration_request_endpoint(req_id: str, payload: CollaborationResponse):
    try:
        res = db.respond_collaboration_request(
            request_id=req_id,
            action=payload.action,
            response_notes=payload.response_notes
        )
        return {"success": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collaborations")
async def list_collaborations_endpoint(university_id: Optional[str] = None, industry_id: Optional[str] = None):
    collabs = db.list_collaborations(university_id=university_id, industry_id=industry_id)
    return {"success": True, "collaborations": collabs, "total": len(collabs)}


@app.get("/api/collaborations/{collab_id}")
async def get_collaboration_endpoint(collab_id: str):
    collab = db.get_collaboration(collab_id)
    if not collab:
        raise HTTPException(status_code=404, detail="Collaboration workspace not found.")
    return {"success": True, "collaboration": collab}


@app.post("/api/collaborations/{collab_id}/tasks")
async def add_collaboration_task_endpoint(collab_id: str, payload: CollaborationTaskCreate):
    try:
        task = db.add_collaboration_task(
            collab_id=collab_id,
            title=payload.title,
            assigned_to=payload.assigned_to,
            due_date=payload.due_date
        )
        return {"success": True, "task": task}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/collaborations/tasks/{task_id}")
async def update_collaboration_task_endpoint(task_id: str, status: str = "DONE"):
    ok = db.toggle_collaboration_task(task_id, status)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True, "taskId": task_id, "status": status}


@app.post("/api/collaborations/{collab_id}/messages")
async def add_collaboration_message_endpoint(collab_id: str, payload: CollaborationMessageCreate):
    try:
        msg = db.add_collaboration_message(
            collab_id=collab_id,
            sender_role=payload.sender_role,
            sender_name=payload.sender_name,
            message=payload.message
        )
        return {"success": True, "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collaborations/{collab_id}/progress")
async def update_collaboration_progress_endpoint(collab_id: str, payload: CollaborationProgressUpdate):
    try:
        ok = db.update_collaboration_progress(
            collab_id=collab_id,
            progress_percentage=payload.progress_percentage,
            milestone=payload.milestone,
            status=payload.status
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Collaboration not found")
        return {"success": True, "collabId": collab_id, "progress": payload.progress_percentage}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/notifications")
async def get_notifications_endpoint(role: str = "CITIZEN", recipient_id: Optional[str] = None):
    notifs = db.get_notifications(role, recipient_id)
    return {"success": True, "notifications": notifs}


@app.post("/api/notifications/{notif_id}/read")
async def mark_notification_read_endpoint(notif_id: str):
    db.mark_notification_read(notif_id)
    return {"success": True, "notifId": notif_id}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
