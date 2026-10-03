import os
import json
import uuid
import sqlite3
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv

from backend.similarity import cosine_similarity

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "raka.db"
DATABASE_URL = os.getenv("DATABASE_URL")

# Determine mode: Supabase Postgres or SQLite
USE_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith("postgres"))

def get_connection():
    if USE_POSTGRES:
        import psycopg2
        import psycopg2.extras
        return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def _ensure_sqlite_column(cursor, table: str, column: str, col_type: str):
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    if column not in cols:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

def init_db():
    if USE_POSTGRES:
        # Postgres migrations are executed via migrations/001_initial_schema.sql
        pass
    else:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS problems (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            problem_text TEXT NOT NULL,
            image_url TEXT,
            has_location INTEGER DEFAULT 0,
            latitude REAL,
            longitude REAL,
            location_accuracy REAL,
            manual_location TEXT,
            status TEXT NOT NULL,
            is_demo INTEGER DEFAULT 0,
            similarity_status TEXT DEFAULT 'PENDING',
            preferred_language TEXT DEFAULT 'en',
            audio_url TEXT,
            voice_transcript TEXT,
            detected_language TEXT,
            city TEXT,
            district TEXT,
            state TEXT,
            locality TEXT,
            location_status TEXT DEFAULT 'CAPTURED',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS problem_ai_analysis (
            id TEXT PRIMARY KEY,
            problem_id TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            problem_summary TEXT NOT NULL,
            urgency TEXT NOT NULL,
            complexity TEXT NOT NULL,
            technical_discipline TEXT NOT NULL,
            suggested_departments TEXT NOT NULL,
            suggested_stakeholders TEXT,
            required_skills TEXT NOT NULL,
            possible_solution_types TEXT NOT NULL,
            estimated_project_duration TEXT,
            impact_areas TEXT NOT NULL,
            keywords TEXT NOT NULL,
            image_observations TEXT NOT NULL,
            reasoning_summary TEXT NOT NULL,
            confidence REAL NOT NULL,
            model TEXT,
            problem_title TEXT,
            sub_category TEXT,
            problem_understanding TEXT,
            visual_analysis TEXT,
            voice_analysis TEXT,
            impact_analysis TEXT,
            solution_context TEXT,
            routing_context TEXT,
            evidence TEXT,
            uncertainty TEXT,
            classification_confidence REAL DEFAULT 0.92,
            image_confidence REAL DEFAULT 0.88,
            language_confidence REAL DEFAULT 0.95,
            raw_analysis_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS problem_stakeholder_recommendations (
            id TEXT PRIMARY KEY,
            problem_id TEXT NOT NULL,
            stakeholder_type TEXT NOT NULL,
            name TEXT NOT NULL,
            jurisdiction_or_scope TEXT,
            relevance_level TEXT,
            relevance_score INTEGER,
            why_matched TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS problem_action_recommendations (
            id TEXT PRIMARY KEY,
            problem_id TEXT NOT NULL,
            title TEXT NOT NULL,
            action_type TEXT NOT NULL,
            description TEXT,
            estimated_timeline TEXT,
            why_recommended TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS stakeholder_profiles (
            id TEXT PRIMARY KEY,
            stakeholder_type TEXT NOT NULL,
            name TEXT NOT NULL,
            domain_category TEXT NOT NULL,
            jurisdiction_or_scope TEXT,
            default_relevance TEXT DEFAULT 'HIGH',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS matching_rules (
            id TEXT PRIMARY KEY,
            domain_category TEXT NOT NULL,
            sub_category_pattern TEXT DEFAULT '*',
            required_disciplines TEXT DEFAULT '[]',
            stakeholder_profile_id TEXT,
            relevance_score INTEGER DEFAULT 90,
            match_reason_template TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (stakeholder_profile_id) REFERENCES stakeholder_profiles(id) ON DELETE CASCADE
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS action_types (
            id TEXT PRIMARY KEY,
            domain_category TEXT NOT NULL,
            title TEXT NOT NULL,
            action_type TEXT NOT NULL,
            description TEXT,
            estimated_timeline TEXT,
            why_recommended_template TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Safely migrate existing tables if columns are missing
        for col, ctype in [
            ("preferred_language", "TEXT DEFAULT 'en'"),
            ("audio_url", "TEXT"),
            ("voice_transcript", "TEXT"),
            ("detected_language", "TEXT"),
            ("city", "TEXT"),
            ("district", "TEXT"),
            ("state", "TEXT"),
            ("locality", "TEXT"),
            ("location_status", "TEXT DEFAULT 'CAPTURED'")
        ]:
            _ensure_sqlite_column(c, "problems", col, ctype)

        for col, ctype in [
            ("problem_title", "TEXT"),
            ("sub_category", "TEXT"),
            ("problem_understanding", "TEXT"),
            ("visual_analysis", "TEXT"),
            ("voice_analysis", "TEXT"),
            ("impact_analysis", "TEXT"),
            ("solution_context", "TEXT"),
            ("routing_context", "TEXT"),
            ("evidence", "TEXT"),
            ("uncertainty", "TEXT"),
            ("classification_confidence", "REAL DEFAULT 0.92"),
            ("image_confidence", "REAL DEFAULT 0.88"),
            ("language_confidence", "REAL DEFAULT 0.95"),
            ("raw_analysis_json", "TEXT")
        ]:
            _ensure_sqlite_column(c, "problem_ai_analysis", col, ctype)

        c.execute("""
        CREATE TABLE IF NOT EXISTS problem_embeddings (
            id TEXT PRIMARY KEY,
            problem_id TEXT UNIQUE NOT NULL,
            embedding TEXT NOT NULL, -- JSON array of floats for SQLite
            embedding_model TEXT NOT NULL,
            embedding_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS problem_clusters (
            id TEXT PRIMARY KEY,
            cluster_title TEXT NOT NULL,
            category TEXT NOT NULL,
            primary_problem_id TEXT,
            problem_count INTEGER DEFAULT 1,
            status TEXT DEFAULT 'UNDER_VALIDATION',
            needs_review INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (primary_problem_id) REFERENCES problems(id) ON DELETE SET NULL
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS problem_cluster_members (
            id TEXT PRIMARY KEY,
            cluster_id TEXT NOT NULL,
            problem_id TEXT UNIQUE NOT NULL,
            membership_score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cluster_id) REFERENCES problem_clusters(id) ON DELETE CASCADE,
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS ai_analysis_logs (
            id TEXT PRIMARY KEY,
            problem_id TEXT,
            model TEXT,
            operation TEXT,
            input_hash TEXT,
            output_json TEXT,
            confidence REAL,
            latency_ms INTEGER,
            success INTEGER,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Step 5: Challenges Table
        c.execute("""
        CREATE TABLE IF NOT EXISTS challenges (
            id TEXT PRIMARY KEY,
            cluster_id TEXT,
            title TEXT NOT NULL,
            short_description TEXT NOT NULL,
            problem_statement TEXT NOT NULL,
            category TEXT NOT NULL,
            sub_category TEXT,
            problem_type TEXT DEFAULT 'Civic Infrastructure Challenge',
            affected_area TEXT,
            generalized_location TEXT,
            impact_summary TEXT NOT NULL,
            urgency TEXT NOT NULL DEFAULT 'MEDIUM',
            complexity TEXT NOT NULL DEFAULT 'MEDIUM',
            challenge_status TEXT NOT NULL DEFAULT 'DRAFT',
            readiness_score REAL DEFAULT 0.0,
            readiness_reasons TEXT DEFAULT '[]',
            missing_information TEXT DEFAULT '[]',
            warnings TEXT DEFAULT '[]',
            evidence_summary TEXT DEFAULT '[]',
            technical_disciplines TEXT DEFAULT '[]',
            required_skills TEXT DEFAULT '[]',
            stakeholder_types TEXT DEFAULT '[]',
            action_areas TEXT DEFAULT '[]',
            possible_solution_types TEXT DEFAULT '[]',
            expected_outcomes TEXT DEFAULT '[]',
            constraints TEXT DEFAULT '[]',
            success_indicators TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            published_at TIMESTAMP,
            resolved_at TIMESTAMP,
            FOREIGN KEY (cluster_id) REFERENCES problem_clusters(id) ON DELETE SET NULL
        );
        """)

        # Step 5: Challenge Evidence Table
        c.execute("""
        CREATE TABLE IF NOT EXISTS challenge_evidence (
            id TEXT PRIMARY KEY,
            challenge_id TEXT NOT NULL,
            problem_id TEXT,
            cluster_id TEXT,
            evidence_type TEXT NOT NULL,
            evidence_text TEXT NOT NULL,
            source_reference TEXT,
            confidence REAL DEFAULT 0.90,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (challenge_id) REFERENCES challenges(id) ON DELETE CASCADE
        );
        """)

        # Step 5: Chat Conversations Table
        c.execute("""
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            problem_id TEXT,
            cluster_id TEXT,
            challenge_id TEXT,
            preferred_language TEXT DEFAULT 'en',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Step 5: Chat Messages Table
        c.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            language TEXT DEFAULT 'en',
            sources TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE
        );
        """)
        

        # Closed-Loop Intelligence: Solutions Table (Module 7)
        c.execute("""
        CREATE TABLE IF NOT EXISTS solutions (
            id TEXT PRIMARY KEY,
            challenge_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            proposer_name TEXT NOT NULL,
            proposer_type TEXT NOT NULL DEFAULT 'INNOVATOR',
            technical_disciplines TEXT DEFAULT '[]',
            required_resources TEXT DEFAULT '[]',
            expected_outcome TEXT,
            constraints TEXT,
            status TEXT NOT NULL DEFAULT 'PROPOSED',
            review_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (challenge_id) REFERENCES challenges(id) ON DELETE CASCADE
        );
        """)

        # Solution Milestones (Module 8)
        c.execute("""
        CREATE TABLE IF NOT EXISTS solution_milestones (
            id TEXT PRIMARY KEY,
            solution_id TEXT NOT NULL,
            milestone_title TEXT NOT NULL,
            description TEXT,
            target_date TEXT,
            status TEXT NOT NULL DEFAULT 'PENDING',
            completion_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (solution_id) REFERENCES solutions(id) ON DELETE CASCADE
        );
        """)

        # Implementation Updates (Module 8)
        c.execute("""
        CREATE TABLE IF NOT EXISTS implementation_updates (
            id TEXT PRIMARY KEY,
            solution_id TEXT NOT NULL,
            milestone_id TEXT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            progress_percentage INTEGER DEFAULT 0,
            blockers TEXT,
            evidence_url TEXT,
            author_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (solution_id) REFERENCES solutions(id) ON DELETE CASCADE,
            FOREIGN KEY (milestone_id) REFERENCES solution_milestones(id) ON DELETE SET NULL
        );
        """)

        # Impact Records (Module 8 & 28)
        c.execute("""
        CREATE TABLE IF NOT EXISTS impact_records (
            id TEXT PRIMARY KEY,
            solution_id TEXT NOT NULL,
            challenge_id TEXT,
            indicator_name TEXT NOT NULL,
            baseline_value TEXT,
            expected_value TEXT,
            actual_value TEXT,
            unit TEXT,
            measurement_date TEXT,
            evidence_notes TEXT,
            verification_status TEXT DEFAULT 'VERIFIED',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (solution_id) REFERENCES solutions(id) ON DELETE CASCADE,
            FOREIGN KEY (challenge_id) REFERENCES challenges(id) ON DELETE SET NULL
        );
        """)

        # Feedback System (Module 9 & 29)
        c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            user_id TEXT,
            user_role TEXT DEFAULT 'CITIZEN',
            rating INTEGER,
            comment TEXT NOT NULL,
            issue_type TEXT,
            evidence_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Appeals & Correction System (Module 9 & 30)
        c.execute("""
        CREATE TABLE IF NOT EXISTS appeals (
            id TEXT PRIMARY KEY,
            problem_id TEXT NOT NULL,
            user_id TEXT,
            appeal_type TEXT NOT NULL,
            reason TEXT NOT NULL,
            citizen_notes TEXT,
            status TEXT NOT NULL DEFAULT 'SUBMITTED',
            reviewer_notes TEXT,
            resolution_action TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
        );
        """)

        # Stakeholder Reviews Workflow (Module 5, 19 & 20)
        c.execute("""
        CREATE TABLE IF NOT EXISTS stakeholder_reviews (
            id TEXT PRIMARY KEY,
            problem_id TEXT NOT NULL,
            stakeholder_profile_id TEXT,
            reviewer_name TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
        );
        """)

        # Audit Trail (Module 32)
        c.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT 'SYSTEM',
            event TEXT NOT NULL,
            previous_value TEXT,
            new_value TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Problem Validations Table (Module 3 & 9)
        c.execute("""
        CREATE TABLE IF NOT EXISTS problem_validations (
            id TEXT PRIMARY KEY,
            problem_id TEXT UNIQUE NOT NULL,
            validation_status TEXT NOT NULL DEFAULT 'VALID',
            reasons TEXT DEFAULT '[]',
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
        );
        """)

        _seed_master_tables(c)

        conn.commit()
        conn.close()

def _seed_master_tables(cursor):
    cursor.execute("SELECT COUNT(*) FROM stakeholder_profiles")
    row = cursor.fetchone()
    count = row[0] if row else 0
    if count == 0:
        import uuid
        from backend.matching_service import STAKEHOLDER_REGISTRY
        for cat, data in STAKEHOLDER_REGISTRY.items():
            for g in data.get("government", []):
                pid = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO stakeholder_profiles (id, stakeholder_type, name, domain_category, jurisdiction_or_scope, default_relevance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (pid, "Government / Public Authority", g["name"], cat, g["scope"], g.get("relevance", "HIGH")))
                rid = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO matching_rules (id, domain_category, sub_category_pattern, required_disciplines, stakeholder_profile_id, relevance_score, match_reason_template)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (rid, cat, "*", json.dumps([]), pid, 95 if g.get("relevance") == "HIGH" else 80, f"Jurisdiction aligns with {cat}"))
            for r in data.get("research", []):
                pid = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO stakeholder_profiles (id, stakeholder_type, name, domain_category, jurisdiction_or_scope, default_relevance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (pid, "Research & Innovation Partner", r["name"], cat, r["scope"], r.get("relevance", "HIGH")))
                rid = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO matching_rules (id, domain_category, sub_category_pattern, required_disciplines, stakeholder_profile_id, relevance_score, match_reason_template)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (rid, cat, "*", json.dumps([]), pid, 88 if r.get("relevance") == "HIGH" else 75, f"Technical diagnostics & innovation partner for {cat}"))
            for a in data.get("actions", []):
                aid = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO action_types (id, domain_category, title, action_type, description, estimated_timeline, why_recommended_template)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (aid, cat, a["title"], a["type"], a["desc"], a.get("timeline", "24 - 48 Hours"), f"Addresses civic issue in {cat}"))

init_db()

# DB Query Helper Functions
def save_problem(
    problem_id: str,
    problem_text: str,
    image_url: Optional[str],
    has_location: bool,
    latitude: Optional[float],
    longitude: Optional[float],
    location_accuracy: Optional[float],
    manual_location: Optional[str],
    is_demo: bool = False,
    user_id: Optional[str] = None,
    preferred_language: str = "en",
    audio_url: Optional[str] = None,
    voice_transcript: Optional[str] = None,
    detected_language: Optional[str] = None,
    city: Optional[str] = None,
    district: Optional[str] = None,
    state: Optional[str] = None,
    locality: Optional[str] = None,
    location_status: str = "CAPTURED"
):
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("""
            INSERT INTO problems (
                id, user_id, problem_text, image_url, has_location, latitude, longitude,
                location_accuracy, manual_location, status, is_demo, similarity_status,
                preferred_language, audio_url, voice_transcript, detected_language,
                city, district, state, locality, location_status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                problem_text = EXCLUDED.problem_text,
                image_url = EXCLUDED.image_url,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                location_accuracy = EXCLUDED.location_accuracy,
                manual_location = EXCLUDED.manual_location,
                preferred_language = EXCLUDED.preferred_language,
                audio_url = EXCLUDED.audio_url,
                voice_transcript = EXCLUDED.voice_transcript,
                detected_language = EXCLUDED.detected_language,
                city = EXCLUDED.city,
                district = EXCLUDED.district,
                state = EXCLUDED.state,
                locality = EXCLUDED.locality,
                location_status = EXCLUDED.location_status,
                updated_at = NOW();
        """, (
            problem_id, user_id, problem_text, image_url, 1 if has_location else 0,
            latitude, longitude, location_accuracy, manual_location, 'PENDING',
            is_demo, 'PENDING', preferred_language, audio_url, voice_transcript,
            detected_language, city, district, state, locality, location_status
        ))
    else:
        c.execute("""
            INSERT OR REPLACE INTO problems (
                id, user_id, problem_text, image_url, has_location, latitude, longitude,
                location_accuracy, manual_location, status, is_demo, similarity_status,
                preferred_language, audio_url, voice_transcript, detected_language,
                city, district, state, locality, location_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            problem_id, user_id, problem_text, image_url, 1 if has_location else 0,
            latitude, longitude, location_accuracy, manual_location, 'PENDING',
            1 if is_demo else 0, 'PENDING', preferred_language, audio_url,
            voice_transcript, detected_language, city, district, state, locality, location_status
        ))
    conn.commit()
    conn.close()

def save_analysis(problem_id: str, analysis: Any, model: str = "gemini-2.5-flash"):
    conn = get_connection()
    c = conn.cursor()
    import uuid
    analysis_id = str(uuid.uuid4())
    
    # Extract rich multimodal fields if present
    raw_json = analysis.model_dump_json() if hasattr(analysis, "model_dump_json") else (json.dumps(analysis) if isinstance(analysis, dict) else "")
    title = getattr(analysis, "problem_title", "")
    sub_cat = getattr(analysis, "sub_category", "")
    if hasattr(analysis, "classification") and hasattr(analysis.classification, "sub_category"):
        sub_cat = analysis.classification.sub_category
    
    prob_und = json.dumps(analysis.problem_understanding.model_dump()) if hasattr(analysis, "problem_understanding") and hasattr(analysis.problem_understanding, "model_dump") else ""
    vis_ana = json.dumps(analysis.visual_analysis.model_dump()) if hasattr(analysis, "visual_analysis") and hasattr(analysis.visual_analysis, "model_dump") else ""
    voi_ana = json.dumps(analysis.citizen_statement.model_dump()) if hasattr(analysis, "citizen_statement") and hasattr(analysis.citizen_statement, "model_dump") else ""
    imp_ana = json.dumps(analysis.impact.model_dump()) if hasattr(analysis, "impact") and hasattr(analysis.impact, "model_dump") else ""
    sol_ctx = json.dumps(analysis.solution_context.model_dump()) if hasattr(analysis, "solution_context") and hasattr(analysis.solution_context, "model_dump") else ""
    rout_ctx = json.dumps(analysis.routing_context.model_dump()) if hasattr(analysis, "routing_context") and hasattr(analysis.routing_context, "model_dump") else ""
    evid = json.dumps([e.model_dump() for e in analysis.evidence]) if hasattr(analysis, "evidence") and isinstance(analysis.evidence, list) and analysis.evidence and hasattr(analysis.evidence[0], "model_dump") else "[]"
    uncert = json.dumps(analysis.uncertainty.model_dump()) if hasattr(analysis, "uncertainty") and hasattr(analysis.uncertainty, "model_dump") else ""
    
    class_conf = getattr(analysis, "classification_confidence", 0.92)
    img_conf = getattr(analysis, "image_confidence", 0.88)
    lang_conf = getattr(analysis, "language_confidence", 0.95)

    if USE_POSTGRES:
        c.execute("""
            INSERT INTO problem_ai_analysis (
                id, problem_id, category, problem_summary, urgency, complexity,
                technical_discipline, suggested_departments, required_skills,
                possible_solution_types, estimated_project_duration, impact_areas,
                keywords, image_observations, reasoning_summary, confidence, model,
                problem_title, sub_category, problem_understanding, visual_analysis,
                voice_analysis, impact_analysis, solution_context, routing_context,
                evidence, uncertainty, classification_confidence, image_confidence,
                language_confidence, raw_analysis_json
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (problem_id) DO UPDATE SET
                category = EXCLUDED.category,
                problem_summary = EXCLUDED.problem_summary,
                urgency = EXCLUDED.urgency,
                complexity = EXCLUDED.complexity,
                technical_discipline = EXCLUDED.technical_discipline,
                suggested_departments = EXCLUDED.suggested_departments,
                required_skills = EXCLUDED.required_skills,
                possible_solution_types = EXCLUDED.possible_solution_types,
                estimated_project_duration = EXCLUDED.estimated_project_duration,
                impact_areas = EXCLUDED.impact_areas,
                keywords = EXCLUDED.keywords,
                image_observations = EXCLUDED.image_observations,
                reasoning_summary = EXCLUDED.reasoning_summary,
                confidence = EXCLUDED.confidence,
                model = EXCLUDED.model,
                problem_title = EXCLUDED.problem_title,
                sub_category = EXCLUDED.sub_category,
                problem_understanding = EXCLUDED.problem_understanding,
                visual_analysis = EXCLUDED.visual_analysis,
                voice_analysis = EXCLUDED.voice_analysis,
                impact_analysis = EXCLUDED.impact_analysis,
                solution_context = EXCLUDED.solution_context,
                routing_context = EXCLUDED.routing_context,
                evidence = EXCLUDED.evidence,
                uncertainty = EXCLUDED.uncertainty,
                classification_confidence = EXCLUDED.classification_confidence,
                image_confidence = EXCLUDED.image_confidence,
                language_confidence = EXCLUDED.language_confidence,
                raw_analysis_json = EXCLUDED.raw_analysis_json;
        """, (
            analysis_id, problem_id, analysis.category, analysis.problem_summary, analysis.urgency,
            analysis.complexity, json.dumps(analysis.technical_discipline), json.dumps(analysis.suggested_departments),
            json.dumps(analysis.required_skills), json.dumps(analysis.possible_solution_types),
            analysis.estimated_project_duration, json.dumps(analysis.impact_areas), json.dumps(analysis.keywords),
            json.dumps(analysis.image_observations), analysis.reasoning_summary, analysis.confidence, model,
            title, sub_cat, prob_und, vis_ana, voi_ana, imp_ana, sol_ctx, rout_ctx, evid, uncert,
            class_conf, img_conf, lang_conf, raw_json
        ))
        c.execute("UPDATE problems SET status = 'AI_ANALYZED', updated_at = NOW() WHERE id = %s", (problem_id,))
    else:
        c.execute("""
            INSERT OR REPLACE INTO problem_ai_analysis (
                id, problem_id, category, problem_summary, urgency, complexity,
                technical_discipline, suggested_departments, required_skills,
                possible_solution_types, estimated_project_duration, impact_areas,
                keywords, image_observations, reasoning_summary, confidence, model,
                problem_title, sub_category, problem_understanding, visual_analysis,
                voice_analysis, impact_analysis, solution_context, routing_context,
                evidence, uncertainty, classification_confidence, image_confidence,
                language_confidence, raw_analysis_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_id, problem_id, analysis.category, analysis.problem_summary, analysis.urgency,
            analysis.complexity, json.dumps(analysis.technical_discipline), json.dumps(analysis.suggested_departments),
            json.dumps(analysis.required_skills), json.dumps(analysis.possible_solution_types),
            analysis.estimated_project_duration, json.dumps(analysis.impact_areas), json.dumps(analysis.keywords),
            json.dumps(analysis.image_observations), analysis.reasoning_summary, analysis.confidence, model,
            title, sub_cat, prob_und, vis_ana, voi_ana, imp_ana, sol_ctx, rout_ctx, evid, uncert,
            class_conf, img_conf, lang_conf, raw_json
        ))
        c.execute("UPDATE problems SET status = 'AI_ANALYZED', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (problem_id,))
    conn.commit()
    conn.close()

def save_step4_recommendations(problem_id: str, step4_payload: Any):
    conn = get_connection()
    c = conn.cursor()
    import uuid
    if USE_POSTGRES:
        c.execute("DELETE FROM problem_stakeholder_recommendations WHERE problem_id = %s", (problem_id,))
        c.execute("DELETE FROM problem_action_recommendations WHERE problem_id = %s", (problem_id,))
        for s in step4_payload.stakeholders:
            sid = str(uuid.uuid4())
            c.execute("""
                INSERT INTO problem_stakeholder_recommendations (id, problem_id, stakeholder_type, name, jurisdiction_or_scope, relevance_level, relevance_score, why_matched)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (sid, problem_id, s.stakeholder_type, s.name, s.jurisdiction_or_scope, s.relevance_level, s.relevance_score, json.dumps(s.why_matched)))
        for a in step4_payload.action_areas:
            aid = str(uuid.uuid4())
            c.execute("""
                INSERT INTO problem_action_recommendations (id, problem_id, title, action_type, description, estimated_timeline, why_recommended)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (aid, problem_id, a.title, a.action_type, a.description, a.estimated_timeline, json.dumps(a.why_recommended)))
    else:
        c.execute("DELETE FROM problem_stakeholder_recommendations WHERE problem_id = ?", (problem_id,))
        c.execute("DELETE FROM problem_action_recommendations WHERE problem_id = ?", (problem_id,))
        for s in step4_payload.stakeholders:
            sid = str(uuid.uuid4())
            c.execute("""
                INSERT INTO problem_stakeholder_recommendations (id, problem_id, stakeholder_type, name, jurisdiction_or_scope, relevance_level, relevance_score, why_matched)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (sid, problem_id, s.stakeholder_type, s.name, s.jurisdiction_or_scope, s.relevance_level, s.relevance_score, json.dumps(s.why_matched)))
        for a in step4_payload.action_areas:
            aid = str(uuid.uuid4())
            c.execute("""
                INSERT INTO problem_action_recommendations (id, problem_id, title, action_type, description, estimated_timeline, why_recommended)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (aid, problem_id, a.title, a.action_type, a.description, a.estimated_timeline, json.dumps(a.why_recommended)))
    conn.commit()
    conn.close()

def get_step4_recommendations(problem_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    stakeholders = []
    actions = []
    if USE_POSTGRES:
        c.execute("SELECT stakeholder_type, name, jurisdiction_or_scope, relevance_level, relevance_score, why_matched FROM problem_stakeholder_recommendations WHERE problem_id = %s", (problem_id,))
        for r in c.fetchall():
            stakeholders.append({
                "stakeholder_type": r["stakeholder_type"],
                "name": r["name"],
                "jurisdiction_or_scope": r["jurisdiction_or_scope"],
                "relevance_level": r["relevance_level"],
                "relevance_score": r["relevance_score"],
                "why_matched": json.loads(r["why_matched"]) if r["why_matched"] else []
            })
        c.execute("SELECT title, action_type, description, estimated_timeline, why_recommended FROM problem_action_recommendations WHERE problem_id = %s", (problem_id,))
        for r in c.fetchall():
            actions.append({
                "title": r["title"],
                "action_type": r["action_type"],
                "description": r["description"],
                "estimated_timeline": r["estimated_timeline"],
                "why_recommended": json.loads(r["why_recommended"]) if r["why_recommended"] else []
            })
    else:
        c.execute("SELECT stakeholder_type, name, jurisdiction_or_scope, relevance_level, relevance_score, why_matched FROM problem_stakeholder_recommendations WHERE problem_id = ?", (problem_id,))
        for r in c.fetchall():
            stakeholders.append({
                "stakeholder_type": r[0],
                "name": r[1],
                "jurisdiction_or_scope": r[2],
                "relevance_level": r[3],
                "relevance_score": r[4],
                "why_matched": json.loads(r[5]) if r[5] else []
            })
        c.execute("SELECT title, action_type, description, estimated_timeline, why_recommended FROM problem_action_recommendations WHERE problem_id = ?", (problem_id,))
        for r in c.fetchall():
            actions.append({
                "title": r[0],
                "action_type": r[1],
                "description": r[2],
                "estimated_timeline": r[3],
                "why_recommended": json.loads(r[4]) if r[4] else []
            })
    conn.close()
    if not stakeholders and not actions:
        return None
    return {
        "stakeholders": stakeholders,
        "action_areas": actions,
        "disclaimer": "These are AI-assisted stakeholder recommendations, not official assignments."
    }


def save_embedding(problem_id: str, embedding: List[float], text_embedded: str, model_name: str = "gemini-embedding-2"):
    conn = get_connection()
    c = conn.cursor()
    import uuid
    emb_id = str(uuid.uuid4())
    if USE_POSTGRES:
        c.execute("""
            INSERT INTO problem_embeddings (id, problem_id, embedding, embedding_model, embedding_text, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (problem_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                embedding_text = EXCLUDED.embedding_text,
                updated_at = NOW();
        """, (emb_id, problem_id, embedding, model_name, text_embedded))
    else:
        c.execute("""
            INSERT OR REPLACE INTO problem_embeddings (id, problem_id, embedding, embedding_model, embedding_text, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (emb_id, problem_id, json.dumps(embedding), model_name, text_embedded))
    conn.commit()
    conn.close()

def update_similarity_status(problem_id: str, status: str):
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("UPDATE problems SET similarity_status = %s, updated_at = NOW() WHERE id = %s", (status, problem_id))
    else:
        c.execute("UPDATE problems SET similarity_status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, problem_id))
    conn.commit()
    conn.close()

def get_candidates(problem_id: str, embedding: List[float], is_demo: bool, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves top K candidate reports based on vector cosine similarity.
    Uses pgvector <=> operator in Postgres, or Python cosine similarity in SQLite.
    """
    conn = get_connection()
    c = conn.cursor()
    candidates = []

    if USE_POSTGRES:
        c.execute("""
            SELECT p.id, p.problem_text, p.latitude, p.longitude, p.created_at, p.status,
                   a.category, a.problem_summary, a.technical_discipline, a.impact_areas,
                   1 - (e.embedding <=> %s::vector) as cosine
            FROM problem_embeddings e
            JOIN problems p ON p.id = e.problem_id
            LEFT JOIN problem_ai_analysis a ON a.problem_id = p.id
            WHERE p.id <> %s AND p.is_demo = %s AND p.status = 'AI_ANALYZED'
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """, (embedding, problem_id, is_demo, embedding, limit))
        rows = c.fetchall()
        for r in rows:
            td = r["technical_discipline"]
            if isinstance(td, str):
                td = json.loads(td)
            candidates.append({
                "problem_id": r["id"],
                "cosine": float(r["cosine"]),
                "category": r["category"],
                "problem_summary": r["problem_summary"],
                "technical_discipline": td or [],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "created_at": r["created_at"],
                "status": r["status"]
            })
    else:
        # SQLite fallback: fetch stored embeddings for active analyzed reports with same is_demo
        c.execute("""
            SELECT p.id, p.problem_text, p.latitude, p.longitude, p.created_at, p.status,
                   a.category, a.problem_summary, a.technical_discipline, a.impact_areas,
                   e.embedding
            FROM problem_embeddings e
            JOIN problems p ON p.id = e.problem_id
            LEFT JOIN problem_ai_analysis a ON a.problem_id = p.id
            WHERE p.id <> ? AND p.is_demo = ? AND p.status = 'AI_ANALYZED'
        """, (problem_id, 1 if is_demo else 0))
        rows = c.fetchall()
        scored = []
        for r in rows:
            try:
                cand_vec = json.loads(r["embedding"])
                cos = cosine_similarity(embedding, cand_vec)
                td = json.loads(r["technical_discipline"]) if r["technical_discipline"] else []
                scored.append({
                    "problem_id": r["id"],
                    "cosine": cos,
                    "category": r["category"],
                    "problem_summary": r["problem_summary"],
                    "technical_discipline": td,
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "created_at": r["created_at"],
                    "status": r["status"]
                })
            except Exception:
                continue
        # Sort descending by cosine similarity
        scored.sort(key=lambda x: x["cosine"], reverse=True)
        candidates = scored[:limit]

    conn.close()
    return candidates

def get_cluster_state() -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Returns (existing_memberships: problem_id -> cluster_id, existing_clusters: cluster_id -> meta)"""
    conn = get_connection()
    c = conn.cursor()
    memberships = {}
    clusters = {}

    if USE_POSTGRES:
        c.execute("SELECT problem_id, cluster_id FROM problem_cluster_members;")
        for r in c.fetchall():
            memberships[str(r["problem_id"])] = str(r["cluster_id"])

        c.execute("SELECT id, cluster_title, category, primary_problem_id, problem_count, status, needs_review, created_at FROM problem_clusters;")
        for r in c.fetchall():
            clusters[str(r["id"])] = {
                "id": str(r["id"]),
                "cluster_title": r["cluster_title"],
                "category": r["category"],
                "primary_problem_id": str(r["primary_problem_id"]) if r["primary_problem_id"] else None,
                "problem_count": r["problem_count"],
                "status": r["status"],
                "needs_review": bool(r["needs_review"]),
                "created_at": str(r["created_at"]),
            }
    else:
        c.execute("SELECT problem_id, cluster_id FROM problem_cluster_members;")
        for r in c.fetchall():
            memberships[str(r[0])] = str(r[1])

        c.execute("SELECT id, cluster_title, category, primary_problem_id, problem_count, status, needs_review, created_at FROM problem_clusters;")
        for r in c.fetchall():
            clusters[str(r[0])] = {
                "id": str(r[0]),
                "cluster_title": r[1],
                "category": r[2],
                "primary_problem_id": str(r[3]) if r[3] else None,
                "problem_count": r[4],
                "status": r[5],
                "needs_review": bool(r[6]),
                "created_at": str(r[7]),
            }

    conn.close()
    return memberships, clusters

def apply_cluster_plan(plan: Dict[str, Any], new_problem_id: str, problem_category: str, problem_summary: str):
    """Executes cluster updates deterministically in DB."""
    action = plan.get("action")
    if action == "NONE":
        return

    conn = get_connection()
    c = conn.cursor()
    import uuid

    if action == "CREATE_NEW":
        cid = plan["target_cluster_id"]
        # Generate title: <Category>: <first sentence of AI summary>
        first_sentence = problem_summary.split(".")[0].strip()[:80]
        title = f"{problem_category}: {first_sentence}"
        
        if USE_POSTGRES:
            c.execute("""
                INSERT INTO problem_clusters (id, cluster_title, category, primary_problem_id, problem_count, status, needs_review, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, 'UNDER_VALIDATION', false, NOW(), NOW());
            """, (cid, title, problem_category, new_problem_id, len(plan["member_ids_to_add"])))
        else:
            c.execute("""
                INSERT INTO problem_clusters (id, cluster_title, category, primary_problem_id, problem_count, status, needs_review)
                VALUES (?, ?, ?, ?, ?, 'UNDER_VALIDATION', 0);
            """, (cid, title, problem_category, new_problem_id, len(plan["member_ids_to_add"])))

        score_lookup = {l["problem_id"]: l["combined_score"] for l in plan.get("links", [])}
        for mid in plan["member_ids_to_add"]:
            mem_id = str(uuid.uuid4())
            mscore = score_lookup.get(mid, 1.0)
            if USE_POSTGRES:
                c.execute("""
                    INSERT INTO problem_cluster_members (id, cluster_id, problem_id, membership_score, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (problem_id) DO UPDATE SET cluster_id = EXCLUDED.cluster_id;
                """, (mem_id, cid, mid, mscore))
            else:
                c.execute("""
                    INSERT OR REPLACE INTO problem_cluster_members (id, cluster_id, problem_id, membership_score)
                    VALUES (?, ?, ?, ?);
                """, (mem_id, cid, mid, mscore))

    elif action == "ADD_TO_EXISTING":
        cid = plan["target_cluster_id"]
        score_lookup = {l["problem_id"]: l["combined_score"] for l in plan.get("links", [])}
        for mid in plan["member_ids_to_add"]:
            mem_id = str(uuid.uuid4())
            mscore = score_lookup.get(mid, 0.85)
            if USE_POSTGRES:
                c.execute("""
                    INSERT INTO problem_cluster_members (id, cluster_id, problem_id, membership_score, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (problem_id) DO NOTHING;
                """, (mem_id, cid, mid, mscore))
            else:
                c.execute("""
                    INSERT OR IGNORE INTO problem_cluster_members (id, cluster_id, problem_id, membership_score)
                    VALUES (?, ?, ?, ?);
                """, (mem_id, cid, mid, mscore))

    elif action == "MERGE_CLUSTERS":
        target_cid = plan["target_cluster_id"]
        absorbed = plan["absorbed_cluster_ids"]
        score_lookup = {l["problem_id"]: l["combined_score"] for l in plan.get("links", [])}
        
        # Add new member
        mem_id = str(uuid.uuid4())
        mscore = max([l["combined_score"] for l in plan.get("links", [])] or [0.85])
        if USE_POSTGRES:
            c.execute("""
                INSERT INTO problem_cluster_members (id, cluster_id, problem_id, membership_score, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (problem_id) DO UPDATE SET cluster_id = %s;
            """, (mem_id, target_cid, new_problem_id, mscore, target_cid))
            # Move members of absorbed clusters
            for acid in absorbed:
                c.execute("UPDATE problem_cluster_members SET cluster_id = %s WHERE cluster_id = %s", (target_cid, acid))
                c.execute("DELETE FROM problem_clusters WHERE id = %s", (acid,))
        else:
            c.execute("""
                INSERT OR REPLACE INTO problem_cluster_members (id, cluster_id, problem_id, membership_score)
                VALUES (?, ?, ?, ?);
            """, (mem_id, target_cid, new_problem_id, mscore))
            for acid in absorbed:
                c.execute("UPDATE problem_cluster_members SET cluster_id = ? WHERE cluster_id = ?", (target_cid, acid))
                c.execute("DELETE FROM problem_clusters WHERE id = ?", (acid,))

    conn.commit()
    conn.close()

    # Recalculate cluster count and review status
    target_cid = plan.get("target_cluster_id")
    if target_cid:
        refresh_cluster(target_cid)

def refresh_cluster(cluster_id: str):
    """Updates problem_count and checks needs_review status for a cluster."""
    conn = get_connection()
    c = conn.cursor()
    
    if USE_POSTGRES:
        c.execute("SELECT COUNT(*) as cnt FROM problem_cluster_members WHERE cluster_id = %s", (cluster_id,))
        cnt = c.fetchone()["cnt"]
        needs_review = cnt > 25
        c.execute("""
            UPDATE problem_clusters
            SET problem_count = %s, needs_review = %s, updated_at = NOW()
            WHERE id = %s;
        """, (cnt, needs_review, cluster_id))
    else:
        c.execute("SELECT COUNT(*) FROM problem_cluster_members WHERE cluster_id = ?", (cluster_id,))
        cnt = c.fetchone()[0]
        needs_review = 1 if cnt > 25 else 0
        c.execute("""
            UPDATE problem_clusters
            SET problem_count = ?, needs_review = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?;
        """, (cnt, needs_review, cluster_id))
        
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Step 5: Challenge Database Operations
# ---------------------------------------------------------------------------

def save_challenge(ch: Dict[str, Any]) -> str:
    """Inserts or updates a structured challenge."""
    conn = get_connection()
    c = conn.cursor()
    cid = ch.get("id") or str(uuid.uuid4())
    
    def _j(val):
        return json.dumps(val) if isinstance(val, (list, dict)) else (val or "[]")

    if USE_POSTGRES:
        c.execute("""
            INSERT INTO challenges (
                id, cluster_id, title, short_description, problem_statement, category, sub_category,
                problem_type, affected_area, generalized_location, impact_summary, urgency, complexity,
                challenge_status, readiness_score, readiness_reasons, missing_information, warnings,
                evidence_summary, technical_disciplines, required_skills, stakeholder_types, action_areas,
                possible_solution_types, expected_outcomes, constraints, success_indicators,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                short_description = EXCLUDED.short_description,
                problem_statement = EXCLUDED.problem_statement,
                category = EXCLUDED.category,
                sub_category = EXCLUDED.sub_category,
                impact_summary = EXCLUDED.impact_summary,
                urgency = EXCLUDED.urgency,
                complexity = EXCLUDED.complexity,
                challenge_status = EXCLUDED.challenge_status,
                readiness_score = EXCLUDED.readiness_score,
                readiness_reasons = EXCLUDED.readiness_reasons,
                missing_information = EXCLUDED.missing_information,
                warnings = EXCLUDED.warnings,
                evidence_summary = EXCLUDED.evidence_summary,
                technical_disciplines = EXCLUDED.technical_disciplines,
                required_skills = EXCLUDED.required_skills,
                stakeholder_types = EXCLUDED.stakeholder_types,
                action_areas = EXCLUDED.action_areas,
                possible_solution_types = EXCLUDED.possible_solution_types,
                expected_outcomes = EXCLUDED.expected_outcomes,
                constraints = EXCLUDED.constraints,
                success_indicators = EXCLUDED.success_indicators,
                updated_at = NOW();
        """, (
            cid, ch.get("cluster_id"), ch.get("title"), ch.get("short_description"), ch.get("problem_statement"),
            ch.get("category"), ch.get("sub_category"), ch.get("problem_type", "Civic Infrastructure Challenge"),
            ch.get("affected_area"), ch.get("generalized_location"), ch.get("impact_summary"),
            ch.get("urgency", "MEDIUM"), ch.get("complexity", "MEDIUM"), ch.get("challenge_status", "DRAFT"),
            float(ch.get("readiness_score", 0.0)), _j(ch.get("readiness_reasons")), _j(ch.get("missing_information")),
            _j(ch.get("warnings")), _j(ch.get("evidence_summary")), _j(ch.get("technical_disciplines")),
            _j(ch.get("required_skills")), _j(ch.get("stakeholder_types")), _j(ch.get("action_areas")),
            _j(ch.get("possible_solution_types")), _j(ch.get("expected_outcomes")), _j(ch.get("constraints")),
            _j(ch.get("success_indicators"))
        ))
    else:
        c.execute("""
            INSERT OR REPLACE INTO challenges (
                id, cluster_id, title, short_description, problem_statement, category, sub_category,
                problem_type, affected_area, generalized_location, impact_summary, urgency, complexity,
                challenge_status, readiness_score, readiness_reasons, missing_information, warnings,
                evidence_summary, technical_disciplines, required_skills, stakeholder_types, action_areas,
                possible_solution_types, expected_outcomes, constraints, success_indicators,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
        """, (
            cid, ch.get("cluster_id"), ch.get("title"), ch.get("short_description"), ch.get("problem_statement"),
            ch.get("category"), ch.get("sub_category"), ch.get("problem_type", "Civic Infrastructure Challenge"),
            ch.get("affected_area"), ch.get("generalized_location"), ch.get("impact_summary"),
            ch.get("urgency", "MEDIUM"), ch.get("complexity", "MEDIUM"), ch.get("challenge_status", "DRAFT"),
            float(ch.get("readiness_score", 0.0)), _j(ch.get("readiness_reasons")), _j(ch.get("missing_information")),
            _j(ch.get("warnings")), _j(ch.get("evidence_summary")), _j(ch.get("technical_disciplines")),
            _j(ch.get("required_skills")), _j(ch.get("stakeholder_types")), _j(ch.get("action_areas")),
            _j(ch.get("possible_solution_types")), _j(ch.get("expected_outcomes")), _j(ch.get("constraints")),
            _j(ch.get("success_indicators"))
        ))

    conn.commit()
    conn.close()
    return cid


def _parse_challenge_row(r) -> Dict[str, Any]:
    """Helper to deserialize challenge columns."""
    if not r:
        return {}
    def _pj(val):
        if isinstance(val, (list, dict)):
            return val
        if val and isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return []
        return []

    def _safe_get(row, key, default=None):
        """Safe accessor that works with both dict and sqlite3.Row."""
        try:
            val = row[key]
            return val if val is not None else default
        except (KeyError, IndexError):
            return default

    return {
        "id": str(r["id"]),
        "cluster_id": str(r["cluster_id"]) if r["cluster_id"] else None,
        "title": r["title"],
        "short_description": r["short_description"],
        "problem_statement": r["problem_statement"],
        "category": r["category"],
        "sub_category": r["sub_category"],
        "problem_type": r["problem_type"],
        "affected_area": r["affected_area"],
        "generalized_location": r["generalized_location"],
        "impact_summary": r["impact_summary"],
        "urgency": r["urgency"],
        "complexity": r["complexity"],
        "challenge_status": r["challenge_status"],
        "readiness_score": float(r["readiness_score"] or 0.0),
        "readiness_reasons": _pj(r["readiness_reasons"]),
        "missing_information": _pj(r["missing_information"]),
        "warnings": _pj(r["warnings"]),
        "evidence_summary": _pj(r["evidence_summary"]),
        "technical_disciplines": _pj(r["technical_disciplines"]),
        "required_skills": _pj(r["required_skills"]),
        "stakeholder_types": _pj(r["stakeholder_types"]),
        "action_areas": _pj(r["action_areas"]),
        "possible_solution_types": _pj(r["possible_solution_types"]),
        "expected_outcomes": _pj(r["expected_outcomes"]),
        "constraints": _pj(r["constraints"]),
        "success_indicators": _pj(r["success_indicators"]),
        "created_at": str(r["created_at"]),
        "updated_at": str(r["updated_at"]),
        "published_at": str(_safe_get(r, "published_at")) if _safe_get(r, "published_at") else None,
        "resolved_at": str(_safe_get(r, "resolved_at")) if _safe_get(r, "resolved_at") else None,
    }


def get_challenge(challenge_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a challenge by ID."""
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("SELECT * FROM challenges WHERE id = %s", (challenge_id,))
    else:
        c.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return _parse_challenge_row(row)


def get_challenge_by_cluster(cluster_id: str) -> Optional[Dict[str, Any]]:
    """Fetches challenge associated with a cluster ID."""
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("SELECT * FROM challenges WHERE cluster_id = %s ORDER BY created_at DESC LIMIT 1", (cluster_id,))
    else:
        c.execute("SELECT * FROM challenges WHERE cluster_id = ? ORDER BY created_at DESC LIMIT 1", (cluster_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return _parse_challenge_row(row)


def list_challenges(
    search: str = "",
    category: str = "",
    sub_category: str = "",
    status: str = "",
    urgency: str = "",
    discipline: str = "",
    readiness: str = ""
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Lists challenges matching filters and returns live database statistics.
    Never uses hardcoded metrics.
    """
    conn = get_connection()
    c = conn.cursor()

    # 1. Compute exact live statistics across database
    c.execute("SELECT challenge_status, COUNT(*) as cnt FROM challenges GROUP BY challenge_status")
    stat_rows = c.fetchall()
    stats = {
        "total": 0,
        "draft": 0,
        "validation_required": 0,
        "validated": 0,
        "ready_for_solutions": 0,
        "in_progress": 0,
        "resolved": 0
    }
    for sr in stat_rows:
        st = (sr["challenge_status"] if USE_POSTGRES else sr[0]) or ""
        cnt = sr["cnt"] if USE_POSTGRES else sr[1]
        stats["total"] += cnt
        st_norm = st.lower().replace("-", "_")
        if st_norm in stats:
            stats[st_norm] += cnt
        elif st_norm == "ready_for_solutions":
            stats["ready_for_solutions"] += cnt

    # 2. Build filtered query
    query = "SELECT * FROM challenges WHERE 1=1"
    params = []

    if search:
        s_term = f"%{search.strip().lower()}%"
        if USE_POSTGRES:
            query += " AND (LOWER(title) LIKE %s OR LOWER(problem_statement) LIKE %s OR LOWER(category) LIKE %s)"
        else:
            query += " AND (LOWER(title) LIKE ? OR LOWER(problem_statement) LIKE ? OR LOWER(category) LIKE ?)"
        params.extend([s_term, s_term, s_term])

    if category:
        query += " AND category = %s" if USE_POSTGRES else " AND category = ?"
        params.append(category)

    if sub_category:
        query += " AND sub_category = %s" if USE_POSTGRES else " AND sub_category = ?"
        params.append(sub_category)

    if status:
        query += " AND challenge_status = %s" if USE_POSTGRES else " AND challenge_status = ?"
        params.append(status.upper())

    if urgency:
        query += " AND urgency = %s" if USE_POSTGRES else " AND urgency = ?"
        params.append(urgency.upper())

    if discipline:
        d_term = f"%{discipline.strip().lower()}%"
        query += " AND LOWER(technical_disciplines) LIKE %s" if USE_POSTGRES else " AND LOWER(technical_disciplines) LIKE ?"
        params.append(d_term)

    if readiness:
        if readiness.upper() == "READY":
            query += " AND readiness_score >= 0.65"
        elif readiness.upper() == "NEEDS_VALIDATION":
            query += " AND readiness_score < 0.65"

    query += " ORDER BY created_at DESC"

    c.execute(query, tuple(params))
    rows = c.fetchall()
    conn.close()

    challenges = [_parse_challenge_row(r) for r in rows]
    return challenges, stats


def update_challenge_status(challenge_id: str, new_status: str) -> bool:
    """Updates the lifecycle status of a challenge."""
    conn = get_connection()
    c = conn.cursor()
    now_func = "NOW()" if USE_POSTGRES else "CURRENT_TIMESTAMP"
    pub_clause = f", published_at = {now_func}" if new_status.upper() == "PUBLISHED" else ""
    res_clause = f", resolved_at = {now_func}" if new_status.upper() == "RESOLVED" else ""

    if USE_POSTGRES:
        c.execute(f"""
            UPDATE challenges
            SET challenge_status = %s, updated_at = NOW() {pub_clause} {res_clause}
            WHERE id = %s
        """, (new_status.upper(), challenge_id))
    else:
        c.execute(f"""
            UPDATE challenges
            SET challenge_status = ?, updated_at = CURRENT_TIMESTAMP {pub_clause} {res_clause}
            WHERE id = ?
        """, (new_status.upper(), challenge_id))
    
    rc = c.rowcount
    conn.commit()
    conn.close()
    return rc > 0


def update_challenge(challenge_id: str, updates: Dict[str, Any]) -> bool:
    """Updates editable fields of a challenge."""
    conn = get_connection()
    c = conn.cursor()
    allowed_fields = [
        "title", "short_description", "problem_statement", "category", "sub_category",
        "urgency", "complexity", "impact_summary"
    ]
    sets = []
    params = []
    for k, v in updates.items():
        if k in allowed_fields and v is not None:
            sets.append(f"{k} = %s" if USE_POSTGRES else f"{k} = ?")
            params.append(v)
    if not sets:
        conn.close()
        return False

    now_clause = "NOW()" if USE_POSTGRES else "CURRENT_TIMESTAMP"
    query = f"UPDATE challenges SET {', '.join(sets)}, updated_at = {now_clause} WHERE id = " + ("%s" if USE_POSTGRES else "?")
    params.append(challenge_id)

    c.execute(query, tuple(params))
    rc = c.rowcount
    conn.commit()
    conn.close()
    return rc > 0


# ---------------------------------------------------------------------------
# Step 5: Challenge Evidence Operations
# ---------------------------------------------------------------------------

def save_challenge_evidence(challenge_id: str, evidence_items: List[Dict[str, Any]]):
    """Saves evidence items attached to a challenge."""
    conn = get_connection()
    c = conn.cursor()
    for ev in evidence_items:
        eid = ev.get("id") or str(uuid.uuid4())
        etype = ev.get("evidence_type", "REPORTED_BY_CITIZEN")
        etext = ev.get("evidence_text", "")
        sref = ev.get("source_reference", "")
        conf = float(ev.get("confidence", 0.90))
        pid = ev.get("problem_id")
        cid = ev.get("cluster_id")

        if USE_POSTGRES:
            c.execute("""
                INSERT INTO challenge_evidence (id, challenge_id, problem_id, cluster_id, evidence_type, evidence_text, source_reference, confidence, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (eid, challenge_id, pid, cid, etype, etext, sref, conf))
        else:
            c.execute("""
                INSERT INTO challenge_evidence (id, challenge_id, problem_id, cluster_id, evidence_type, evidence_text, source_reference, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (eid, challenge_id, pid, cid, etype, etext, sref, conf))

    conn.commit()
    conn.close()


def get_challenge_evidence(challenge_id: str) -> List[Dict[str, Any]]:
    """Fetches all evidence items for a given challenge."""
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("SELECT * FROM challenge_evidence WHERE challenge_id = %s ORDER BY created_at ASC", (challenge_id,))
    else:
        c.execute("SELECT * FROM challenge_evidence WHERE challenge_id = ? ORDER BY created_at ASC", (challenge_id,))
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id": str(r["id"]),
            "challenge_id": str(r["challenge_id"]),
            "problem_id": str(r["problem_id"]) if r["problem_id"] else None,
            "cluster_id": str(r["cluster_id"]) if r["cluster_id"] else None,
            "evidence_type": r["evidence_type"],
            "evidence_text": r["evidence_text"],
            "source_reference": r["source_reference"],
            "confidence": float(r["confidence"] or 0.90)
        })
    return result


# ---------------------------------------------------------------------------
# Step 5: Chatbot Memory & Conversation Operations
# ---------------------------------------------------------------------------

def create_or_get_chat_conversation(
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    problem_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    challenge_id: Optional[str] = None,
    preferred_language: str = "en"
) -> str:
    """Creates a new conversation or verifies an existing conversation ID."""
    conn = get_connection()
    c = conn.cursor()
    cid = conversation_id or str(uuid.uuid4())

    if USE_POSTGRES:
        c.execute("""
            INSERT INTO chat_conversations (id, user_id, problem_id, cluster_id, challenge_id, preferred_language, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                problem_id = COALESCE(EXCLUDED.problem_id, chat_conversations.problem_id),
                cluster_id = COALESCE(EXCLUDED.cluster_id, chat_conversations.cluster_id),
                challenge_id = COALESCE(EXCLUDED.challenge_id, chat_conversations.challenge_id),
                preferred_language = EXCLUDED.preferred_language,
                updated_at = NOW();
        """, (cid, user_id, problem_id, cluster_id, challenge_id, preferred_language))
    else:
        # Check if exists
        c.execute("SELECT id FROM chat_conversations WHERE id = ?", (cid,))
        exists = c.fetchone()
        if exists:
            c.execute("""
                UPDATE chat_conversations
                SET problem_id = COALESCE(?, problem_id),
                    cluster_id = COALESCE(?, cluster_id),
                    challenge_id = COALESCE(?, challenge_id),
                    preferred_language = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?;
            """, (problem_id, cluster_id, challenge_id, preferred_language, cid))
        else:
            c.execute("""
                INSERT INTO chat_conversations (id, user_id, problem_id, cluster_id, challenge_id, preferred_language, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
            """, (cid, user_id, problem_id, cluster_id, challenge_id, preferred_language))

    conn.commit()
    conn.close()
    return cid


def save_chat_message(
    conversation_id: str,
    role: str,
    message: str,
    language: str = "en",
    sources: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Appends a message to the conversation."""
    conn = get_connection()
    c = conn.cursor()
    mid = str(uuid.uuid4())
    s_json = json.dumps(sources or [])

    if USE_POSTGRES:
        c.execute("""
            INSERT INTO chat_messages (id, conversation_id, role, message, language, sources, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW());
            UPDATE chat_conversations SET updated_at = NOW() WHERE id = %s;
        """, (mid, conversation_id, role.upper(), message, language, s_json, conversation_id))
    else:
        c.execute("""
            INSERT INTO chat_messages (id, conversation_id, role, message, language, sources, created_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """, (mid, conversation_id, role.upper(), message, language, s_json))
        c.execute("UPDATE chat_conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (conversation_id,))

    conn.commit()
    conn.close()
    return mid


def get_chat_messages(conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves recent message history for context continuity."""
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("""
            SELECT * FROM chat_messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
            LIMIT %s;
        """, (conversation_id, limit))
    else:
        c.execute("""
            SELECT * FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ?;
        """, (conversation_id, limit))
    rows = c.fetchall()
    conn.close()
    messages = []
    for r in rows:
        src = r["sources"]
        if isinstance(src, str):
            try:
                src = json.loads(src)
            except Exception:
                src = []
        messages.append({
            "id": str(r["id"]),
            "role": r["role"],
            "message": r["message"],
            "language": r["language"],
            "sources": src or [],
            "created_at": str(r["created_at"])
        })
    return messages


def get_chat_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves conversation metadata."""
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("SELECT * FROM chat_conversations WHERE id = %s", (conversation_id,))
    else:
        c.execute("SELECT * FROM chat_conversations WHERE id = ?", (conversation_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": str(row["id"]),
        "user_id": str(row["user_id"]) if row["user_id"] else None,
        "problem_id": str(row["problem_id"]) if row["problem_id"] else None,
        "cluster_id": str(row["cluster_id"]) if row["cluster_id"] else None,
        "challenge_id": str(row["challenge_id"]) if row["challenge_id"] else None,
        "preferred_language": row["preferred_language"],
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"])
    }



# ===========================================================================
# CLOSED-LOOP INTELLIGENCE: SOLUTIONS, IMPLEMENTATION & IMPACT DATA ACCESS
# ===========================================================================

def create_solution(data: Dict[str, Any]) -> Dict[str, Any]:
    sol_id = data.get("id") or str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    
    disciplines = data.get("technical_disciplines", [])
    if isinstance(disciplines, list):
        disciplines = json.dumps(disciplines)
    resources = data.get("required_resources", [])
    if isinstance(resources, list):
        resources = json.dumps(resources)

    if USE_POSTGRES:
        c.execute("""
            INSERT INTO solutions (id, challenge_id, title, description, proposer_name, proposer_type,
                                   technical_disciplines, required_resources, expected_outcome, constraints, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *;
        """, (
            sol_id, data["challenge_id"], data["title"], data["description"],
            data.get("proposer_name", "Anonymous Innovator"), data.get("proposer_type", "INNOVATOR"),
            disciplines, resources, data.get("expected_outcome", ""), data.get("constraints", ""),
            data.get("status", "PROPOSED")
        ))
    else:
        c.execute("""
            INSERT INTO solutions (id, challenge_id, title, description, proposer_name, proposer_type,
                                   technical_disciplines, required_resources, expected_outcome, constraints, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            sol_id, data["challenge_id"], data["title"], data["description"],
            data.get("proposer_name", "Anonymous Innovator"), data.get("proposer_type", "INNOVATOR"),
            disciplines, resources, data.get("expected_outcome", ""), data.get("constraints", ""),
            data.get("status", "PROPOSED")
        ))
    conn.commit()
    conn.close()

    log_audit_event("SOLUTION", sol_id, "SOLUTION_PROPOSED", actor=data.get("proposer_name", "INNOVATOR"),
                    new_value=data["title"], source="INNOVATION_WORKFLOW")
    return get_solution(sol_id)


def get_solution(solution_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("SELECT * FROM solutions WHERE id = %s", (solution_id,))
    else:
        c.execute("SELECT * FROM solutions WHERE id = ?", (solution_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None

    # Fetch milestones
    if USE_POSTGRES:
        c.execute("SELECT * FROM solution_milestones WHERE solution_id = %s ORDER BY created_at ASC", (solution_id,))
    else:
        c.execute("SELECT * FROM solution_milestones WHERE solution_id = ? ORDER BY created_at ASC", (solution_id,))
    milestones = [dict(m) for m in c.fetchall()]

    # Fetch updates
    if USE_POSTGRES:
        c.execute("SELECT * FROM implementation_updates WHERE solution_id = %s ORDER BY created_at DESC", (solution_id,))
    else:
        c.execute("SELECT * FROM implementation_updates WHERE solution_id = ? ORDER BY created_at DESC", (solution_id,))
    updates = [dict(u) for u in c.fetchall()]

    # Fetch impact
    if USE_POSTGRES:
        c.execute("SELECT * FROM impact_records WHERE solution_id = %s ORDER BY created_at DESC", (solution_id,))
    else:
        c.execute("SELECT * FROM impact_records WHERE solution_id = ? ORDER BY created_at DESC", (solution_id,))
    impacts = [dict(i) for i in c.fetchall()]
    conn.close()

    sol = dict(row)
    for col in ["technical_disciplines", "required_resources"]:
        if isinstance(sol.get(col), str):
            try:
                sol[col] = json.loads(sol[col])
            except Exception:
                sol[col] = []
    sol["milestones"] = milestones
    sol["implementation_updates"] = updates
    sol["impact_records"] = impacts
    return sol


def list_solutions(challenge_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    query = "SELECT s.*, c.title as challenge_title, c.category as domain FROM solutions s JOIN challenges c ON c.id = s.challenge_id WHERE 1=1"
    params = []
    if challenge_id:
        query += " AND s.challenge_id = " + ("%s" if USE_POSTGRES else "?")
        params.append(challenge_id)
    if status:
        query += " AND s.status = " + ("%s" if USE_POSTGRES else "?")
        params.append(status)
    query += " ORDER BY s.created_at DESC"
    c.execute(query, tuple(params))
    rows = c.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        for col in ["technical_disciplines", "required_resources"]:
            if isinstance(d.get(col), str):
                try:
                    d[col] = json.loads(d[col])
                except Exception:
                    d[col] = []
        results.append(d)
    return results


def update_solution_status(solution_id: str, new_status: str, review_notes: Optional[str] = None) -> bool:
    sol = get_solution(solution_id)
    if not sol:
        return False
    old_status = sol["status"]
    conn = get_connection()
    c = conn.cursor()
    now_clause = "NOW()" if USE_POSTGRES else "CURRENT_TIMESTAMP"
    if USE_POSTGRES:
        c.execute(f"""
            UPDATE solutions SET status = %s, review_notes = COALESCE(%s, review_notes), updated_at = {now_clause}
            WHERE id = %s
        """, (new_status, review_notes, solution_id))
    else:
        c.execute(f"""
            UPDATE solutions SET status = ?, review_notes = COALESCE(?, review_notes), updated_at = {now_clause}
            WHERE id = ?
        """, (new_status, review_notes, solution_id))
    conn.commit()
    conn.close()

    log_audit_event("SOLUTION", solution_id, "STATUS_CHANGE", actor="REVIEWER",
                    previous_value=old_status, new_value=new_status, source="ADMIN_WORKFLOW")
    return True


def add_implementation_update(data: Dict[str, Any]) -> Dict[str, Any]:
    up_id = data.get("id") or str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("""
            INSERT INTO implementation_updates (id, solution_id, milestone_id, title, description,
                                                progress_percentage, blockers, evidence_url, author_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            up_id, data["solution_id"], data.get("milestone_id"), data["title"], data["description"],
            data.get("progress_percentage", 0), data.get("blockers"), data.get("evidence_url"),
            data.get("author_name", "Implementation Team")
        ))
    else:
        c.execute("""
            INSERT INTO implementation_updates (id, solution_id, milestone_id, title, description,
                                                progress_percentage, blockers, evidence_url, author_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            up_id, data["solution_id"], data.get("milestone_id"), data["title"], data["description"],
            data.get("progress_percentage", 0), data.get("blockers"), data.get("evidence_url"),
            data.get("author_name", "Implementation Team")
        ))
    conn.commit()
    conn.close()
    log_audit_event("SOLUTION", data["solution_id"], "IMPLEMENTATION_PROGRESS", actor=data.get("author_name", "TEAM"),
                    new_value=f"{data.get('progress_percentage', 0)}% - {data['title']}", source="IMPLEMENTATION_TRACKER")
    return {"id": up_id, "status": "SUCCESS"}


def create_impact_record(data: Dict[str, Any]) -> Dict[str, Any]:
    imp_id = data.get("id") or str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("""
            INSERT INTO impact_records (id, solution_id, challenge_id, indicator_name, baseline_value,
                                       expected_value, actual_value, unit, measurement_date, evidence_notes, verification_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            imp_id, data["solution_id"], data.get("challenge_id"), data["indicator_name"],
            data.get("baseline_value"), data.get("expected_value"), data.get("actual_value"),
            data.get("unit"), data.get("measurement_date"), data.get("evidence_notes"),
            data.get("verification_status", "VERIFIED")
        ))
    else:
        c.execute("""
            INSERT INTO impact_records (id, solution_id, challenge_id, indicator_name, baseline_value,
                                       expected_value, actual_value, unit, measurement_date, evidence_notes, verification_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            imp_id, data["solution_id"], data.get("challenge_id"), data["indicator_name"],
            data.get("baseline_value"), data.get("expected_value"), data.get("actual_value"),
            data.get("unit"), data.get("measurement_date"), data.get("evidence_notes"),
            data.get("verification_status", "VERIFIED")
        ))
    conn.commit()
    conn.close()
    log_audit_event("IMPACT", imp_id, "IMPACT_RECORDED", actor="AUDITOR",
                    new_value=f"{data['indicator_name']}: {data.get('actual_value')} {data.get('unit', '')}",
                    source="IMPACT_MODULE")
    return {"id": imp_id, "status": "SUCCESS"}


def list_impact_records(challenge_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    query = """
        SELECT i.*, s.title as solution_title, c.title as challenge_title, c.category as domain
        FROM impact_records i
        JOIN solutions s ON s.id = i.solution_id
        LEFT JOIN challenges c ON c.id = i.challenge_id
        WHERE 1=1
    """
    params = []
    if challenge_id:
        query += " AND (i.challenge_id = " + ("%s" if USE_POSTGRES else "?") + " OR s.challenge_id = " + ("%s" if USE_POSTGRES else "?") + ")"
        params.extend([challenge_id, challenge_id])
    query += " ORDER BY i.created_at DESC"
    c.execute(query, tuple(params))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ===========================================================================
# CLOSED-LOOP INTELLIGENCE: FEEDBACK, APPEALS & ROOT CAUSE LEARNING
# ===========================================================================

def submit_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    fb_id = data.get("id") or str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("""
            INSERT INTO feedback (id, target_type, target_id, user_id, user_role, rating, comment, issue_type, evidence_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            fb_id, data["target_type"], data["target_id"], data.get("user_id"),
            data.get("user_role", "CITIZEN"), data.get("rating", 5), data["comment"],
            data.get("issue_type"), data.get("evidence_url")
        ))
    else:
        c.execute("""
            INSERT INTO feedback (id, target_type, target_id, user_id, user_role, rating, comment, issue_type, evidence_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            fb_id, data["target_type"], data["target_id"], data.get("user_id"),
            data.get("user_role", "CITIZEN"), data.get("rating", 5), data["comment"],
            data.get("issue_type"), data.get("evidence_url")
        ))
    conn.commit()
    conn.close()
    log_audit_event("FEEDBACK", fb_id, "FEEDBACK_SUBMITTED", actor=data.get("user_role", "CITIZEN"),
                    new_value=f"Rating {data.get('rating')}/5 on {data['target_type']}", source="CITIZEN_FEEDBACK")
    return {"id": fb_id, "status": "SUCCESS"}


def submit_appeal(problem_id: str, appeal_type: str, reason: str, citizen_notes: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    ap_id = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("""
            INSERT INTO appeals (id, problem_id, user_id, appeal_type, reason, citizen_notes, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'SUBMITTED');
        """, (ap_id, problem_id, user_id, appeal_type, reason, citizen_notes))
    else:
        c.execute("""
            INSERT INTO appeals (id, problem_id, user_id, appeal_type, reason, citizen_notes, status)
            VALUES (?, ?, ?, ?, ?, ?, 'SUBMITTED');
        """, (ap_id, problem_id, user_id, appeal_type, reason, citizen_notes))
    conn.commit()
    conn.close()

    log_audit_event("APPEAL", ap_id, "APPEAL_SUBMITTED", actor="CITIZEN",
                    new_value=f"Appeal on {appeal_type}: {reason}", source="CLOSED_LOOP_CORRECTION")
    return {"id": ap_id, "problemId": problem_id, "status": "SUBMITTED"}


def list_appeals(status: Optional[str] = None, problem_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    query = """
        SELECT a.*, p.problem_text, p.city, p.locality, ai.category, ai.sub_category
        FROM appeals a
        JOIN problems p ON p.id = a.problem_id
        LEFT JOIN problem_ai_analysis ai ON ai.problem_id = a.problem_id
        WHERE 1=1
    """
    params = []
    if status:
        query += " AND a.status = " + ("%s" if USE_POSTGRES else "?")
        params.append(status)
    if problem_id:
        query += " AND a.problem_id = " + ("%s" if USE_POSTGRES else "?")
        params.append(problem_id)
    query += " ORDER BY a.created_at DESC"
    c.execute(query, tuple(params))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def resolve_appeal(appeal_id: str, new_status: str, reviewer_notes: Optional[str] = None, resolution_action: Optional[str] = None) -> bool:
    conn = get_connection()
    c = conn.cursor()
    now_clause = "NOW()" if USE_POSTGRES else "CURRENT_TIMESTAMP"
    if USE_POSTGRES:
        c.execute(f"""
            UPDATE appeals SET status = %s, reviewer_notes = %s, resolution_action = %s, resolved_at = {now_clause}
            WHERE id = %s
        """, (new_status, reviewer_notes, resolution_action, appeal_id))
    else:
        c.execute(f"""
            UPDATE appeals SET status = ?, reviewer_notes = ?, resolution_action = ?, resolved_at = {now_clause}
            WHERE id = ?
        """, (new_status, reviewer_notes, resolution_action, appeal_id))
    conn.commit()
    conn.close()

    log_audit_event("APPEAL", appeal_id, "APPEAL_RESOLVED", actor="REVIEWER",
                    new_value=f"Status: {new_status} | Action: {resolution_action}", source="ADMIN_CORRECTION")
    return True


def record_stakeholder_review(problem_id: str, stakeholder_profile_id: Optional[str], reviewer_name: str, action: str, reason: Optional[str] = None) -> Dict[str, Any]:
    sr_id = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("""
            INSERT INTO stakeholder_reviews (id, problem_id, stakeholder_profile_id, reviewer_name, action, reason)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (sr_id, problem_id, stakeholder_profile_id, reviewer_name, action, reason))
    else:
        c.execute("""
            INSERT INTO stakeholder_reviews (id, problem_id, stakeholder_profile_id, reviewer_name, action, reason)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (sr_id, problem_id, stakeholder_profile_id, reviewer_name, action, reason))
    conn.commit()
    conn.close()

    log_audit_event("STAKEHOLDER", problem_id, f"REVIEW_{action}", actor=reviewer_name,
                    new_value=reason or action, source="STAKEHOLDER_CONSOLE")
    return {"id": sr_id, "status": "RECORDED"}


def log_audit_event(entity_type: str, entity_id: str, event: str, actor: str = "SYSTEM",
                    previous_value: Optional[str] = None, new_value: Optional[str] = None, source: Optional[str] = None):
    try:
        conn = get_connection()
        c = conn.cursor()
        eid = str(uuid.uuid4())
        if USE_POSTGRES:
            c.execute("""
                INSERT INTO audit_log (id, entity_type, entity_id, actor, event, previous_value, new_value, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """, (eid, entity_type, entity_id, actor, event, str(previous_value) if previous_value else None,
                    str(new_value) if new_value else None, source))
        else:
            c.execute("""
                INSERT INTO audit_log (id, entity_type, entity_id, actor, event, previous_value, new_value, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (eid, entity_type, entity_id, actor, event, str(previous_value) if previous_value else None,
                    str(new_value) if new_value else None, source))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[AuditLog] Failed to log event: {e}")


def list_audit_events(entity_type: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    query = "SELECT * FROM audit_log WHERE 1=1"
    params = []
    if entity_type:
        query += " AND entity_type = " + ("%s" if USE_POSTGRES else "?")
        params.append(entity_type)
    if entity_id:
        query += " AND entity_id = " + ("%s" if USE_POSTGRES else "?")
        params.append(entity_id)
    query += " ORDER BY created_at DESC LIMIT " + ("%s" if USE_POSTGRES else "?")
    params.append(limit)
    c.execute(query, tuple(params))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ===========================================================================
# ADMIN REVIEW QUEUE & REAL ANALYTICS INTELLIGENCE
# ===========================================================================

def get_admin_review_queue() -> Dict[str, Any]:
    conn = get_connection()
    c = conn.cursor()

    # 1. Low confidence cases (< 0.85) or Critical urgency
    c.execute("""
        SELECT p.id, p.problem_text, p.city, p.locality, p.created_at,
               a.category, a.sub_category, a.urgency, a.classification_confidence as confidence
        FROM problems p
        JOIN problem_ai_analysis a ON a.problem_id = p.id
        WHERE a.classification_confidence < 0.85 OR a.urgency = 'CRITICAL'
        ORDER BY p.created_at DESC LIMIT 15;
    """)
    low_conf = [dict(r) for r in c.fetchall()]

    # 2. Unmatched problems
    c.execute("""
        SELECT p.id, p.problem_text, p.city, p.locality, a.category, p.created_at
        FROM problems p
        JOIN problem_ai_analysis a ON a.problem_id = p.id
        LEFT JOIN problem_stakeholder_recommendations r ON r.problem_id = p.id
        GROUP BY p.id
        HAVING COUNT(r.id) = 0
        ORDER BY p.created_at DESC LIMIT 10;
    """)
    unmatched = [dict(r) for r in c.fetchall()]

    # 3. Open appeals
    c.execute("""
        SELECT a.*, p.problem_text, p.city, ai.category
        FROM appeals a
        JOIN problems p ON p.id = a.problem_id
        LEFT JOIN problem_ai_analysis ai ON ai.problem_id = a.problem_id
        WHERE a.status IN ('SUBMITTED', 'UNDER_REVIEW')
        ORDER BY a.created_at DESC LIMIT 10;
    """)
    open_appeals = [dict(r) for r in c.fetchall()]

    # 4. Draft challenges awaiting validation
    c.execute("""
        SELECT id, title, category, urgency, readiness_score, challenge_status, created_at
        FROM challenges
        WHERE challenge_status IN ('DRAFT', 'VALIDATION_REQUIRED')
        ORDER BY created_at DESC LIMIT 10;
    """)
    draft_ch = [dict(r) for r in c.fetchall()]

    conn.close()
    return {
        "low_confidence_cases": low_conf,
        "unmatched_problems": unmatched,
        "open_appeals": open_appeals,
        "draft_challenges": draft_ch
    }


def get_analytics_intelligence_stats() -> Dict[str, Any]:
    conn = get_connection()
    c = conn.cursor()

    # Total counts
    c.execute("SELECT COUNT(*) as total FROM problems")
    total_problems = c.fetchone()["total"] if USE_POSTGRES else c.fetchone()[0]

    c.execute("SELECT COUNT(*) as cnt FROM problem_clusters")
    total_clusters = c.fetchone()["cnt"] if USE_POSTGRES else c.fetchone()[0]

    c.execute("SELECT COUNT(*) as cnt FROM challenges")
    total_challenges = c.fetchone()["cnt"] if USE_POSTGRES else c.fetchone()[0]

    c.execute("SELECT COUNT(*) as cnt FROM solutions")
    total_solutions = c.fetchone()["cnt"] if USE_POSTGRES else c.fetchone()[0]

    c.execute("SELECT COUNT(*) as cnt FROM appeals")
    total_appeals = c.fetchone()["cnt"] if USE_POSTGRES else c.fetchone()[0]

    # Category breakdown
    c.execute("SELECT category, COUNT(*) as count FROM problem_ai_analysis GROUP BY category ORDER BY count DESC")
    cat_rows = [{"category": (r["category"] if USE_POSTGRES else r[0]), "count": (r["count"] if USE_POSTGRES else r[1])} for r in c.fetchall()]

    # Urgency breakdown
    c.execute("SELECT urgency, COUNT(*) as count FROM problem_ai_analysis GROUP BY urgency")
    urg_rows = [{"urgency": (r["urgency"] if USE_POSTGRES else r[0]), "count": (r["count"] if USE_POSTGRES else r[1])} for r in c.fetchall()]

    # Average confidence
    c.execute("SELECT AVG(classification_confidence) as avg_conf FROM problem_ai_analysis")
    avg_conf_row = c.fetchone()
    avg_conf = round(float(avg_conf_row["avg_conf"] if USE_POSTGRES else (avg_conf_row[0] or 0.88)), 2)

    # Challenge statuses
    c.execute("SELECT challenge_status, COUNT(*) as count FROM challenges GROUP BY challenge_status")
    ch_status_rows = {((r["challenge_status"] if USE_POSTGRES else r[0]) or "UNKNOWN"): (r["count"] if USE_POSTGRES else r[1]) for r in c.fetchall()}

    # Solution statuses
    c.execute("SELECT status, COUNT(*) as count FROM solutions GROUP BY status")
    sol_status_rows = {((r["status"] if USE_POSTGRES else r[0]) or "UNKNOWN"): (r["count"] if USE_POSTGRES else r[1]) for r in c.fetchall()}

    conn.close()

    return {
        "total_problems": total_problems,
        "total_clusters": total_clusters,
        "total_challenges": total_challenges,
        "total_solutions": total_solutions,
        "total_appeals": total_appeals,
        "average_confidence": avg_conf,
        "categories": cat_rows,
        "urgency": urg_rows,
        "challenges_by_status": ch_status_rows,
        "solutions_by_status": sol_status_rows
    }


def get_root_cause_insights() -> List[Dict[str, Any]]:
    """
    Module 31: Analyzes system feedback, appeals, and stakeholder review rejections
    to identify systemic failure patterns in taxonomy, routing, and model confidence.
    """
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        SELECT appeal_type, COUNT(*) as count, reason
        FROM appeals
        GROUP BY appeal_type, reason
        ORDER BY count DESC LIMIT 5;
    """)
    appeal_patterns = [dict(r) for r in c.fetchall()]

    c.execute("""
        SELECT action, COUNT(*) as count, reason
        FROM stakeholder_reviews
        WHERE action IN ('WRONG_DOMAIN', 'CANNOT_HANDLE')
        GROUP BY action, reason
        ORDER BY count DESC LIMIT 5;
    """)
    rejection_patterns = [dict(r) for r in c.fetchall()]

    conn.close()

    insights = []
    if appeal_patterns:
        insights.append({
            "type": "TAXONOMY_DISCREPANCY",
            "severity": "MEDIUM",
            "title": "Recurring Classification Appeals Detected",
            "description": f"{len(appeal_patterns)} citizen appeal patterns suggest potential category boundary ambiguities in civic road drainage vs municipal sanitation.",
            "data": appeal_patterns
        })
    else:
        insights.append({
            "type": "HEALTHY_TAXONOMY",
            "severity": "LOW",
            "title": "High Classification Stability",
            "description": "No recurring citizen appeals currently recorded. Domain taxonomy is performing within normal bounds.",
            "data": []
        })

    if rejection_patterns:
        insights.append({
            "type": "ROUTING_MISMATCH",
            "severity": "HIGH",
            "title": "Stakeholder Domain Jurisdiction Gaps",
            "description": f"{len(rejection_patterns)} stakeholder rejections recorded. Review action areas and municipal ward mapping.",
            "data": rejection_patterns
        })
    else:
        insights.append({
            "type": "ROUTING_ACCURACY",
            "severity": "LOW",
            "title": "High Stakeholder Routing Alignment",
            "description": "Stakeholders are accepting matched action areas with zero reported jurisdictional discrepancies.",
            "data": []
        })

    return insights


# ===========================================================================
# UNIVERSITY, INDUSTRY & COLLABORATION WORKFLOW (STEP 6 EXTENSION)
# ===========================================================================

def get_university_profile(univ_id: str = "univ_coep") -> Optional[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    query = "SELECT * FROM university_profiles WHERE id = " + ("%s" if USE_POSTGRES else "?")
    c.execute(query, (univ_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    if isinstance(d.get("areas_of_expertise"), str):
        try: d["areas_of_expertise"] = json.loads(d["areas_of_expertise"])
        except: d["areas_of_expertise"] = []
    if isinstance(d.get("faculty_leads"), str):
        try: d["faculty_leads"] = json.loads(d["faculty_leads"])
        except: d["faculty_leads"] = []
    return d


def list_universities() -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM university_profiles ORDER BY institution_name ASC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    for d in rows:
        if isinstance(d.get("areas_of_expertise"), str):
            try: d["areas_of_expertise"] = json.loads(d["areas_of_expertise"])
            except: d["areas_of_expertise"] = []
    return rows


def get_industry_profile(ind_id: str = "ind_ecoroads") -> Optional[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    query = "SELECT * FROM industry_profiles WHERE id = " + ("%s" if USE_POSTGRES else "?")
    c.execute(query, (ind_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    if isinstance(d.get("technical_capabilities"), str):
        try: d["technical_capabilities"] = json.loads(d["technical_capabilities"])
        except: d["technical_capabilities"] = []
    return d


def list_industries() -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM industry_profiles ORDER BY company_name ASC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    for d in rows:
        if isinstance(d.get("technical_capabilities"), str):
            try: d["technical_capabilities"] = json.loads(d["technical_capabilities"])
            except: d["technical_capabilities"] = []
    return rows


def list_university_solutions(university_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    if university_id:
        query = """
            SELECT s.*, c.title as challenge_title, c.category as challenge_category
            FROM solutions s
            LEFT JOIN challenges c ON c.id = s.challenge_id
            WHERE s.university_id = ?
            ORDER BY s.created_at DESC
        """ if not USE_POSTGRES else """
            SELECT s.*, c.title as challenge_title, c.category as challenge_category
            FROM solutions s
            LEFT JOIN challenges c ON c.id = s.challenge_id
            WHERE s.university_id = %s
            ORDER BY s.created_at DESC
        """
        c.execute(query, (university_id,))
    else:
        c.execute("""
            SELECT s.*, c.title as challenge_title, c.category as challenge_category
            FROM solutions s
            LEFT JOIN challenges c ON c.id = s.challenge_id
            ORDER BY s.created_at DESC
        """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    for d in rows:
        if isinstance(d.get("technical_disciplines"), str):
            try: d["technical_disciplines"] = json.loads(d["technical_disciplines"])
            except: d["technical_disciplines"] = []
        if isinstance(d.get("required_resources"), str):
            try: d["required_resources"] = json.loads(d["required_resources"])
            except: d["required_resources"] = []
    return rows


def create_collaboration_request(
    solution_id: str,
    challenge_id: Optional[str],
    industry_id: str,
    company_name: str,
    why_interested: str,
    contribution: Optional[str] = None,
    technical_capability: Optional[str] = None,
    resources: Optional[str] = None,
    commercialization_capability: Optional[str] = None,
    pilot_capability: Optional[str] = None,
    message: Optional[str] = None
) -> Dict[str, Any]:
    req_id = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()

    if USE_POSTGRES:
        c.execute("""
            INSERT INTO collaboration_requests 
            (id, solution_id, challenge_id, industry_id, company_name, why_interested, contribution, technical_capability, resources, commercialization_capability, pilot_capability, message, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING', NOW(), NOW())
        """, (req_id, solution_id, challenge_id, industry_id, company_name, why_interested, contribution, technical_capability, resources, commercialization_capability, pilot_capability, message))
    else:
        c.execute("""
            INSERT INTO collaboration_requests 
            (id, solution_id, challenge_id, industry_id, company_name, why_interested, contribution, technical_capability, resources, commercialization_capability, pilot_capability, message, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (req_id, solution_id, challenge_id, industry_id, company_name, why_interested, contribution, technical_capability, resources, commercialization_capability, pilot_capability, message))
    
    conn.commit()
    conn.close()

    # Create notification for University
    create_notification(
        recipient_role="UNIVERSITY",
        recipient_id="univ_coep",
        title=f"New Collaboration Request from {company_name}",
        message=f"{company_name} is interested in co-developing solution {solution_id[:8]} with prototype & pilot support.",
        link=f"/university?tab=requests"
    )

    log_audit_event("COLLABORATION_REQUEST", req_id, "INDUSTRY", "CREATED", None, "PENDING", company_name)
    return {"id": req_id, "status": "PENDING", "company_name": company_name}


def list_collaboration_requests(solution_id: Optional[str] = None, industry_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    params = []
    where = []
    if solution_id:
        where.append("r.solution_id = " + ("%s" if USE_POSTGRES else "?"))
        params.append(solution_id)
    if industry_id:
        where.append("r.industry_id = " + ("%s" if USE_POSTGRES else "?"))
        params.append(industry_id)

    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    query = f"""
        SELECT r.*, s.title as solution_title, s.proposer_name as university_name, c.title as challenge_title
        FROM collaboration_requests r
        LEFT JOIN solutions s ON s.id = r.solution_id
        LEFT JOIN challenges c ON c.id = r.challenge_id
        {where_clause}
        ORDER BY r.created_at DESC
    """
    c.execute(query, tuple(params))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def respond_collaboration_request(request_id: str, action: str, response_notes: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection()
    c = conn.cursor()
    
    # Fetch request
    q = "SELECT * FROM collaboration_requests WHERE id = " + ("%s" if USE_POSTGRES else "?")
    c.execute(q, (request_id,))
    req = c.fetchone()
    if not req:
        conn.close()
        raise ValueError(f"Collaboration request {request_id} not found.")

    req = dict(req)
    new_status = "ACCEPTED" if action.upper() == "ACCEPT" else ("DECLINED" if action.upper() == "DECLINE" else "MORE_INFO_REQUESTED")

    # Update request
    up_q = "UPDATE collaboration_requests SET status = " + ("%s, university_response = %s, updated_at = NOW() WHERE id = %s" if USE_POSTGRES else "?, university_response = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?")
    c.execute(up_q, (new_status, response_notes, request_id))
    conn.commit()

    collab_id = None
    if new_status == "ACCEPTED":
        # Create active collaboration workspace
        collab_id = str(uuid.uuid4())
        sol_id = req["solution_id"]
        
        # Get solution
        c.execute("SELECT * FROM solutions WHERE id = " + ("%s" if USE_POSTGRES else "?"), (sol_id,))
        sol = dict(c.fetchone() or {})
        univ_name = sol.get("proposer_name", "Academic Research Lab")
        univ_id = sol.get("university_id", "univ_coep")
        
        # Insert collaboration
        if USE_POSTGRES:
            c.execute("""
                INSERT INTO collaborations 
                (id, request_id, solution_id, challenge_id, university_id, university_name, industry_id, company_name, shared_objectives, current_milestone, progress_percentage, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'Sprint 1: Joint Field Feasibility & Specs', 15, 'COLLABORATION_ACTIVE', NOW(), NOW())
            """, (collab_id, request_id, sol_id, req.get("challenge_id"), univ_id, univ_name, req["industry_id"], req["company_name"], "Execute joint lab validation, prototype assembly, and Municipal corridor field pilot."))
        else:
            c.execute("""
                INSERT INTO collaborations 
                (id, request_id, solution_id, challenge_id, university_id, university_name, industry_id, company_name, shared_objectives, current_milestone, progress_percentage, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Sprint 1: Joint Field Feasibility & Specs', 15, 'COLLABORATION_ACTIVE', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (collab_id, request_id, sol_id, req.get("challenge_id"), univ_id, univ_name, req["industry_id"], req["company_name"], "Execute joint lab validation, prototype assembly, and Municipal corridor field pilot."))

        # Initialize default collaborative tasks
        default_tasks = [
            ("Finalize Joint Technical & Material Specifications", "University"),
            ("Procure IoT Telemetry Hardware & Lab Samples", "Industry"),
            ("Bench-Test Cold-Mix Bonding / Moisture Rig", "University"),
            ("Coordinate Municipal Field Permissions for Corridor Pilot", "Industry")
        ]
        for task_title, assigned in default_tasks:
            tid = str(uuid.uuid4())
            if USE_POSTGRES:
                c.execute("INSERT INTO collaboration_tasks (id, collaboration_id, title, assigned_to, status, created_at) VALUES (%s, %s, %s, %s, 'TODO', NOW())", (tid, collab_id, task_title, assigned))
            else:
                c.execute("INSERT INTO collaboration_tasks (id, collaboration_id, title, assigned_to, status, created_at) VALUES (?, ?, ?, ?, 'TODO', CURRENT_TIMESTAMP)", (tid, collab_id, task_title, assigned))

        # Initial message
        mid = str(uuid.uuid4())
        msg = f"Collaboration initiated between {univ_name} and {req['company_name']}! Workspace is now active."
        if USE_POSTGRES:
            c.execute("INSERT INTO collaboration_messages (id, collaboration_id, sender_role, sender_name, message, created_at) VALUES (%s, %s, 'SYSTEM', 'RAKA Platform', %s, NOW())", (mid, collab_id, msg))
        else:
            c.execute("INSERT INTO collaboration_messages (id, collaboration_id, sender_role, sender_name, message, created_at) VALUES (?, ?, 'SYSTEM', 'RAKA Platform', ?, CURRENT_TIMESTAMP)", (mid, collab_id, msg))

        # Update solution status
        c.execute("UPDATE solutions SET status = 'COLLABORATION_ACTIVE' WHERE id = " + ("%s" if USE_POSTGRES else "?"), (sol_id,))

        conn.commit()

        # Create notification for Industry
        create_notification(
            recipient_role="INDUSTRY",
            recipient_id=req["industry_id"],
            title=f"Collaboration Accepted by {univ_name}!",
            message=f"Your collaboration request for '{sol.get('title')}' has been accepted. Joint workspace is ready.",
            link=f"/collaborations/{collab_id}"
        )

        log_audit_event("COLLABORATION", collab_id, "UNIVERSITY", "ACCEPTED", "PENDING", "COLLABORATION_ACTIVE", f"Accepted by {univ_name}")

    conn.close()
    return {"requestId": request_id, "status": new_status, "collaborationId": collab_id}


def get_collaboration(collab_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT c.*, s.title as solution_title, s.description as solution_description, s.solution_type, 
               s.technical_disciplines, ch.title as challenge_title, ch.problem_statement, ch.category, ch.urgency
        FROM collaborations c
        LEFT JOIN solutions s ON s.id = c.solution_id
        LEFT JOIN challenges ch ON ch.id = c.challenge_id
        WHERE c.id = """ + ("%s" if USE_POSTGRES else "?"), (collab_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    d = dict(row)

    # Fetch tasks
    c.execute("SELECT * FROM collaboration_tasks WHERE collaboration_id = " + ("%s" if USE_POSTGRES else "?") + " ORDER BY created_at ASC", (collab_id,))
    d["tasks"] = [dict(r) for r in c.fetchall()]

    # Fetch messages
    c.execute("SELECT * FROM collaboration_messages WHERE collaboration_id = " + ("%s" if USE_POSTGRES else "?") + " ORDER BY created_at ASC", (collab_id,))
    d["messages"] = [dict(r) for r in c.fetchall()]

    conn.close()
    return d


def list_collaborations(university_id: Optional[str] = None, industry_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    params = []
    where = []
    if university_id:
        where.append("c.university_id = " + ("%s" if USE_POSTGRES else "?"))
        params.append(university_id)
    if industry_id:
        where.append("c.industry_id = " + ("%s" if USE_POSTGRES else "?"))
        params.append(industry_id)

    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    query = f"""
        SELECT c.*, s.title as solution_title, ch.title as challenge_title
        FROM collaborations c
        LEFT JOIN solutions s ON s.id = c.solution_id
        LEFT JOIN challenges ch ON ch.id = c.challenge_id
        {where_clause}
        ORDER BY c.updated_at DESC
    """
    c.execute(query, tuple(params))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def update_collaboration_progress(collab_id: str, progress_percentage: int, milestone: Optional[str] = None, status: Optional[str] = None) -> bool:
    conn = get_connection()
    c = conn.cursor()
    sets = ["progress_percentage = " + ("%s" if USE_POSTGRES else "?")]
    params = [progress_percentage]
    if milestone:
        sets.append("current_milestone = " + ("%s" if USE_POSTGRES else "?"))
        params.append(milestone)
    if status:
        sets.append("status = " + ("%s" if USE_POSTGRES else "?"))
        params.append(status)

    sets.append("updated_at = " + ("NOW()" if USE_POSTGRES else "CURRENT_TIMESTAMP"))
    params.append(collab_id)

    q = f"UPDATE collaborations SET {', '.join(sets)} WHERE id = " + ("%s" if USE_POSTGRES else "?")
    c.execute(q, tuple(params))
    rc = c.rowcount
    conn.commit()
    conn.close()
    return rc > 0


def add_collaboration_task(collab_id: str, title: str, assigned_to: str = "University", due_date: Optional[str] = None) -> Dict[str, Any]:
    tid = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("INSERT INTO collaboration_tasks (id, collaboration_id, title, assigned_to, status, due_date, created_at) VALUES (%s, %s, %s, %s, 'TODO', %s, NOW())", (tid, collab_id, title, assigned_to, due_date))
    else:
        c.execute("INSERT INTO collaboration_tasks (id, collaboration_id, title, assigned_to, status, due_date, created_at) VALUES (?, ?, ?, ?, 'TODO', ?, CURRENT_TIMESTAMP)", (tid, collab_id, title, assigned_to, due_date))
    conn.commit()
    conn.close()
    return {"id": tid, "collaboration_id": collab_id, "title": title, "assigned_to": assigned_to, "status": "TODO"}


def toggle_collaboration_task(task_id: str, new_status: str) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE collaboration_tasks SET status = " + ("%s WHERE id = %s" if USE_POSTGRES else "? WHERE id = ?"), (new_status, task_id))
    rc = c.rowcount
    conn.commit()
    conn.close()
    return rc > 0


def add_collaboration_message(collab_id: str, sender_role: str, sender_name: str, message: str) -> Dict[str, Any]:
    mid = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("INSERT INTO collaboration_messages (id, collaboration_id, sender_role, sender_name, message, created_at) VALUES (%s, %s, %s, %s, %s, NOW())", (mid, collab_id, sender_role, sender_name, message))
    else:
        c.execute("INSERT INTO collaboration_messages (id, collaboration_id, sender_role, sender_name, message, created_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)", (mid, collab_id, sender_role, sender_name, message))
    conn.commit()
    conn.close()
    return {"id": mid, "collaboration_id": collab_id, "sender_role": sender_role, "sender_name": sender_name, "message": message}


def create_notification(recipient_role: str, recipient_id: Optional[str], title: str, message: str, link: Optional[str] = None) -> str:
    nid = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    if USE_POSTGRES:
        c.execute("INSERT INTO notifications (id, recipient_role, recipient_id, title, message, link, is_read, created_at) VALUES (%s, %s, %s, %s, %s, %s, 0, NOW())", (nid, recipient_role, recipient_id, title, message, link))
    else:
        c.execute("INSERT INTO notifications (id, recipient_role, recipient_id, title, message, link, is_read, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP)", (nid, recipient_role, recipient_id, title, message, link))
    conn.commit()
    conn.close()
    return nid


def get_notifications(recipient_role: str, recipient_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    query = "SELECT * FROM notifications WHERE recipient_role = " + ("%s" if USE_POSTGRES else "?") + " ORDER BY created_at DESC LIMIT 20"
    c.execute(query, (recipient_role,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def mark_notification_read(notif_id: str) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE notifications SET is_read = 1 WHERE id = " + ("%s" if USE_POSTGRES else "?"), (notif_id,))
    rc = c.rowcount
    conn.commit()
    conn.close()
    return rc > 0
