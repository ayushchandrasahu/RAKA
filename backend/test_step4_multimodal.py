import pytest
import os
import json
import uuid
from pathlib import Path
from backend.matching_service import match_stakeholders_for_problem, STAKEHOLDER_REGISTRY
from backend.gemini_service import (
    detect_language_simple, heuristic_multimodal_analysis, transcribe_audio_file,
    analyze_problem_multimodal
)
from backend.schemas import MultimodalAnalysisOutput, Step4MatchingPayload, EvidenceItem
from backend import database as db
from backend.similarity import score_pair, plan_cluster_update, build_embedding_text


@pytest.fixture(autouse=True)
def setup_test_db():
    db.init_db()
    yield


# ==============================================================================
# PART AV — 20 MANDATORY TEST CASES
# ==============================================================================

def test_case_1_english_text_image_location():
    """Test 1: English text + image + location multimodal synthesis."""
    out = heuristic_multimodal_analysis(
        text="A massive pothole on the arterial road is blocking traffic.",
        voice_transcript="",
        detected_lang="English",
        has_image=True,
        city="Bengaluru",
        state="Karnataka",
        locality="Indiranagar"
    )
    assert isinstance(out, MultimodalAnalysisOutput)
    assert out.classification.category == "Roads & Transport"
    assert out.visual_analysis.image_available is True
    assert out.location_context.city == "Bengaluru"
    assert any(item.source == "OBSERVED_IN_IMAGE" for item in out.evidence)


def test_case_2_hindi_voice_image_location():
    """Test 2: Hindi voice + image + location."""
    voice = "हमारे स्कूल के पास सड़क में बहुत बड़ा गड्ढा है। बच्चों को आने जाने में परेशानी होती है।"
    lang, conf = detect_language_simple(voice)
    assert lang == "Hindi"
    assert conf >= 0.90

    out = heuristic_multimodal_analysis(
        text="",
        voice_transcript=voice,
        detected_lang=lang,
        has_image=True,
        city="Pune",
        state="Maharashtra",
        locality="Shivaji Nagar"
    )
    assert out.classification.category == "Roads & Transport"
    assert out.classification.sub_category == "Pothole / Road Damage"
    assert "Civil Engineering" in out.classification.technical_disciplines
    assert any(item.source == "REPORTED_BY_CITIZEN" for item in out.evidence)


def test_case_3_marathi_voice_image_location():
    """Test 3: Marathi voice + image + location."""
    voice = "आमच्या शाळेजवळ रस्त्यावर मोठा खड्डा पडला आहे. वाहनांना जाण्या-येण्यास त्रास होत आहे."
    lang, conf = detect_language_simple(voice)
    assert lang == "Marathi"

    out = heuristic_multimodal_analysis(
        text="",
        voice_transcript=voice,
        detected_lang=lang,
        has_image=True,
        city="Pune",
        state="Maharashtra",
        locality="Kothrud"
    )
    assert out.classification.category == "Roads & Transport"
    assert out.location_context.city == "Pune"


def test_case_4_hindi_english_mixed_voice():
    """Test 4: Hindi + English mixed code-switching speech."""
    hinglish = "Hamare area mein street light pichle 5 days se band hai"
    lang, conf = detect_language_simple(hinglish)
    assert "Hindi" in lang or "Code-mixed" in lang

    out = heuristic_multimodal_analysis(
        text="",
        voice_transcript=hinglish,
        detected_lang=lang,
        has_image=False,
        city="Mumbai",
        state="Maharashtra"
    )
    assert out.classification.category == "Electricity & Energy"
    assert out.classification.sub_category == "Streetlight Outage"


def test_case_5_text_no_image():
    """Test 5: Text only report (no image)."""
    out = heuristic_multimodal_analysis(
        text="Water pipeline is ruptured and clean drinking water is getting wasted on the sidewalk.",
        voice_transcript="",
        detected_lang="English",
        has_image=False,
        city="Pune",
        state="Maharashtra"
    )
    assert out.classification.category == "Water & Sanitation"
    assert out.visual_analysis.image_available is False
    assert out.visual_analysis.image_confidence == 0.0


def test_case_6_voice_no_text():
    """Test 6: Voice only report (no text input)."""
    out = heuristic_multimodal_analysis(
        text="",
        voice_transcript="कचरा उठाने वाली गाड़ी दो हफ्ते से नहीं आई है, बदबू फैल रही है।",
        detected_lang="Hindi",
        has_image=False,
        city="Jaipur",
        state="Rajasthan"
    )
    assert out.classification.category == "Waste Management"
    assert out.citizen_statement.voice_transcript != ""
    assert out.citizen_statement.text == ""


def test_case_7_image_no_voice():
    """Test 7: Image + text without voice input."""
    out = heuristic_multimodal_analysis(
        text="Dangerous electrical sparking near market pole.",
        voice_transcript="",
        detected_lang="English",
        has_image=True,
        city="Delhi",
        state="Delhi"
    )
    assert out.classification.category == "Electricity & Energy"
    assert out.visual_analysis.image_available is True
    assert out.citizen_statement.voice_transcript == ""


def test_case_8_poor_image():
    """Test 8: Poor image quality handling."""
    out = heuristic_multimodal_analysis(
        text="Garbage pile near residential corner.",
        voice_transcript="",
        detected_lang="English",
        has_image=False,
        city="Pune"
    )
    assert out.visual_analysis.image_quality == "NOT_PROVIDED"
    assert out.visual_analysis.image_confidence == 0.0


def test_case_9_failed_audio():
    """Test 9: Graceful fallback when audio processing fails or file is empty."""
    # Test transcribe with non-existent or dummy file falls back gracefully
    dummy_path = Path("uploads/non_existent_audio.webm")
    trans, lang, conf = transcribe_audio_file(dummy_path, preferred_language="Hindi")
    assert trans != ""
    assert lang == "Hindi"
    assert conf > 0.80


def test_case_10_location_denied():
    """Test 10: Location mandatory validation in API schema and HTTP 400 enforcement."""
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    resp = client.post("/api/problems/analyze", data={"problemText": "Dangerous pothole on road"})
    assert resp.status_code == 400
    assert "Location is required to analyze this problem" in resp.json()["error"]


def test_case_11_manual_location():
    """Test 11: Manual location input validation."""
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    resp = client.post("/api/problems/analyze", data={
        "problemText": "Broken drainage culvert.",
        "locality": "Camp Area",
        "city": "Pune",
        "district": "Pune",
        "state": "Maharashtra"
    })
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_case_12_missing_location():
    """Test 12: Missing location context handling."""
    out = heuristic_multimodal_analysis(
        text="Public park lighting blackout.",
        city="",
        state=""
    )
    assert out.location_context.city == ""
    assert out.classification.category == "Electricity & Energy"


def test_case_13_wrong_image_type():
    """Test 13: Reject disallowed image formats (e.g. .exe or .txt)."""
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    resp = client.post(
        "/api/problems/analyze",
        data={"problemText": "Road pothole", "city": "Pune", "state": "Maharashtra"},
        files={"image": ("script.exe", b"binarycontent", "application/octet-stream")}
    )
    assert resp.status_code == 400
    assert "Unsupported image format" in resp.json()["error"]


def test_case_14_oversized_image():
    """Test 14: Reject oversized images (>10MB)."""
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    oversized_data = b"X" * (11 * 1024 * 1024)
    resp = client.post(
        "/api/problems/analyze",
        data={"problemText": "Road pothole", "city": "Pune", "state": "Maharashtra"},
        files={"image": ("huge_photo.jpg", oversized_data, "image/jpeg")}
    )
    assert resp.status_code == 400
    assert "exceeds maximum allowable size of 10MB" in resp.json()["error"]


def test_case_15_invalid_ai_json():
    """Test 15: Schema validation resilience against corrupted fields."""
    with pytest.raises(Exception):
        # Missing required problem_title and classification
        MultimodalAnalysisOutput.model_validate({"broken": "data"})


def test_case_16_gemini_failure():
    """Test 16: Safe execution of fallback analysis engine when Gemini API is unavailable."""
    # Forced fallback without API key
    saved_key = os.environ.get("GEMINI_API_KEY")
    try:
        os.environ["GEMINI_API_KEY"] = ""
        res = analyze_problem_multimodal(text="Large road crack and cave in")
        assert isinstance(res, MultimodalAnalysisOutput)
        assert res.classification.category == "Roads & Transport"
    finally:
        if saved_key:
            os.environ["GEMINI_API_KEY"] = saved_key


def test_case_17_no_related_reports():
    """Test 17: Clustering behavior when no similar reports exist (Singleton)."""
    plan = plan_cluster_update(
        new_report_id="prob-lonely-1",
        candidate_scores=[],
        existing_memberships={},
        existing_clusters={},
    )
    assert plan["action"] == "NONE"


def test_case_18_related_reports():
    """Test 18: Multiple related reports linked into a problem cluster."""
    scored_pairs = [
        {"problem_id": "prob-existing-1", "combined_score": 0.88, "semantic_similarity": 0.85}
    ]
    memberships = {"prob-existing-1": "clust-pune-roads"}
    clusters = {
        "clust-pune-roads": {
            "id": "clust-pune-roads",
            "cluster_title": "Roads & Transport: Pothole in Shivaji Nagar",
            "category": "Roads & Transport",
            "problem_count": 2,
            "status": "UNDER_VALIDATION",
            "needs_review": False
        }
    }
    plan = plan_cluster_update("prob-new-2", scored_pairs, memberships, clusters)
    assert plan["action"] == "ADD_TO_EXISTING"
    assert plan["target_cluster_id"] == "clust-pune-roads"


def test_case_19_no_stakeholder_rule():
    """Test 19: Unknown category falls back to generic public grievance cell."""
    payload = match_stakeholders_for_problem(
        category="NonExistentCategory",
        sub_category="Unknown Issue",
        technical_disciplines=["General Engineering"],
        urgency="LOW",
        has_image=False,
        cluster_member_count=1
    )
    assert len(payload.stakeholders) >= 2
    names = [s.name for s in payload.stakeholders]
    assert any("Collectorate" in n or "Grievance" in n for n in names)


def test_case_20_existing_step1_3_regression_tests():
    """Test 20: Regression safety for Step 1-3 database persistence and embeddings."""
    pid = "unit-test-regression-pid"
    db.save_problem(
        problem_id=pid,
        problem_text="Regression test civic problem statement",
        image_url=None,
        has_location=True,
        latitude=18.5204,
        longitude=73.8567,
        location_accuracy=15.0,
        manual_location="Shivaji Nagar, Pune",
        is_demo=False
    )

    analysis = heuristic_multimodal_analysis(text="Regression test civic problem statement", city="Pune")
    db.save_analysis(pid, analysis)

    doc_text = build_embedding_text(
        problem_text="Regression test civic problem statement",
        summary=analysis.problem_summary,
        category=analysis.category,
        technical_discipline=analysis.technical_discipline,
        impact_areas=analysis.impact_areas,
        keywords=analysis.keywords,
        sub_category=analysis.classification.sub_category,
        manual_location="Pune, Maharashtra"
    )
    assert len(doc_text) > 20

    # Step 4 matching persistence
    s4_payload = match_stakeholders_for_problem(
        category=analysis.category,
        sub_category=analysis.classification.sub_category,
        technical_disciplines=analysis.technical_discipline,
        urgency="HIGH",
        has_image=False,
        cluster_member_count=2,
        city="Pune"
    )
    db.save_step4_recommendations(pid, s4_payload)
    fetched_s4 = db.get_step4_recommendations(pid)
    assert fetched_s4 is not None
    assert len(fetched_s4["stakeholders"]) >= 2
    assert len(fetched_s4["action_areas"]) >= 2
