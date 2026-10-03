# RAKA Domain AI — Steps 1 to 4
## Citizen Problem Classification, Multimodal AI Analysis & Intelligent Routing System

RAKA (**Rashtriya Adhikar & Karyavahi AI**) is an enterprise-grade Civic Problem Intelligence and Routing Platform engineered for Smart India Hackathon and civic governance.

Instead of treating citizen grievances as isolated text complaints, RAKA combines **what citizens speak**, **what photographs capture**, **where the problem is situated**, and **contextual engineering disciplines** to synthesize a structured understanding, link related community reports into **Problem Clusters**, and intelligently route them to **Public Authorities**, **Research & Innovation Partners**, and **Civic Action Areas**.

---

## 1. Complete RAKA Workflow

```
[ STEP 1: REPORT ]
Text + Real Browser Mic (Voice) + Photo (Quality Checked) + Mandatory Location (GPS / Fallback)
       │
       ▼
[ STEP 2: UNDERSTAND ]
Multimodal AI Synthesis
├── Citizen Card: "RAKA UNDERSTANDS YOUR PROBLEM" (English / हिन्दी / मराठी)
└── Judge Drawer: 7 Inspection Tabs (Input, Vision, Voice, Classification, Impact, Routing, Uncertainty)
       │
       ▼
[ STEP 3: CLUSTER ]
Semantic Similarity + Distance + Category + Discipline + Temporal Decay
├── Multi-Signal Scoring Engine (No-Penalty Privacy Rule)
└── Problem Clusters & Visual Indicator Tree
       │
       ▼
[ STEP 4: ROUTE ]
Multimodal Intelligent Matching
├── Government / Public Authorities (PWD, Municipal Corporations, DISCOMs, Jal Boards)
├── Research & Innovation Partners (CRRI, IIT Engineering Depts, NEERI, NIUA)
└── Action Areas (Concrete intervention scope with realistic turnaround timelines)
```

---

## 2. Key Capabilities for Hackathon Judges

1. **Multimodal Ground Truth:** AI does not rely on typed text alone. Spoken testimony (Hindi, Marathi, English, Hinglish) and photographs are verified and separated into:
   - `OBSERVED_IN_IMAGE`: Concrete physical evidence verified from the camera.
   - `REPORTED_BY_CITIZEN`: Spoken or written claims by the citizen.
   - `INFERRED_FROM_CONTEXT`: Technical deductions by AI (engineering discipline, safety hazards).
2. **Citizen-Friendly + Technical Dual View:**
   - **Normal Citizen View:** Clean, jargon-free summary with instant language switching (`[ English ] [ हिन्दी ] [ मराठी ]`) and an explainable `[ 💡 Why this classification? ]` panel.
   - **Judge / Technical View:** Slide-over drawer with 7 in-depth tabs (`INPUT`, `VISION`, `VOICE`, `CLASSIFICATION`, `IMPACT`, `ROUTING`, `UNCERTAINTY`).
3. **Mandatory Location Architecture:** Location is strictly required via GPS or manual fallback (State, District, City, Locality). Exact coordinates are kept private and generalized to neighborhood levels.
4. **Step 4 Intelligent Matching:** Matched against 53 verified stakeholder profiles, 53 matching rules, and 34 action types across all 15 civic domains.
5. **No Auto-Merge Guarantee:** Citizen reports are never silenced, deleted, or forcibly merged. Citizens retain sovereignty over their complaints.

---

## 3. Technology Stack

| Layer | Component | Description |
|---|---|---|
| **Frontend** | Single Page Application (`index.html`) | Semantic HTML5, Vanilla CSS3 design system, and Vanilla JavaScript. Zero external UI framework overhead. |
| **Backend** | FastAPI (`backend/main.py`) | Async Python ASGI framework with strict Pydantic schemas. |
| **Generative AI** | Gemini 2.5 / 3.8 Flash | Multimodal synthesis, multilingual transcription, and domain classification. |
| **Vector Engine** | `gemini-embedding-2` / Multi-Signal | 768-dimensional normalized embeddings + geographical and domain scoring. |
| **Database** | PostgreSQL / SQLite (`backend/database.py`) | Relational schema with HNSW vector indexing, RLS, and local SQLite support. |
| **Matching Registry** | Matching Engine (`backend/matching_service.py`) | 15 civic domains, 53 stakeholder profiles, 53 matching rules, and 34 action types. |

---

## 4. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check returning operational status and active capabilities |
| `POST` | `/api/problems/analyze` | Unified multimodal submission (enforces mandatory location, voice, image, text) |
| `GET` | `/api/problems/{id}/analysis` | Fetches complete problem record, citizen explanation, clustering, and Step 4 matches |
| `POST` | `/api/voice/transcribe` | Standalone voice upload and multilingual speech-to-text with script classification |
| `POST` | `/api/location/reverse` | Privacy-preserving reverse geocoding to generalized locality and city |
| `GET` | `/api/clusters/{id}` | Detailed cluster breakdown with visual indicator tree and member problems |

---

## 5. Setup & Running Locally

### 5.1 Prerequisites
- Python 3.10+
- (Optional) `GEMINI_API_KEY` for Google Gemini Studio

### 5.2 Installation
```bash
git clone <repo-url>
cd RAKA
pip install -r requirements.txt
cp .env.example .env
```

### 5.3 Database Seeding
Populate realistic demonstration records across civic domains:
```bash
python -m backend.seed_demo
```

### 5.4 Start Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser.

---

## 6. Automated Testing & Verification

Run the full pytest suite (33 passing tests covering Part AV test cases):
```bash
python -m pytest backend/ -v
```

Run similarity benchmark evaluation on labeled pairs:
```bash
python -m eval.evaluate
```

Verify deterministic cluster rebuilding:
```bash
python -m backend.rebuild_clusters
```
