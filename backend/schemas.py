from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Citizen & Multimodal Statement Schemas
# ---------------------------------------------------------------------------
class CitizenStatement(BaseModel):
    text: Optional[str] = ""
    voice_transcript: Optional[str] = ""
    detected_language: Optional[str] = "English"
    input_languages: List[str] = Field(default_factory=lambda: ["English"])

class LocationContext(BaseModel):
    city: Optional[str] = ""
    district: Optional[str] = ""
    state: Optional[str] = ""
    locality: Optional[str] = ""
    location_available: bool = True

class VisualAnalysis(BaseModel):
    image_available: bool = False
    visible_objects: List[str] = Field(default_factory=list)
    visible_problem_evidence: List[str] = Field(default_factory=list)
    environment_context: List[str] = Field(default_factory=list)
    image_quality: str = "GOOD"  # GOOD, FAIR, POOR
    image_confidence: float = 0.85

class ProblemUnderstanding(BaseModel):
    primary_issue: str
    affected_area: str = ""
    affected_people_or_systems: List[str] = Field(default_factory=list)
    possible_causes: List[str] = Field(default_factory=list)
    observed_impacts: List[str] = Field(default_factory=list)
    reported_duration: str = "Recently"
    reported_frequency: str = "Ongoing"

class ClassificationData(BaseModel):
    category: str
    sub_category: str
    technical_disciplines: List[str]
    problem_type: str = "Physical Infrastructure"
    urgency: str  # LOW, MEDIUM, HIGH, CRITICAL
    complexity: str  # LOW, MEDIUM, HIGH

class ImpactData(BaseModel):
    impact_areas: List[str]
    severity_reason: str

class SolutionContext(BaseModel):
    possible_solution_types: List[str]
    required_skills: List[str]
    potential_action_areas: List[str]

class RoutingContext(BaseModel):
    suggested_stakeholder_types: List[str]
    suggested_departments: List[str]

class EvidenceItem(BaseModel):
    source: str  # IMAGE, VOICE, TEXT, LOCATION, CLUSTER, OBSERVED_IN_IMAGE, REPORTED_BY_CITIZEN, INFERRED_FROM_CONTEXT
    statement: str
    confidence: float = 0.90

class EvidenceCategorized(BaseModel):
    text_evidence: List[str] = Field(default_factory=list)
    voice_evidence: List[str] = Field(default_factory=list)
    image_evidence: List[str] = Field(default_factory=list)
    location_evidence: List[str] = Field(default_factory=list)
    cluster_evidence: List[str] = Field(default_factory=list)

class UncertaintyData(BaseModel):
    missing_information: List[str] = Field(default_factory=list)
    uncertain_fields: List[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Master Multimodal Step 2 Analysis Schema
# ---------------------------------------------------------------------------
class MultimodalAnalysisOutput(BaseModel):
    problem_title: str
    problem_summary: str
    citizen_statement: CitizenStatement
    location_context: LocationContext
    visual_analysis: VisualAnalysis
    problem_understanding: ProblemUnderstanding
    classification: ClassificationData
    impact: ImpactData
    solution_context: SolutionContext
    routing_context: RoutingContext
    evidence: Any = Field(default_factory=list)  # Accepts List[EvidenceItem] or EvidenceCategorized or dict
    uncertainty: UncertaintyData = Field(default_factory=UncertaintyData)
    classification_confidence: float = 0.92
    image_confidence: float = 0.88
    language_confidence: float = 0.95
    confidence: float = 0.90

    # Backwards compatibility getters
    @property
    def category(self) -> str:
        return self.classification.category

    @property
    def urgency(self) -> str:
        return self.classification.urgency

    @property
    def complexity(self) -> str:
        return self.classification.complexity

    @property
    def technical_discipline(self) -> List[str]:
        return self.classification.technical_disciplines

    @property
    def suggested_departments(self) -> List[str]:
        return self.routing_context.suggested_departments

    @property
    def image_observations(self) -> List[str]:
        return self.visual_analysis.visible_problem_evidence

    @property
    def impact_areas(self) -> List[str]:
        return self.impact.impact_areas

    @property
    def required_skills(self) -> List[str]:
        return self.solution_context.required_skills

    @property
    def possible_solution_types(self) -> List[str]:
        return self.solution_context.possible_solution_types

    @property
    def estimated_project_duration(self) -> str:
        return "2 - 5 days"

    @property
    def keywords(self) -> List[str]:
        return [self.classification.sub_category] + self.visual_analysis.visible_objects[:4]

    @property
    def reasoning_summary(self) -> str:
        return f"Identified as {self.classification.category} ({self.classification.sub_category}) based on multimodal analysis of citizen reports, physical visual evidence, and location context."

# Alias for backwards compatibility with Step 2/3
AnalysisOutput = MultimodalAnalysisOutput


# ---------------------------------------------------------------------------
# Step 4 Matching Schemas
# ---------------------------------------------------------------------------
class StakeholderRecommendation(BaseModel):
    stakeholder_type: str  # Government/Public, R&I, Technical, Industry/Community
    name: str
    jurisdiction_or_scope: str
    relevance_level: str  # HIGH, MEDIUM, LOW
    relevance_score: int  # 0 to 100
    why_matched: List[str]

class ActionAreaRecommendation(BaseModel):
    title: str
    action_type: str  # Assessment, Inspection, Repair Feasibility, Resource Deployment
    description: str
    estimated_timeline: str
    why_recommended: List[str]

class Step4MatchingPayload(BaseModel):
    problem_title: str
    category: str
    sub_category: str
    technical_disciplines: List[str]
    stakeholders: List[StakeholderRecommendation]
    action_areas: List[ActionAreaRecommendation]
    why_recommendations: List[str]
    disclaimer: str = "These are AI-assisted stakeholder recommendations, not official department assignments."


# ---------------------------------------------------------------------------
# Step 3 Similarity & Cluster Schemas
# ---------------------------------------------------------------------------
class MatchDetail(BaseModel):
    match_level: str  # POSSIBLE_DUPLICATE or RELATED_REPORT
    combined_score: float
    percentage: int
    other_problem_id: Optional[str] = None
    other_summary: str
    other_category: str
    other_date: str
    other_status: str
    semantic_similarity: float
    category_match: str
    discipline_match: str
    distance_display: str
    temporal_display: str
    reasons: List[str]

class SimilarityPayload(BaseModel):
    status: str
    headline_message: str
    related_count: int
    cluster_id: Optional[str] = None
    cluster_title: Optional[str] = None
    cluster_report_count: Optional[int] = None
    matches: List[MatchDetail] = Field(default_factory=list)

class ClusterMemberSummary(BaseModel):
    problem_summary: str
    category: str
    reported_date: str
    status: str
    link_percentage: Optional[int] = None

class ClusterPayload(BaseModel):
    id: str
    title: str
    category: str
    problem_count: int
    recent_count_30d: int
    status: str
    needs_review: bool
    review_notice: Optional[str] = None
    generalized_area: str
    average_similarity_pct: int
    impact_areas: List[str]
    technical_disciplines: List[str]
    why_grouped: List[Dict[str, str]]
    members: List[ClusterMemberSummary]


# ---------------------------------------------------------------------------
# Step 5: Intelligent Challenge & Chatbot Schemas
# ---------------------------------------------------------------------------

class ChallengeReadiness(BaseModel):
    is_ready: bool
    score: float
    reasons: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ChallengeOutputAI(BaseModel):
    title: str
    short_description: str
    problem_statement: str
    category: str
    sub_category: Optional[str] = ""
    problem_type: Optional[str] = "Civic Infrastructure Challenge"
    affected_area: Optional[str] = ""
    impact_summary: str
    urgency: str
    complexity: str
    evidence_summary: List[str] = Field(default_factory=list)
    technical_disciplines: List[str] = Field(default_factory=list)
    required_skills: List[str] = Field(default_factory=list)
    stakeholder_types: List[str] = Field(default_factory=list)
    action_areas: List[str] = Field(default_factory=list)
    possible_solution_types: List[str] = Field(default_factory=list)
    expected_outcomes: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    success_indicators: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    challenge_readiness: Optional[Dict[str, Any]] = None


class ChallengeEvidenceItem(BaseModel):
    id: Optional[str] = None
    evidence_type: str  # REPORTED_BY_CITIZEN, OBSERVED_IN_IMAGE, FROM_VOICE_TRANSCRIPT, SUPPORTED_BY_CLUSTER, AI_CLASSIFICATION, LOCATION_CONTEXT, STAKEHOLDER_MATCH, INFERRED_FROM_CONTEXT, NOT_AVAILABLE
    evidence_text: str
    source_reference: Optional[str] = None
    confidence: float = 0.90


class ChallengeRecord(BaseModel):
    id: str
    cluster_id: Optional[str] = None
    title: str
    short_description: str
    problem_statement: str
    category: str
    sub_category: Optional[str] = ""
    problem_type: Optional[str] = "Civic Infrastructure Challenge"
    affected_area: Optional[str] = ""
    generalized_location: Optional[str] = ""
    impact_summary: str
    urgency: str
    complexity: str
    challenge_status: str  # DRAFT, VALIDATION_REQUIRED, VALIDATED, READY_FOR_SOLUTIONS, PUBLISHED, IN_PROGRESS, RESOLVED, ARCHIVED
    readiness_score: float
    readiness_reasons: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    evidence_summary: List[str] = Field(default_factory=list)
    technical_disciplines: List[str] = Field(default_factory=list)
    required_skills: List[str] = Field(default_factory=list)
    stakeholder_types: List[str] = Field(default_factory=list)
    action_areas: List[str] = Field(default_factory=list)
    possible_solution_types: List[str] = Field(default_factory=list)
    expected_outcomes: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    success_indicators: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    published_at: Optional[str] = None
    resolved_at: Optional[str] = None
    evidence: List[ChallengeEvidenceItem] = Field(default_factory=list)


class ChallengeDashboardStats(BaseModel):
    total: int = 0
    draft: int = 0
    validation_required: int = 0
    validated: int = 0
    ready_for_solutions: int = 0
    in_progress: int = 0
    resolved: int = 0


class ChatRequest(BaseModel):
    message: str
    problemId: Optional[str] = None
    clusterId: Optional[str] = None
    challengeId: Optional[str] = None
    language: Optional[str] = "en"
    conversationId: Optional[str] = None


class ChatSource(BaseModel):
    type: str  # AI_ANALYSIS, CLUSTER, CHALLENGE, STAKEHOLDER, ACTION_AREA, IMAGE_EVIDENCE, VOICE_TRANSCRIPT, CITIZEN_REPORT
    field: str
    snippet: Optional[str] = None


class ChatResponse(BaseModel):
    message: str
    language: str = "en"
    conversationId: str
    contextType: str  # PROBLEM, CLUSTER, CHALLENGE, GENERAL
    sources: List[ChatSource] = Field(default_factory=list)
    relatedActions: List[str] = Field(default_factory=list)



# ===========================================================================
# MODULE 7, 8, 9, 14: CLOSED-LOOP INTELLIGENCE & TRACKING SCHEMAS
# ===========================================================================

class SolutionCreate(BaseModel):
    challenge_id: str
    title: str
    description: str
    proposer_name: str
    proposer_type: str = "INNOVATOR"  # STARTUP, UNIVERSITY, RESEARCHER, CITIZEN, NGO, INDUSTRY
    technical_disciplines: List[str] = Field(default_factory=list)
    required_resources: List[str] = Field(default_factory=list)
    expected_outcome: Optional[str] = ""
    constraints: Optional[str] = ""


class SolutionRecord(BaseModel):
    id: str
    challenge_id: str
    challenge_title: Optional[str] = None
    domain: Optional[str] = None
    title: str
    description: str
    proposer_name: str
    proposer_type: str
    technical_disciplines: List[str] = Field(default_factory=list)
    required_resources: List[str] = Field(default_factory=list)
    expected_outcome: Optional[str] = None
    constraints: Optional[str] = None
    status: str  # IDEA, PROPOSED, UNDER_REVIEW, ACCEPTED, PILOT, IMPLEMENTING, IMPLEMENTED, REJECTED, ARCHIVED
    review_notes: Optional[str] = None
    created_at: str
    updated_at: str
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    implementation_updates: List[Dict[str, Any]] = Field(default_factory=list)
    impact_records: List[Dict[str, Any]] = Field(default_factory=list)


class ImplementationUpdateCreate(BaseModel):
    solution_id: str
    milestone_id: Optional[str] = None
    title: str
    description: str
    progress_percentage: int = 0
    blockers: Optional[str] = None
    evidence_url: Optional[str] = None
    author_name: str = "Implementation Team"


class ImpactRecordCreate(BaseModel):
    solution_id: str
    challenge_id: Optional[str] = None
    indicator_name: str
    baseline_value: Optional[str] = None
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    unit: Optional[str] = None
    measurement_date: Optional[str] = None
    evidence_notes: Optional[str] = None
    verification_status: str = "VERIFIED"


class FeedbackCreate(BaseModel):
    target_type: str  # PROBLEM, STAKEHOLDER, SOLUTION, IMPACT, CHALLENGE
    target_id: str
    user_id: Optional[str] = None
    user_role: str = "CITIZEN"
    rating: int = 5
    comment: str
    issue_type: Optional[str] = None
    evidence_url: Optional[str] = None


class AppealCreate(BaseModel):
    problem_id: str
    appeal_type: str  # CLASSIFICATION, REJECTION, ROUTING, PROBLEM_UNDERSTANDING
    reason: str
    citizen_notes: Optional[str] = None
    user_id: Optional[str] = None


class AppealResolve(BaseModel):
    status: str  # ACCEPTED, REJECTED, RECLASSIFIED
    reviewer_notes: Optional[str] = None
    resolution_action: Optional[str] = None


class StakeholderReviewCreate(BaseModel):
    problem_id: str
    stakeholder_profile_id: Optional[str] = None
    reviewer_name: str
    action: str  # ACCEPT, WRONG_DOMAIN, REQUEST_MORE_INFO, CANNOT_HANDLE
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Collaboration, University & Industry Schemas (Step 6)
# ---------------------------------------------------------------------------

class UniversityProfileUpdate(BaseModel):
    institution_name: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    contact_email: Optional[str] = None
    website: Optional[str] = None
    areas_of_expertise: List[str] = Field(default_factory=list)
    faculty_leads: List[str] = Field(default_factory=list)
    labs_facilities: Optional[str] = None
    student_capabilities: Optional[str] = None


class IndustryProfileUpdate(BaseModel):
    company_name: Optional[str] = None
    industry_type: str = "Startup"
    sector: Optional[str] = None
    technical_capabilities: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    team_size: Optional[str] = None
    website: Optional[str] = None
    collaboration_interests: Optional[str] = None


class CollaborationRequestCreate(BaseModel):
    solution_id: str
    challenge_id: Optional[str] = None
    industry_id: str = "ind_ecoroads"
    company_name: str
    why_interested: str
    contribution: Optional[str] = None
    technical_capability: Optional[str] = None
    resources: Optional[str] = None
    commercialization_capability: Optional[str] = None
    pilot_capability: Optional[str] = None
    message: Optional[str] = None


class CollaborationResponse(BaseModel):
    action: str  # ACCEPT, DECLINE, REQUEST_MORE_INFO
    response_notes: Optional[str] = None


class CollaborationProgressUpdate(BaseModel):
    progress_percentage: int
    milestone: Optional[str] = None
    status: Optional[str] = None


class CollaborationTaskCreate(BaseModel):
    title: str
    assigned_to: str = "University"
    due_date: Optional[str] = None


class CollaborationMessageCreate(BaseModel):
    sender_role: str  # UNIVERSITY, INDUSTRY, ADMIN
    sender_name: str
    message: str
