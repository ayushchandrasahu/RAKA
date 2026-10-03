-- ============================================================================
-- Migration: 004_complete_platform_closed_loop.sql
-- Description: Complete Closed-Loop Intelligence Architecture
--              Solutions, Milestones, Implementation, Impact, Feedback, Appeals,
--              Stakeholder Reviews, and Audit Log
-- ============================================================================

-- 1. Solutions Table (Module 7)
CREATE TABLE IF NOT EXISTS solutions (
    id TEXT PRIMARY KEY,
    challenge_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    proposer_name TEXT NOT NULL,
    proposer_type TEXT NOT NULL DEFAULT 'INNOVATOR', -- STARTUP, UNIVERSITY, RESEARCHER, CITIZEN, NGO, INDUSTRY
    technical_disciplines TEXT DEFAULT '[]',
    required_resources TEXT DEFAULT '[]',
    expected_outcome TEXT,
    constraints TEXT,
    status TEXT NOT NULL DEFAULT 'PROPOSED', -- IDEA, PROPOSED, UNDER_REVIEW, ACCEPTED, PILOT, IMPLEMENTING, IMPLEMENTED, REJECTED, ARCHIVED
    review_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (challenge_id) REFERENCES challenges(id) ON DELETE CASCADE
);

-- 2. Solution Milestones (Module 8)
CREATE TABLE IF NOT EXISTS solution_milestones (
    id TEXT PRIMARY KEY,
    solution_id TEXT NOT NULL,
    milestone_title TEXT NOT NULL,
    description TEXT,
    target_date TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING, IN_PROGRESS, COMPLETED, BLOCKED
    completion_date TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (solution_id) REFERENCES solutions(id) ON DELETE CASCADE
);

-- 3. Implementation Updates (Module 8)
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

-- 4. Impact Records (Module 8 & 28)
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
    verification_status TEXT DEFAULT 'VERIFIED', -- UNVERIFIED, VERIFIED, DISPUTED
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (solution_id) REFERENCES solutions(id) ON DELETE CASCADE,
    FOREIGN KEY (challenge_id) REFERENCES challenges(id) ON DELETE SET NULL
);

-- 5. Feedback System (Module 9 & 29)
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    target_type TEXT NOT NULL, -- PROBLEM, STAKEHOLDER, SOLUTION, IMPACT, CHALLENGE
    target_id TEXT NOT NULL,
    user_id TEXT,
    user_role TEXT DEFAULT 'CITIZEN', -- CITIZEN, STAKEHOLDER, REVIEWER, JUDGE
    rating INTEGER, -- 1 to 5
    comment TEXT NOT NULL,
    issue_type TEXT, -- ACCURACY, ROUTING_MISMATCH, TIMELINE_DELAY, OTHER
    evidence_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Appeals & Correction System (Module 9 & 30)
CREATE TABLE IF NOT EXISTS appeals (
    id TEXT PRIMARY KEY,
    problem_id TEXT NOT NULL,
    user_id TEXT,
    appeal_type TEXT NOT NULL, -- CLASSIFICATION, REJECTION, ROUTING, PROBLEM_UNDERSTANDING
    reason TEXT NOT NULL,
    citizen_notes TEXT,
    status TEXT NOT NULL DEFAULT 'SUBMITTED', -- SUBMITTED, UNDER_REVIEW, ACCEPTED, REJECTED, RECLASSIFIED
    reviewer_notes TEXT,
    resolution_action TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
);

-- 7. Stakeholder Reviews Workflow (Module 5, 19 & 20)
CREATE TABLE IF NOT EXISTS stakeholder_reviews (
    id TEXT PRIMARY KEY,
    problem_id TEXT NOT NULL,
    stakeholder_profile_id TEXT,
    reviewer_name TEXT NOT NULL,
    action TEXT NOT NULL, -- ACCEPT, WRONG_DOMAIN, REQUEST_MORE_INFO, CANNOT_HANDLE
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
);

-- 8. Audit Trail (Module 32)
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL, -- PROBLEM, CLUSTER, CHALLENGE, SOLUTION, APPEAL, STAKEHOLDER
    entity_id TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT 'SYSTEM',
    event TEXT NOT NULL,
    previous_value TEXT,
    new_value TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 9. Problem Validations Table (Module 3 & 9)
CREATE TABLE IF NOT EXISTS problem_validations (
    id TEXT PRIMARY KEY,
    problem_id TEXT UNIQUE NOT NULL,
    validation_status TEXT NOT NULL DEFAULT 'VALID', -- VALID, NEEDS_REVIEW, INVALID
    reasons TEXT DEFAULT '[]',
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_solutions_challenge ON solutions(challenge_id);
CREATE INDEX IF NOT EXISTS idx_solutions_status ON solutions(status);
CREATE INDEX IF NOT EXISTS idx_milestones_solution ON solution_milestones(solution_id);
CREATE INDEX IF NOT EXISTS idx_impl_updates_solution ON implementation_updates(solution_id);
CREATE INDEX IF NOT EXISTS idx_impact_solution ON impact_records(solution_id);
CREATE INDEX IF NOT EXISTS idx_feedback_target ON feedback(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_appeals_problem ON appeals(problem_id);
CREATE INDEX IF NOT EXISTS idx_appeals_status ON appeals(status);
CREATE INDEX IF NOT EXISTS idx_stk_reviews_problem ON stakeholder_reviews(problem_id);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);
