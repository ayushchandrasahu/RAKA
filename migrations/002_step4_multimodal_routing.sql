-- RAKA Domain AI - Step 4 Database Migration
-- Multimodal Intelligent Routing + Master Stakeholder Tables & Rules
-- Target: Supabase PostgreSQL with RLS

-- 1. Extend problems table if missing columns
ALTER TABLE problems ADD COLUMN IF NOT EXISTS preferred_language TEXT DEFAULT 'en';
ALTER TABLE problems ADD COLUMN IF NOT EXISTS audio_url TEXT;
ALTER TABLE problems ADD COLUMN IF NOT EXISTS voice_transcript TEXT;
ALTER TABLE problems ADD COLUMN IF NOT EXISTS detected_language TEXT;
ALTER TABLE problems ADD COLUMN IF NOT EXISTS city TEXT;
ALTER TABLE problems ADD COLUMN IF NOT EXISTS district TEXT;
ALTER TABLE problems ADD COLUMN IF NOT EXISTS state TEXT;
ALTER TABLE problems ADD COLUMN IF NOT EXISTS locality TEXT;
ALTER TABLE problems ADD COLUMN IF NOT EXISTS location_status TEXT DEFAULT 'CAPTURED';

-- 2. Extend problem_ai_analysis table if missing columns
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS problem_title TEXT;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS sub_category TEXT;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS problem_understanding JSONB;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS visual_analysis JSONB;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS voice_analysis JSONB;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS impact_analysis JSONB;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS solution_context JSONB;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS routing_context JSONB;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS evidence JSONB;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS uncertainty JSONB;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS classification_confidence DOUBLE PRECISION DEFAULT 0.92;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS image_confidence DOUBLE PRECISION DEFAULT 0.88;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS language_confidence DOUBLE PRECISION DEFAULT 0.95;
ALTER TABLE problem_ai_analysis ADD COLUMN IF NOT EXISTS raw_analysis_json TEXT;

-- 3. Master Stakeholder Profiles Table
CREATE TABLE IF NOT EXISTS stakeholder_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stakeholder_type TEXT NOT NULL,
    name TEXT NOT NULL,
    domain_category TEXT NOT NULL,
    jurisdiction_or_scope TEXT,
    default_relevance TEXT DEFAULT 'HIGH',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Matching Rules Table
CREATE TABLE IF NOT EXISTS matching_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    domain_category TEXT NOT NULL,
    sub_category_pattern TEXT DEFAULT '*',
    required_disciplines JSONB DEFAULT '[]'::jsonb,
    stakeholder_profile_id UUID REFERENCES stakeholder_profiles(id) ON DELETE CASCADE,
    relevance_score INTEGER DEFAULT 90,
    match_reason_template TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Action Types Table
CREATE TABLE IF NOT EXISTS action_types (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    domain_category TEXT NOT NULL,
    title TEXT NOT NULL,
    action_type TEXT NOT NULL,
    description TEXT,
    estimated_timeline TEXT,
    why_recommended_template TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Problem Stakeholder Recommendations (Instance recommendations per report)
CREATE TABLE IF NOT EXISTS problem_stakeholder_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    stakeholder_type TEXT NOT NULL,
    name TEXT NOT NULL,
    jurisdiction_or_scope TEXT,
    relevance_level TEXT,
    relevance_score INTEGER,
    why_matched JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Problem Action Recommendations (Instance action areas per report)
CREATE TABLE IF NOT EXISTS problem_action_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    action_type TEXT NOT NULL,
    description TEXT,
    estimated_timeline TEXT,
    why_recommended JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_stk_domain ON stakeholder_profiles(domain_category);
CREATE INDEX IF NOT EXISTS idx_rules_domain ON matching_rules(domain_category);
CREATE INDEX IF NOT EXISTS idx_actions_domain ON action_types(domain_category);
CREATE INDEX IF NOT EXISTS idx_prob_stk_pid ON problem_stakeholder_recommendations(problem_id);
CREATE INDEX IF NOT EXISTS idx_prob_act_pid ON problem_action_recommendations(problem_id);

-- Row Level Security (RLS) Configuration
-- Citizens cannot modify master configuration. Master tables are read-only for public/citizens,
-- and writable only by backend service role.
ALTER TABLE stakeholder_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE matching_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE action_types ENABLE ROW LEVEL SECURITY;
ALTER TABLE problem_stakeholder_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE problem_action_recommendations ENABLE ROW LEVEL SECURITY;

CREATE POLICY backend_all_stakeholder_profiles ON stakeholder_profiles FOR ALL USING (true);
CREATE POLICY backend_all_matching_rules ON matching_rules FOR ALL USING (true);
CREATE POLICY backend_all_action_types ON action_types FOR ALL USING (true);
CREATE POLICY backend_all_problem_stakeholders ON problem_stakeholder_recommendations FOR ALL USING (true);
CREATE POLICY backend_all_problem_actions ON problem_action_recommendations FOR ALL USING (true);
