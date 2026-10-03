-- RAKA Domain AI - Step 5 Database Migration
-- Module A: Intelligent Challenges & Evidence
-- Module B: Context-Aware Multilingual RAKA AI Assistant Chatbot
-- Target: Supabase PostgreSQL with RLS

-- 1. Challenges Table
CREATE TABLE IF NOT EXISTS challenges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cluster_id UUID REFERENCES problem_clusters(id) ON DELETE SET NULL,
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
    challenge_status TEXT NOT NULL DEFAULT 'DRAFT', -- DRAFT, VALIDATION_REQUIRED, VALIDATED, READY_FOR_SOLUTIONS, PUBLISHED, IN_PROGRESS, RESOLVED, ARCHIVED
    readiness_score DOUBLE PRECISION DEFAULT 0.0,
    readiness_reasons JSONB DEFAULT '[]'::jsonb,
    missing_information JSONB DEFAULT '[]'::jsonb,
    warnings JSONB DEFAULT '[]'::jsonb,
    evidence_summary JSONB DEFAULT '[]'::jsonb,
    technical_disciplines JSONB DEFAULT '[]'::jsonb,
    required_skills JSONB DEFAULT '[]'::jsonb,
    stakeholder_types JSONB DEFAULT '[]'::jsonb,
    action_areas JSONB DEFAULT '[]'::jsonb,
    possible_solution_types JSONB DEFAULT '[]'::jsonb,
    expected_outcomes JSONB DEFAULT '[]'::jsonb,
    constraints JSONB DEFAULT '[]'::jsonb,
    success_indicators JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ
);

-- Indexes for Challenge Queries
CREATE INDEX IF NOT EXISTS idx_challenges_cluster ON challenges(cluster_id);
CREATE INDEX IF NOT EXISTS idx_challenges_status ON challenges(challenge_status);
CREATE INDEX IF NOT EXISTS idx_challenges_category ON challenges(category);
CREATE INDEX IF NOT EXISTS idx_challenges_urgency ON challenges(urgency);
CREATE INDEX IF NOT EXISTS idx_challenges_created_at ON challenges(created_at DESC);

-- 2. Challenge Evidence Table
CREATE TABLE IF NOT EXISTS challenge_evidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    challenge_id UUID NOT NULL REFERENCES challenges(id) ON DELETE CASCADE,
    problem_id UUID REFERENCES problems(id) ON DELETE SET NULL,
    cluster_id UUID REFERENCES problem_clusters(id) ON DELETE SET NULL,
    evidence_type TEXT NOT NULL, -- REPORTED_BY_CITIZEN, OBSERVED_IN_IMAGE, FROM_VOICE_TRANSCRIPT, SUPPORTED_BY_CLUSTER, AI_CLASSIFICATION, LOCATION_CONTEXT, STAKEHOLDER_MATCH, INFERRED_FROM_CONTEXT, NOT_AVAILABLE
    evidence_text TEXT NOT NULL,
    source_reference TEXT,
    confidence DOUBLE PRECISION DEFAULT 0.90,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_challenge_evidence_challenge ON challenge_evidence(challenge_id);
CREATE INDEX IF NOT EXISTS idx_challenge_evidence_type ON challenge_evidence(evidence_type);

-- 3. Chat Conversations Table
CREATE TABLE IF NOT EXISTS chat_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    problem_id UUID REFERENCES problems(id) ON DELETE SET NULL,
    cluster_id UUID REFERENCES problem_clusters(id) ON DELETE SET NULL,
    challenge_id UUID REFERENCES challenges(id) ON DELETE SET NULL,
    preferred_language TEXT DEFAULT 'en',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_conversations_problem ON chat_conversations(problem_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_cluster ON chat_conversations(cluster_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_challenge ON chat_conversations(challenge_id);

-- 4. Chat Messages Table
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL, -- USER, ASSISTANT, SYSTEM
    message TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    sources JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_conv ON chat_messages(conversation_id, created_at ASC);

-- 5. Row Level Security (RLS)
ALTER TABLE challenges ENABLE ROW LEVEL SECURITY;
ALTER TABLE challenge_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- Public can read published challenges or all for demo/judge inspection
CREATE POLICY "Public read challenges" ON challenges
    FOR SELECT USING (true);

CREATE POLICY "Public read challenge evidence" ON challenge_evidence
    FOR SELECT USING (true);

-- Authorized/anon users can create and interact with chat conversations
CREATE POLICY "Public read chat conversations" ON chat_conversations
    FOR SELECT USING (true);

CREATE POLICY "Public insert chat conversations" ON chat_conversations
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Public read chat messages" ON chat_messages
    FOR SELECT USING (true);

CREATE POLICY "Public insert chat messages" ON chat_messages
    FOR INSERT WITH CHECK (true);
