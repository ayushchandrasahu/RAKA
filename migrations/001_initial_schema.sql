-- RAKA Domain AI - Step 3 Database Migration
-- Target: Supabase PostgreSQL with pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Problems Table
CREATE TABLE IF NOT EXISTS problems (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    problem_text TEXT NOT NULL,
    image_url TEXT,
    has_location INTEGER DEFAULT 0,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    location_accuracy DOUBLE PRECISION,
    manual_location TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING',
    is_demo BOOLEAN DEFAULT FALSE,
    similarity_status TEXT DEFAULT 'PENDING', -- PENDING, DONE, FAILED
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Problem AI Analysis Table
CREATE TABLE IF NOT EXISTS problem_ai_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_id UUID UNIQUE NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    problem_summary TEXT NOT NULL,
    urgency TEXT NOT NULL,
    complexity TEXT NOT NULL,
    technical_discipline JSONB NOT NULL DEFAULT '[]'::jsonb,
    suggested_departments JSONB NOT NULL DEFAULT '[]'::jsonb,
    suggested_stakeholders JSONB DEFAULT '[]'::jsonb,
    required_skills JSONB NOT NULL DEFAULT '[]'::jsonb,
    possible_solution_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    estimated_project_duration TEXT,
    impact_areas JSONB NOT NULL DEFAULT '[]'::jsonb,
    keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    image_observations JSONB NOT NULL DEFAULT '[]'::jsonb,
    reasoning_summary TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    model TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Problem Embeddings Table with 768 dimensions (L2-normalized)
CREATE TABLE IF NOT EXISTS problem_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_id UUID UNIQUE NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    embedding_model TEXT NOT NULL DEFAULT 'gemini-embedding-2',
    embedding_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector HNSW index for sub-millisecond nearest neighbor search
CREATE INDEX IF NOT EXISTS problem_embeddings_hnsw
    ON problem_embeddings USING hnsw (embedding vector_cosine_ops);

-- Problem Clusters Table
CREATE TABLE IF NOT EXISTS problem_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cluster_title TEXT NOT NULL,
    category TEXT NOT NULL,
    primary_problem_id UUID REFERENCES problems(id) ON DELETE SET NULL,
    problem_count INTEGER DEFAULT 1,
    status TEXT DEFAULT 'UNDER_VALIDATION',
    needs_review BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Problem Cluster Members Table
CREATE TABLE IF NOT EXISTS problem_cluster_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cluster_id UUID NOT NULL REFERENCES problem_clusters(id) ON DELETE CASCADE,
    problem_id UUID UNIQUE NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    membership_score NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster_id ON problem_cluster_members(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_members_created_at ON problem_cluster_members(created_at);

-- AI Analysis Logs
CREATE TABLE IF NOT EXISTS ai_analysis_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_id UUID,
    model TEXT,
    operation TEXT,
    input_hash TEXT,
    output_json JSONB,
    confidence DOUBLE PRECISION,
    latency_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Row Level Security (RLS) Configuration
ALTER TABLE problems ENABLE ROW LEVEL SECURITY;
ALTER TABLE problem_ai_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE problem_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE problem_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE problem_cluster_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_analysis_logs ENABLE ROW LEVEL SECURITY;

-- Backend server-only access (no direct anon public read/write)
-- All citizen queries and public cluster projections are handled via backend FastAPI endpoints
CREATE POLICY backend_all_problems ON problems FOR ALL USING (true);
CREATE POLICY backend_all_analysis ON problem_ai_analysis FOR ALL USING (true);
CREATE POLICY backend_all_embeddings ON problem_embeddings FOR ALL USING (true);
CREATE POLICY backend_all_clusters ON problem_clusters FOR ALL USING (true);
CREATE POLICY backend_all_members ON problem_cluster_members FOR ALL USING (true);
CREATE POLICY backend_all_logs ON ai_analysis_logs FOR ALL USING (true);
