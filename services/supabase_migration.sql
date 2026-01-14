-- ============================================================
-- AETHER Cloud Integration - Supabase Migration
-- ============================================================
-- Run this in Supabase SQL Editor to add tables needed for
-- AETHER local ↔ cloud sync.
--
-- This creates tables that mirror the local SQLite schema while
-- maintaining the existing Supabase structure.
-- ============================================================

-- Enable UUID extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 1. INSTALLATIONS (tracks AETHER installations)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.installations (
    installation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    name TEXT DEFAULT 'My AETHER',
    description TEXT,

    -- Hardware info
    hostname TEXT,
    platform TEXT,  -- 'raspberry_pi', 'linux', 'windows', 'macos'
    version TEXT,

    -- Network
    local_ip TEXT,

    -- Status
    is_online BOOLEAN DEFAULT false,
    last_seen_at TIMESTAMPTZ DEFAULT now(),

    -- Sync tracking
    last_sync_at TIMESTAMPTZ,
    sync_version INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for user lookups
CREATE INDEX IF NOT EXISTS idx_installations_user ON installations(user_id);

-- ============================================================
-- 2. SCHEDULES (cron-based automation)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.schedules (
    schedule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    installation_id UUID REFERENCES installations(installation_id) ON DELETE CASCADE,

    name TEXT NOT NULL,
    cron TEXT NOT NULL,  -- Cron expression

    -- Action to perform
    action_type TEXT NOT NULL,  -- 'play_look', 'play_sequence', 'play_scene', 'play_chase', 'blackout'
    action_id TEXT,  -- ID of the look/sequence/scene/chase to play
    action_params JSONB DEFAULT '{}',

    -- Status
    enabled BOOLEAN DEFAULT true,
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    run_count INTEGER DEFAULT 0,
    last_error TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for installation lookups
CREATE INDEX IF NOT EXISTS idx_schedules_installation ON schedules(installation_id);
CREATE INDEX IF NOT EXISTS idx_schedules_enabled ON schedules(enabled) WHERE enabled = true;

-- ============================================================
-- 3. GROUPS (fixture groups)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.groups (
    group_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    installation_id UUID REFERENCES installations(installation_id) ON DELETE CASCADE,

    name TEXT NOT NULL,
    universe INTEGER DEFAULT 1,
    channels JSONB,  -- Array of channel numbers or ranges
    color TEXT DEFAULT '#8b5cf6',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index for installation lookups
CREATE INDEX IF NOT EXISTS idx_groups_installation ON groups(installation_id);

-- ============================================================
-- 4. TIMERS (countdown timers)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.timers (
    timer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    installation_id UUID REFERENCES installations(installation_id) ON DELETE CASCADE,

    name TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,

    -- Action when timer completes
    action_type TEXT,
    action_id TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for installation lookups
CREATE INDEX IF NOT EXISTS idx_timers_installation ON timers(installation_id);

-- ============================================================
-- 5. RDM_DEVICES (discovered RDM fixtures)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.rdm_devices (
    rdm_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    installation_id UUID REFERENCES installations(installation_id) ON DELETE CASCADE,
    device_id UUID REFERENCES devices(device_id) ON DELETE CASCADE,

    uid TEXT NOT NULL,  -- RDM UID
    universe INTEGER,
    manufacturer_id INTEGER,
    device_model_id INTEGER,
    device_label TEXT,
    dmx_address INTEGER,
    dmx_footprint INTEGER,
    personality_id INTEGER,
    personality_count INTEGER,
    software_version TEXT,
    sensor_count INTEGER DEFAULT 0,

    last_seen TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for installation and device lookups
CREATE INDEX IF NOT EXISTS idx_rdm_devices_installation ON rdm_devices(installation_id);
CREATE INDEX IF NOT EXISTS idx_rdm_devices_device ON rdm_devices(device_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rdm_devices_uid ON rdm_devices(uid, installation_id);

-- ============================================================
-- 6. Add installation_id to existing tables (if needed)
-- ============================================================

-- Add installation_id to devices if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'devices' AND column_name = 'installation_id'
    ) THEN
        ALTER TABLE devices ADD COLUMN installation_id UUID REFERENCES installations(installation_id);
        CREATE INDEX idx_devices_installation ON devices(installation_id);
    END IF;
END $$;

-- ============================================================
-- 7. AETHER-specific scene_templates modifications
-- ============================================================
-- The existing scene_templates table can store looks, sequences, scenes, chases
-- using the 'mood' field as a type discriminator:
--   mood = 'look' → Look
--   mood = 'sequence' → Sequence
--   mood = 'scene' → Scene
--   mood = 'chase' → Chase
-- template_data contains the type-specific JSON data

-- Add installation tracking to scene_templates
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'scene_templates' AND column_name = 'installation_id'
    ) THEN
        ALTER TABLE scene_templates ADD COLUMN installation_id UUID REFERENCES installations(installation_id);
        CREATE INDEX idx_scene_templates_installation ON scene_templates(installation_id);
    END IF;
END $$;

-- ============================================================
-- 8. SYNC_LOG (tracks sync operations)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.sync_log (
    sync_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    installation_id UUID REFERENCES installations(installation_id) ON DELETE CASCADE,

    sync_type TEXT NOT NULL,  -- 'initial', 'incremental', 'full'
    direction TEXT NOT NULL,  -- 'push', 'pull'

    -- Stats
    records_synced INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,

    -- Status
    status TEXT DEFAULT 'in_progress',  -- 'in_progress', 'completed', 'failed'
    error_message TEXT,

    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sync_log_installation ON sync_log(installation_id);

-- ============================================================
-- 9. RLS Policies (Row Level Security)
-- ============================================================
-- Enable RLS on new tables

ALTER TABLE installations ENABLE ROW LEVEL SECURITY;
ALTER TABLE schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE timers ENABLE ROW LEVEL SECURITY;
ALTER TABLE rdm_devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE sync_log ENABLE ROW LEVEL SECURITY;

-- Policies for authenticated users to access their own data
-- (Adjust based on your auth requirements)

-- Installations: users can see their own installations
CREATE POLICY "Users can view own installations"
    ON installations FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own installations"
    ON installations FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own installations"
    ON installations FOR UPDATE
    USING (auth.uid() = user_id);

-- Service role bypass for backend operations
CREATE POLICY "Service role full access to installations"
    ON installations FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY "Service role full access to schedules"
    ON schedules FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY "Service role full access to groups"
    ON groups FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY "Service role full access to timers"
    ON timers FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY "Service role full access to rdm_devices"
    ON rdm_devices FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

CREATE POLICY "Service role full access to sync_log"
    ON sync_log FOR ALL
    USING (auth.jwt()->>'role' = 'service_role');

-- ============================================================
-- 10. Functions for updated_at triggers
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
DROP TRIGGER IF EXISTS update_installations_updated_at ON installations;
CREATE TRIGGER update_installations_updated_at
    BEFORE UPDATE ON installations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_schedules_updated_at ON schedules;
CREATE TRIGGER update_schedules_updated_at
    BEFORE UPDATE ON schedules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_groups_updated_at ON groups;
CREATE TRIGGER update_groups_updated_at
    BEFORE UPDATE ON groups
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- GRANT PERMISSIONS
-- ============================================================
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;

-- ============================================================
-- Success message
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE 'AETHER Cloud Integration migration complete!';
    RAISE NOTICE 'Tables created/updated: installations, schedules, groups, timers, rdm_devices, sync_log';
END $$;
