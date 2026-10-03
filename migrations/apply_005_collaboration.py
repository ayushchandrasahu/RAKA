import sqlite3

def run_migration():
    conn = sqlite3.connect('raka.db')
    c = conn.cursor()

    # 1. university_profiles
    c.execute("""
    CREATE TABLE IF NOT EXISTS university_profiles (
        id TEXT PRIMARY KEY,
        institution_name TEXT NOT NULL,
        department TEXT NOT NULL,
        location TEXT,
        contact_email TEXT,
        website TEXT,
        areas_of_expertise TEXT DEFAULT '[]',
        faculty_leads TEXT DEFAULT '[]',
        labs_facilities TEXT,
        student_capabilities TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. industry_profiles
    c.execute("""
    CREATE TABLE IF NOT EXISTS industry_profiles (
        id TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        industry_type TEXT NOT NULL DEFAULT 'Startup',
        sector TEXT NOT NULL,
        technical_capabilities TEXT DEFAULT '[]',
        location TEXT,
        team_size TEXT,
        website TEXT,
        collaboration_interests TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 3. Add columns to solutions if missing
    c.execute("PRAGMA table_info(solutions)")
    cols = [col[1] for col in c.fetchall()]
    
    new_cols = [
        ("university_id", "TEXT"),
        ("department", "TEXT"),
        ("solution_type", "TEXT DEFAULT 'Research Proposal'"),
        ("proposed_approach", "TEXT"),
        ("technical_method", "TEXT"),
        ("prototype_required", "INTEGER DEFAULT 0"),
        ("estimated_timeline", "TEXT")
    ]
    for col_name, col_type in new_cols:
        if col_name not in cols:
            try:
                c.execute(f"ALTER TABLE solutions ADD COLUMN {col_name} {col_type};")
                print(f"Added column {col_name} to solutions")
            except Exception as e:
                print(f"Column {col_name} already exists or error: {e}")

    # 4. collaboration_requests
    c.execute("""
    CREATE TABLE IF NOT EXISTS collaboration_requests (
        id TEXT PRIMARY KEY,
        solution_id TEXT NOT NULL,
        challenge_id TEXT,
        industry_id TEXT NOT NULL,
        company_name TEXT NOT NULL,
        why_interested TEXT NOT NULL,
        contribution TEXT,
        technical_capability TEXT,
        resources TEXT,
        commercialization_capability TEXT,
        pilot_capability TEXT,
        message TEXT,
        status TEXT NOT NULL DEFAULT 'PENDING',
        university_response TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 5. collaborations
    c.execute("""
    CREATE TABLE IF NOT EXISTS collaborations (
        id TEXT PRIMARY KEY,
        request_id TEXT,
        solution_id TEXT NOT NULL,
        challenge_id TEXT,
        university_id TEXT,
        university_name TEXT NOT NULL,
        industry_id TEXT,
        company_name TEXT NOT NULL,
        shared_objectives TEXT,
        current_milestone TEXT DEFAULT 'Sprint 1: Joint Field Feasibility & Specs',
        progress_percentage INTEGER DEFAULT 15,
        status TEXT NOT NULL DEFAULT 'COLLABORATION_ACTIVE',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 6. collaboration_tasks
    c.execute("""
    CREATE TABLE IF NOT EXISTS collaboration_tasks (
        id TEXT PRIMARY KEY,
        collaboration_id TEXT NOT NULL,
        title TEXT NOT NULL,
        assigned_to TEXT DEFAULT 'University',
        status TEXT DEFAULT 'TODO',
        due_date TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 7. collaboration_messages
    c.execute("""
    CREATE TABLE IF NOT EXISTS collaboration_messages (
        id TEXT PRIMARY KEY,
        collaboration_id TEXT NOT NULL,
        sender_role TEXT NOT NULL,
        sender_name TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 8. notifications
    c.execute("""
    CREATE TABLE IF NOT EXISTS notifications (
        id TEXT PRIMARY KEY,
        recipient_role TEXT NOT NULL,
        recipient_id TEXT,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        link TEXT,
        is_read INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 9. Seed default high-quality University & Industry profiles
    c.execute("SELECT COUNT(*) FROM university_profiles")
    if c.fetchone()[0] == 0:
        c.execute("""
        INSERT INTO university_profiles (id, institution_name, department, location, contact_email, website, areas_of_expertise, faculty_leads, labs_facilities, student_capabilities)
        VALUES 
        ('univ_coep', 'COEP Technological University, Pune', 'Civil & Environmental Engineering', 'Shivajinagar, Pune, MH', 'research@coep.ac.in', 'https://coep.org.in', 
         '["Civil Engineering", "Hydrology & Drainage", "Environmental Engineering", "Materials Science"]', 
         '["Dr. A. Verma (Professor, Water Resources)", "Dr. S. Kulkarni (Head, Geotech Lab)"]', 
         'Advanced Fluid Mechanics Lab, Soil Mechanics Lab, Environmental Quality Testing Facility', 
         'Final-Year B.Tech & M.Tech thesis candidates ready for field instrumentation and rapid prototyping'),
        
        ('univ_iitb', 'IIT Bombay Urban Water Lab', 'Department of Civil Engineering & Centre for Tech Alternatives', 'Powai, Mumbai, MH', 'urbanwater@iitb.ac.in', 'https://iitb.ac.in', 
         '["Smart Infrastructure", "Urban Hydrology", "IoT Sensing", "Civil Engineering"]', 
         '["Prof. R. Deshmukh (Lead, Urban Infrastructure)", "Dr. P. Sen (IoT & Sensors)"]', 
         'National Geotechnical Center, Sensor Telemetry Testbed', 
         'Dual-degree research scholars specialized in municipal telemetry and eco-materials');
        """)
        print("Seeded default university profiles.")

    c.execute("SELECT COUNT(*) FROM industry_profiles")
    if c.fetchone()[0] == 0:
        c.execute("""
        INSERT INTO industry_profiles (id, company_name, industry_type, sector, technical_capabilities, location, team_size, website, collaboration_interests)
        VALUES 
        ('ind_ecoroads', 'EcoRoads InfraTech Innovations', 'Startup', 'Smart Infrastructure & CleanTech', 
         '["Cold-Mix Bio-Asphalt Paving", "Rapid Road Repair", "Field Pilot Deployment", "Pothole Telemetry"]', 
         'Dadar West, Mumbai, MH', '18 engineers & field crew', 'https://ecoroadsinfra.com', 
         'Seeking university partners for lab durability validation, moisture-resistance testing, and municipal pilot deployment'),
        
        ('ind_aquasense', 'AquaSense IoT Solutions', 'Startup', 'IoT & Civic Tech', 
         '["Ultrasonic Silt Depth Sensors", "Solar-Powered Telemetry", "Municipal GIS Integration", "Flash Flood Alerting"]', 
         'Hinjawadi Phase 1, Pune, MH', '12 IoT hardware & cloud devs', 'https://aquasense-iot.in', 
         'Interested in co-developing drainage sensor nodes with academic civil engineering labs for city-scale early warning');
        """)
        print("Seeded default industry profiles.")

    conn.commit()
    conn.close()
    print("Migration 005 completed successfully!")

if __name__ == "__main__":
    run_migration()
