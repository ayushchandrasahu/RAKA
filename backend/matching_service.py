from typing import List, Dict, Any
from backend.schemas import Step4MatchingPayload, StakeholderRecommendation, ActionAreaRecommendation

# Canonical Stakeholder Registry
STAKEHOLDER_REGISTRY = {
    "Roads & Transport": {
        "government": [
            {"name": "Public Works Department (PWD)", "scope": "State & District Highway Maintenance", "relevance": "HIGH"},
            {"name": "Municipal Corporation Traffic Division", "scope": "Urban Streets & Junction Management", "relevance": "HIGH"},
            {"name": "Ministry of Road Transport & Highways (MoRTH)", "scope": "National Highway Network", "relevance": "MEDIUM"},
        ],
        "research": [
            {"name": "Central Road Research Institute (CRRI)", "scope": "Pavement Engineering & Material Diagnostics", "relevance": "HIGH"},
            {"name": "Civil Engineering Dept, Indian Institute of Technology", "scope": "Structural Durability & Drainage Studies", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "Pavement Condition & Depth Assessment", "type": "Assessment", "timeline": "24 - 48 Hours", "desc": "Field measurement of asphalt depression, base course erosion, and traffic risk."},
            {"title": "Traffic Calming & Warning Signage", "type": "Inspection", "timeline": "Immediate (6 Hours)", "desc": "Erection of safety barricades and warning reflectors around hazard zone."},
            {"title": "Cold/Warm Asphalt Patching Feasibility", "type": "Repair Feasibility", "timeline": "2 - 4 Days", "desc": "Structural road leveling and stormwater culvert clearance."},
        ]
    },
    "Water & Sanitation": {
        "government": [
            {"name": "Municipal Water Supply & Sewerage Board", "scope": "Urban Potable Water & Pipe Network", "relevance": "HIGH"},
            {"name": "Public Health Engineering Department (PHED)", "scope": "District Pipeline & Filtration Systems", "relevance": "HIGH"},
            {"name": "Ministry of Jal Shakti", "scope": "National Water Grid & Water Security", "relevance": "MEDIUM"},
        ],
        "research": [
            {"name": "National Environmental Engineering Research Institute (NEERI)", "scope": "Water Quality & Leak Detection Sensors", "relevance": "HIGH"},
            {"name": "Center for Water Resources & Hydrodynamics", "scope": "Acoustic Pressure Leak Diagnostics", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "Pipeline Acoustic Leak Isolation", "type": "Assessment", "timeline": "12 - 24 Hours", "desc": "Identify underground rupture points without destructive excavation."},
            {"title": "Water Contamination & Pressure Testing", "type": "Inspection", "timeline": "24 Hours", "desc": "Verify residual chlorine and microbiological safety of surrounding tap supplies."},
            {"title": "Pipeline Sleeve Repair or Replacement", "type": "Repair Feasibility", "timeline": "1 - 3 Days", "desc": "Trench isolation and pipe welding/gasket renewal."},
        ]
    },
    "Waste Management": {
        "government": [
            {"name": "Municipal Sanitation Department", "scope": "Daily Solid Waste Logistics & Route Management", "relevance": "HIGH"},
            {"name": "Solid Waste Management Cell", "scope": "Collection Infrastructure & Compaction Units", "relevance": "HIGH"},
            {"name": "State Pollution Control Board", "scope": "Environmental Health & Odor Auditing", "relevance": "MEDIUM"},
        ],
        "research": [
            {"name": "Center for Sustainable Waste Solutions", "scope": "Decentralized Biogas & Waste-to-Energy", "relevance": "HIGH"},
            {"name": "Environmental Engineering Research Group", "scope": "Smart RFID Bin Sensor Optimization", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "Urgent Dump Clearance & Disinfection", "type": "Resource Deployment", "timeline": "12 Hours", "desc": "Deploy compactors and lime disinfectant to eliminate public stench."},
            {"title": "Bin Capacity & Routing Audit", "type": "Assessment", "timeline": "2 - 3 Days", "desc": "Evaluate if pickup frequency matches generation volumes in market."},
            {"title": "Decentralized Community Composting", "type": "Repair Feasibility", "timeline": "1 - 2 Weeks", "desc": "Install aerobic compost drums for vegetable market waste."},
        ]
    },
    "Electricity & Energy": {
        "government": [
            {"name": "State Electricity Distribution Company (DISCOM)", "scope": "Local Grid, Transformers & Feeders", "relevance": "HIGH"},
            {"name": "Municipal Public Lighting Wing", "scope": "Streetlights & Junction High-Masts", "relevance": "HIGH"},
            {"name": "Electrical Inspectorate Division", "scope": "Safety Audits & Bare Wire Inspection", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "Central Power Research Institute (CPRI)", "scope": "Insulation Testing & Transformer Diagnostics", "relevance": "HIGH"},
            {"name": "Smart Grid & Renewable Energy Lab", "scope": "Automated Feeder Monitoring", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "Live Wire Ground De-energization & Shielding", "type": "Emergency Safety", "timeline": "Immediate (2 Hours)", "desc": "Insulate dangling cables and ground fault arresters."},
            {"title": "Illumination Lux Level Audit", "type": "Inspection", "timeline": "24 Hours", "desc": "Measure night brightness on pedestrian pathways."},
            {"title": "LED Driver & Cable Replacement", "type": "Repair Feasibility", "timeline": "1 - 2 Days", "desc": "Restore automatic timer switches and weatherproof enclosures."},
        ]
    },
    "Healthcare": {
        "government": [
            {"name": "Primary Health Centre (PHC) Administration", "scope": "Ward & Village Level Clinical Care", "relevance": "HIGH"},
            {"name": "District Chief Medical Officer (CMO)", "scope": "Medicine Inventory & Staff Deployment", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "Public Health Foundation Research Wing", "scope": "Essential Drug Supply Chain Optimization", "relevance": "HIGH"},
        ],
        "actions": [
            {"title": "Emergency Medicine Stock Replenishment", "type": "Resource Deployment", "timeline": "24 Hours", "desc": "Dispatch anti-rabies, antivenom, and critical antibiotics."},
            {"title": "Medical Staff Roster Audit", "type": "Assessment", "timeline": "2 Days", "desc": "Address doctor absenteeism and diagnostic lab functionality."},
        ]
    },
    "Education": {
        "government": [
            {"name": "District Education Office (Primary Education)", "scope": "School Infrastructure & Facilities", "relevance": "HIGH"},
            {"name": "Sarva Shiksha Abhiyan Engineering Wing", "scope": "Classroom & Boundary Wall Renovation", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "Institute of Educational Infrastructure Planning", "scope": "Child-Safe School Structural Standards", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "Structural Safety & Barrier Inspection", "type": "Inspection", "timeline": "24 Hours", "desc": "Assess wall collapse danger and child safety around premise."},
            {"title": "Infrastructure Rebuilding Feasibility", "type": "Repair Feasibility", "timeline": "1 - 2 Weeks", "desc": "Prepare masonry estimate for boundary wall and roof."},
        ]
    },
    "Agriculture": {
        "government": [
            {"name": "District Agriculture Office & Krishi Vigyan Kendra (KVK)", "scope": "Farmer Advisory, Seeds & Crop Support", "relevance": "HIGH"},
            {"name": "State Irrigation & Water Resources Department", "scope": "Canal Networks & Agricultural Drainage", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "Indian Council of Agricultural Research (ICAR)", "scope": "Agronomy, Pest Control & Water Use Efficiency", "relevance": "HIGH"},
        ],
        "actions": [
            {"title": "Irrigation Canal Breach & Flow Inspection", "type": "Inspection", "timeline": "24 Hours", "desc": "Assess water loss and channel embankment siltation."},
            {"title": "Emergency Borewell Pump Restoration", "type": "Resource Deployment", "timeline": "48 Hours", "desc": "Deploy motor rewinding and phase correction team."},
        ]
    },
    "Public Safety": {
        "government": [
            {"name": "Local Police Commissionerate / Station", "scope": "Law Enforcement & Public Hazard Response", "relevance": "HIGH"},
            {"name": "Municipal Disaster Management & Safety Cell", "scope": "Open Manholes, Collapsed Trees & Flooding", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "National Institute of Disaster Management (NIDM)", "scope": "Civic Risk Mapping & Safety Standards", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "Emergency Physical Hazard Barricading", "type": "Emergency Safety", "timeline": "Immediate (2 Hours)", "desc": "Cordon off dangerous open pits, loose slabs, or exposed cavities."},
            {"title": "Public Lighting & Surveillance Audit", "type": "Inspection", "timeline": "24 Hours", "desc": "Identify dark blindspots and security vulnerabilities."},
        ]
    },
    "Infrastructure": {
        "government": [
            {"name": "State Bridge & Building Construction Corporation", "scope": "Bridges, Flyovers & Public Structures", "relevance": "HIGH"},
            {"name": "Municipal Corporation Civil Works Wing", "scope": "Urban Retaining Walls & Storm Culverts", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "CSIR - Structural Engineering Research Centre (SERC)", "scope": "Non-Destructive Testing & Concrete Fatigue", "relevance": "HIGH"},
        ],
        "actions": [
            {"title": "Non-Destructive Structural Health Audit", "type": "Assessment", "timeline": "48 Hours", "desc": "Ultrasonic crack measurement and load-bearing inspection."},
            {"title": "Culvert Desilting & Structural Retrofit", "type": "Repair Feasibility", "timeline": "3 - 5 Days", "desc": "Restore hydraulic capacity and reinforce cracked abutments."},
        ]
    },
    "Environment": {
        "government": [
            {"name": "State Pollution Control Board (SPCB)", "scope": "Industrial Effluent, Air Quality & Noise Audits", "relevance": "HIGH"},
            {"name": "District Forest & Environment Department", "scope": "Tree Preservation & Eco-Sensitive Zones", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "Centre for Science and Environment (CSE)", "scope": "Pollution Source Apportionment & Clean Energy", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "Ambient Air & Water Pollution Sampling", "type": "Inspection", "timeline": "24 Hours", "desc": "Field monitoring of PM2.5, BOD, and toxic chemical levels."},
            {"title": "Illegal Waste Burning Containment", "type": "Resource Deployment", "timeline": "Immediate (6 Hours)", "desc": "Extinguish open burning and issue statutory abatement notice."},
        ]
    },
    "Accessibility": {
        "government": [
            {"name": "Department of Social Justice & Empowerment", "scope": "Barrier-Free Public Environment Standards", "relevance": "HIGH"},
            {"name": "Municipal Accessibility & Smart City Cell", "scope": "Footpaths, Crossings & Civic Buildings", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "National Institute for the Orthopaedically Handicapped", "scope": "Universal Design & Assistive Architecture", "relevance": "HIGH"},
        ],
        "actions": [
            {"title": "Universal Accessibility Audit", "type": "Assessment", "timeline": "48 Hours", "desc": "Measure curb ramp slope, tactile tiles, and doorway clearances."},
            {"title": "Tactile Paving & Ramp Installation", "type": "Repair Feasibility", "timeline": "1 - 2 Weeks", "desc": "Install anti-skid ramp and tactile guidance strips."},
        ]
    },
    "Housing": {
        "government": [
            {"name": "State Housing and Area Development Authority (MHADA/HUDCO)", "scope": "Affordable Housing & Structural Safety", "relevance": "HIGH"},
            {"name": "Slum Rehabilitation & Settlement Wing", "scope": "Civic Amenities in Informal Settlements", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "Central Building Research Institute (CBRI)", "scope": "Cost-Effective Earthquake & Storm Safe Masonry", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "Structural Safety & Waterlogging Audit", "type": "Inspection", "timeline": "24 Hours", "desc": "Check colony foundation settlement and drainage outfall."},
            {"title": "Emergency Tenement Roofing Feasibility", "type": "Repair Feasibility", "timeline": "3 - 5 Days", "desc": "Provide leakproof sheeting and storm anchoring."},
        ]
    },
    "Digital Services": {
        "government": [
            {"name": "National Informatics Centre (NIC) District Unit", "scope": "Government Portals & E-Governance Infrastructure", "relevance": "HIGH"},
            {"name": "CSC e-Governance Services India Ltd", "scope": "Gram Panchayat Kiosks & Citizen Entitlements", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "Digital India Civic Technology Laboratory", "scope": "Last-Mile Uptime & Biometric Accessibility", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "Kiosk Network & Biometric Diagnostic", "type": "Assessment", "timeline": "24 Hours", "desc": "Test optic fibre latency and UIDAI authentication ping."},
            {"title": "CSC Terminal Hardware Replacement", "type": "Resource Deployment", "timeline": "48 Hours", "desc": "Dispatch replacement fingerprint scanner and solar UPS."},
        ]
    },
    "Employment & Livelihood": {
        "government": [
            {"name": "District Employment & Skill Development Office", "scope": "Livelihood Training & Job Fairs", "relevance": "HIGH"},
            {"name": "MGNREGA District Programme Cell", "scope": "Rural Wage Employment & Job Cards", "relevance": "HIGH"},
        ],
        "research": [
            {"name": "National Institute of Rural Development & Panchayati Raj (NIRDPR)", "scope": "Livelihood Schemes & Social Audits", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "MGNREGA Wage Disbursal Audit", "type": "Assessment", "timeline": "48 Hours", "desc": "Reconcile pending muster rolls and bank transfer rejections."},
            {"title": "Skill Training Camp Organization", "type": "Resource Deployment", "timeline": "2 Weeks", "desc": "Schedule mobile PMKVY skill batch in the block."},
        ]
    },
    "Other": {
        "government": [
            {"name": "District Collectorate Public Grievance Cell", "scope": "Inter-departmental Coordination", "relevance": "HIGH"},
            {"name": "Municipal Corporation Public Works", "scope": "Civic Amenities & Maintenance", "relevance": "MEDIUM"},
        ],
        "research": [
            {"name": "State Technical University Civic Innovation Center", "scope": "Urban Engineering & Systems Diagnostics", "relevance": "MEDIUM"},
        ],
        "actions": [
            {"title": "On-Site Inter-Departmental Field Inspection", "type": "Inspection", "timeline": "24 - 48 Hours", "desc": "Joint inspection by local civic officer and technical experts."},
            {"title": "Root Cause & Remediation Assessment", "type": "Assessment", "timeline": "3 - 5 Days", "desc": "Identify systemic cause and formulate repair plan."},
        ]
    }
}

# Fallback generic domain template
GENERIC_DOMAIN_TEMPLATE = {
    "government": [
        {"name": "District Collectorate Public Grievance Cell", "scope": "Inter-departmental Coordination", "relevance": "HIGH"},
        {"name": "Municipal Corporation Public Works", "scope": "Civic Amenities & Maintenance", "relevance": "MEDIUM"},
    ],
    "research": [
        {"name": "State Technical University Civic Innovation Center", "scope": "Urban Engineering & Systems Diagnostics", "relevance": "MEDIUM"},
    ],
    "actions": [
        {"title": "On-Site Inter-Departmental Field Inspection", "type": "Inspection", "timeline": "24 - 48 Hours", "desc": "Joint inspection by local civic officer and technical experts."},
        {"title": "Root Cause & Remediation Assessment", "type": "Assessment", "timeline": "3 - 5 Days", "desc": "Identify systemic cause and formulate repair plan."},
    ]
}


def match_stakeholders_for_problem(
    category: str,
    sub_category: str,
    technical_disciplines: List[str],
    urgency: str,
    has_image: bool = False,
    cluster_member_count: int = 1,
    city: str = "",
) -> Step4MatchingPayload:
    """
    Step 4 Deterministic Rule-Based Stakeholder & Action Area Matching Engine.
    Uses category, sub-category, engineering disciplines, and cluster evidence.
    """
    domain_data = STAKEHOLDER_REGISTRY.get(category, GENERIC_DOMAIN_TEMPLATE)

    stakeholders = []
    # 1. Government Stakeholders
    for g in domain_data.get("government", []):
        why = [f"Matches problem domain ({category})"]
        if city:
            why.append(f"Jurisdiction covers {city}")
        if urgency in ("HIGH", "CRITICAL") and g["relevance"] == "HIGH":
            why.append("Urgent citizen safety escalation")
        
        score = 95 if g["relevance"] == "HIGH" else 80
        stakeholders.append(StakeholderRecommendation(
            stakeholder_type="Government / Public Authority",
            name=g["name"],
            jurisdiction_or_scope=g["scope"],
            relevance_level=g["relevance"],
            relevance_score=score,
            why_matched=why,
        ))

    # 2. Research & Innovation Stakeholders
    for r in domain_data.get("research", []):
        why = [f"Technical expertise in {', '.join(technical_disciplines[:2]) or 'Civic Engineering'}"]
        if cluster_member_count > 1:
            why.append(f"Cluster evidence shows recurring systemic issue ({cluster_member_count} reports)")
        if has_image:
            why.append("Physical visual evidence available for diagnosis")
            
        score = 88 if r["relevance"] == "HIGH" else 75
        stakeholders.append(StakeholderRecommendation(
            stakeholder_type="Research & Innovation Partner",
            name=r["name"],
            jurisdiction_or_scope=r["scope"],
            relevance_level=r["relevance"],
            relevance_score=score,
            why_matched=why,
        ))

    # 3. Action Area Recommendations
    action_areas = []
    for a in domain_data.get("actions", []):
        why_act = [f"Directly addresses {sub_category or category}"]
        if has_image:
            why_act.append("Supported by photographic evidence")
        action_areas.append(ActionAreaRecommendation(
            title=a["title"],
            action_type=a["type"],
            description=a["desc"],
            estimated_timeline=a["timeline"],
            why_recommended=why_act,
        ))

    # Global Why Recommendations
    why_global = [
        f"Domain classification matches {category}",
        f"Identified sub-issue: {sub_category}",
        f"Technical disciplines aligned with {', '.join(technical_disciplines[:2]) or 'Engineering'}",
    ]
    if cluster_member_count > 1:
        why_global.append(f"Problem cluster connects {cluster_member_count} related citizen reports")
    if has_image:
        why_global.append("Visual evidence corroborates reported physical condition")

    return Step4MatchingPayload(
        problem_title=f"{category}: {sub_category}",
        category=category,
        sub_category=sub_category,
        technical_disciplines=technical_disciplines,
        stakeholders=stakeholders,
        action_areas=action_areas,
        why_recommendations=why_global,
        disclaimer="These are AI-assisted stakeholder recommendations, not official department assignments."
    )
