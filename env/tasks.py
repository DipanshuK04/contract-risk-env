TASK_CONFIG = {
    "easy": {
        "description": "Detect missing clauses in a simple NDA",
        "max_steps": 10,
        "difficulty": "easy",
        "contract_prefix": "nda",     
        "contract_count": 5,
        "allowed_actions": ["flag_missing", "approve", "submit_report"],  
        "graded_on": ["missing_clauses"]            
    },
    "medium": {
        "description": "Flag missing clauses AND risky values in a vendor agreement",
        "max_steps": 20,
        "difficulty": "medium",
        "contract_prefix": "vendor",
        "contract_count": 5,
        "allowed_actions": ["flag_missing", "flag_risky", "approve", "submit_report"],
        "graded_on": ["missing_clauses", "risky_values"]
    },
    "hard": {
        "description": "Detect conflicts between two contracts",
        "max_steps": 30,
        "difficulty": "hard",
        "contract_prefix": "conflict",
        "contract_count": 5,
        "allowed_actions": ["flag_risky", "submit_report"],
        "graded_on": ["conflicts"]

    }
}
