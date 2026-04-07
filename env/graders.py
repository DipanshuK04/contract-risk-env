def grade_easy(env) -> float:
    """
    Easy: ONLY checks missing clauses (not risky values)
    Score = F1 on missing clause detection
    """
    gt = env.current_contract["ground_truth"]
    true_missing = set(gt["missing_clauses"])

    agent_missing = {
        a.get("target_clause", "") for a in env.actions_taken
        if a.get("action_type") == "flag_missing" and a["reward"] > 0
    }

    if not true_missing:
        false_positives = len(agent_missing)
        score = max(1.0 - false_positives * 0.2, 0.0)
        return round(score, 4)

    tp = len(true_missing & agent_missing)           # corrctly flagged...
    fp = len(agent_missing - true_missing)           # wrogly flagged..
    fn = len(true_missing - agent_missing)           # missed...

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)

def grade_medium(env) -> float:
    """
    Medium: checks BOTH missing_clauses AND risky_values...
    Score = combined F1 on both..
    """
    gt = env.current_contract["ground_truth"]

    true_missing = set(gt["missing_clauses"])
    true_risky   = set(gt["risky_values"])

    agent_missing = {
        a.get("target_clause", "") for a in env.actions_taken
        if a.get("action_type") == "flag_missing" and a.get("reward", 0) > 0
    }
    agent_risky = {
        a.get("target_clause", "") for a in env.actions_taken
        if a.get("action_type") == "flag_risky" and a.get("reward", 0) > 0
    }
    def f1_score(true_set, agent_set):
        if not true_set:
            fp = len(agent_set)
            return max(1.0 - fp * 0.2, 0.0)
        tp = len(true_set & agent_set)
        fp = len(agent_set - true_set)
        fn = len(true_set - agent_set)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    missing_score = f1_score(true_missing, agent_missing)
    risky_score   = f1_score(true_risky, agent_risky)

    final = (missing_score + risky_score) / 2
    return round(final, 4)

def grade_hard(env) -> float:
    """
    Hard: conflict detection between two contracts
    Score = F1 on conflict fields..
    """
    gt = env.current_contract["ground_truth"]
    true_conflicts = set(c["field"] for c in gt["conflicts"])

    agent_flagged = {
        a.get("target_clause", "") for a in env.actions_taken
        if  a.get("action_type") == "flag_risky" and a.get("reward", 0) > 0
    }

    if not true_conflicts:
        fp = len(agent_flagged)
        return round(max(1.0 - fp * 0.2, 0.0), 4)

    tp = len(true_conflicts & agent_flagged)
    fp = len(agent_flagged - true_conflicts)
    fn = len(true_conflicts - agent_flagged)

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

def run_grader(task_id:str,env)->float:
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return GRADERS[task_id](env)
