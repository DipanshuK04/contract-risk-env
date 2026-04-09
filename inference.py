from typing import Optional, List
import os
import asyncio
import requests
import json
import sys
import textwrap

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ["MODEL_NAME"]

ENV_URL      = os.getenv("ENV_URL", "https://iamDipanshuK04-contract-env.hf.space")

BENCHMARK         = os.getenv("CONTRACT_RISK_BENCHMARK") or "contract-risk-env"
MAX_STEPS         = 25
SUCCESS_THRESHOLD = 0.5
TEMPERATURE       = 0.0
MAX_TOKENS        = 200

TASK_ALLOWED_ACTIONS = {
    "easy":   ["flag_missing", "approve", "submit_report"],
    "medium": ["flag_missing", "flag_risky", "approve", "submit_report"],
    "hard":   ["flag_risky", "submit_report"],
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are a contract risk analyst AI agent.
    You will be given contract data and must identify risks.
    For each step, respond with ONLY a valid JSON object — no markdown, no explanation, with exact keywords:
    {
      "action_type": "flag_missing",
      "target_clause": "ip_ownership",
      "severity": "high",
      "reason": "IP ownership clause is absent"
    }
    As per the task_id which denotes the difficulty level [low,medium,high], choose the valid action type only from the available options
    action_type options:
    - flag_missing  → clause is False/missing in the contract
    - flag_risky    → a value is dangerously low or conflicting
    - approve       → clause is present and value looks fine
    - submit_report → you are done analyzing, end the episode
""").strip()


def call_llm(messages: list) -> str:

    base = API_BASE_URL.rstrip("/")

    # LiteLLM / OpenAI-compatible proxy endpoint
    url = f"{base}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


def build_user_prompt(
    obs: dict,
    task_id: str,
    step: int,
    last_reward: float,
    history: List[str],
) -> str:

    already_acted = [
        a.get("target_clause", "")
        for a in obs.get("actions_taken", [])
    ]
    already_acted = [a for a in already_acted if a]

    history_block = "\n".join(history[-5:]) if history else "None"
    clauses = obs.get("clauses", {})
    values  = obs.get("values",  {})

    if task_id == "hard":
        a_vals = values.get("contract_a", {})
        b_vals = values.get("contract_b", {})
        a_clauses = clauses.get("contract_a", {})
        b_clauses = clauses.get("contract_b", {})

        all_fields = set(a_vals.keys()) | set(b_vals.keys())

        conflicts_found = []
        comparison_lines = []
        for field in sorted(all_fields):
            val_a = a_vals.get(field, "MISSING")
            val_b = b_vals.get(field, "MISSING")
            is_conflict = val_a != val_b
            if is_conflict:
                conflicts_found.append(field)
                comparison_lines.append(
                    f"  {field}: A={val_a}  vs  B={val_b}  ← CONFLICT"
                )
            else:
                comparison_lines.append(
                    f"  {field}: A={val_a}  ==  B={val_b}"
                )

        comparison_str = "\n".join(comparison_lines)
        remaining_conflicts = [f for f in conflicts_found if f not in already_acted]
        should_submit = len(remaining_conflicts) == 0

        return textwrap.dedent(f"""
            TASK: {task_id} (CONFLICT DETECTION)
            CONTRACT ID: {obs['contract_id']}
            ── CONTRACT A CLAUSES ──
            {json.dumps(a_clauses, indent=2)}
            ── CONTRACT B CLAUSES ──
            {json.dumps(b_clauses, indent=2)}
            ── FIELD BY FIELD COMPARISON ──
            {comparison_str}
            ── CONFLICT SUMMARY ──
            All conflicting fields:     {conflicts_found}
            Already flagged:            {already_acted}
            REMAINING TO FLAG:          {remaining_conflicts}
            ── EPISODE STATE ──
            STEP: {step}
            STEPS REMAINING: {obs['remaining_budget']}
            LAST REWARD: {last_reward:.2f}
            ALLOWED ACTIONS: {TASK_ALLOWED_ACTIONS[task_id]}
            ── HISTORY ──
            {history_block}
            ── RULES ──
            1. Use flag_risky for each field in REMAINING TO FLAG
            2. target_clause MUST be an exact field name from comparison above
            3. NEVER use contract_a or contract_b as target_clause
            4. NEVER repeat a clause from Already flagged
            5. {'⚠️ REMAINING IS EMPTY → use submit_report NOW' if should_submit else 'Flag the remaining conflicts above'}
            Respond ONLY with valid JSON, no markdown:
            {{
              "action_type": "flag_risky",
              "target_clause": "exact_field_name",
              "severity": "high",
              "reason": "A has X but B has Y"
            }}
        """).strip()

    missing_clauses = [k for k, v in clauses.items() if v is False]
    present_clauses = [k for k, v in clauses.items() if v is True]
    value_keys      = list(values.keys())

    suspicious_values = {}
    for k, v in values.items():
        if isinstance(v, (int, float)):
            if "usd" in k.lower() and v < 10000:
                suspicious_values[k] = f"{v} ← very low amount"
            elif "days" in k.lower() and v < 7:
                suspicious_values[k] = f"{v} ← very short period"
            elif "years" in k.lower() and v < 2:
                suspicious_values[k] = f"{v} ← very short duration"
            elif "percent" in k.lower() and v < 90:
                suspicious_values[k] = f"{v} ← below standard threshold"
            else:
                suspicious_values[k] = f"{v}"
        else:
            suspicious_values[k] = f"{v}"

    if task_id == "easy":
        remaining_missing = [c for c in missing_clauses if c not in already_acted]
        remaining_risky   = []
        should_submit     = len(remaining_missing) == 0
    else:  # medium
        remaining_missing = [c for c in missing_clauses if c not in already_acted]
        remaining_risky   = [v for v in value_keys      if v not in already_acted]
        should_submit     = (len(remaining_missing) == 0 and len(remaining_risky) == 0)

    return textwrap.dedent(f"""
        TASK: {task_id}
        CONTRACT ID: {obs['contract_id']}
        CONTRACT TYPE: {obs['contract_type']}
        ── CLAUSES (true=present, false=missing) ──
        {json.dumps(clauses, indent=2)}
        ── VALUES ──
        {json.dumps(values, indent=2)}
        ── WHAT TO FLAG ──
        Missing clauses (flag_missing targets): {missing_clauses}
        Risky value keys (flag_risky targets):  {value_keys if task_id == "medium" else "NOT graded in easy task"}
        Present clauses (approve targets):      {present_clauses}
        ── REMAINING (still need to flag these) ──
        Missing clauses left: {remaining_missing}
        Risky values left:    {remaining_risky}
        ── EPISODE STATE ──
        STEP: {step}
        STEPS REMAINING: {obs['remaining_budget']}
        LAST REWARD: {last_reward:.2f}
        ALREADY ACTED ON — never repeat: {already_acted}
        ALLOWED ACTIONS: {TASK_ALLOWED_ACTIONS[task_id]}
        ── HISTORY ──
        {history_block}
        ── RULES ──
        1. target_clause MUST be exact key from CLAUSES or VALUES above
        2. flag_missing  → pick ONLY from remaining missing: {remaining_missing}
        3. flag_risky    → pick ONLY from remaining risky:   {remaining_risky}
        4. approve       → pick from present clauses:        {present_clauses}
        5. NEVER repeat anything from ALREADY ACTED ON
        6. NEVER invent clause names
        7. {'NOTHING LEFT TO FLAG → use submit_report NOW' if should_submit else 'Flag remaining items above'}
        ── VALUES WITH RISK ANALYSIS ──
        {json.dumps(suspicious_values, indent=2)}
        Respond ONLY with valid JSON, no markdown:
        {{
          "action_type": "flag_missing",
          "target_clause": "exact_key_from_above",
          "severity": "high",
          "reason": "brief reason"
        }}
    """).strip()


def get_llm_action(
    obs: dict,
    task_id: str,
    step: int,
    last_reward: float,
    history: List[str],
) -> dict:

    if obs.get("remaining_budget", 0) <= 0:
        return {
            "action_type":   "submit_report",
            "target_clause": "none",
            "severity":      None,
            "reason":        "budget exhausted",
        }

    user_prompt = build_user_prompt(obs, task_id, step, last_reward, history)

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        raw = call_llm(messages).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)

        if action.get("action_type") == "submit_report":
            action["target_clause"] = "none"

        if action.get("severity") in [None, "null", "none", ""]:
            action["severity"] = None

        return {
            "action_type":   action.get("action_type",  "submit_report"),
            "target_clause": action.get("target_clause", "none"),
            "severity":      action.get("severity",None),
            "reason":        action.get("reason",""),
        }

    except Exception as e:
        return {
            "action_type":   "submit_report",
            "target_clause": "none",
            "severity":      None,
            "reason":        f"llm_error: {str(e)}",
        }


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)
    sys.stdout.flush()


# def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
#     error_val = error if error else "null"
#     done_val  = str(done).lower()
#     print(
#         f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
#         flush=True,
#     )
#     sys.stdout.flush()
def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    safe_reward = min(max(float(reward), 0.01), 0.99)
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={safe_reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

# def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
#     rewards_str = ",".join(f"{r:.2f}" for r in rewards)
#     print(
#         f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
#         flush=True,
#     )
#     sys.stdout.flush()
def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Clamp score one final time before logging
    safe_score = min(max(float(score), 0.01), 0.99)
    rewards_str = ",".join(f"{min(max(r, 0.01), 0.99):.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={safe_score:.2f} rewards={rewards_str}",
        flush=True,
    )
    # print(
    #     f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
    #     flush=True,
    # )

async def run_task(task_id: str) -> float:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.01
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            try:
                action = get_llm_action(obs, task_id, step, last_reward, history)
                if not action or "action_type" not in action or "target_clause" not in action:
                    raise ValueError("Invalid LLM action format")
                action_str = f"{action['action_type']}('{action['target_clause']}')"
            except Exception:
                action     = {"action_type": "submit_report", "target_clause": "none"}
                action_str = "submit_report('none')"

            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json=action,
                    timeout=30,
                )
                step_resp.raise_for_status()
                result = step_resp.json()
                obs = result["observation"]
                reward    = float(result["reward"])
                done = bool(result["done"])
                error_msg = None
            except Exception as e:
                reward = 0.0
                done = True
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            history.append(f"Step {step}: {action_str} → reward={reward:+.2f}")

            if done:
                break

        try:
            grade_resp = requests.post(f"{ENV_URL}/grade", timeout=30)
            grade_resp.raise_for_status()
            score = max(float(grade_resp.json()["score"]),0.01)
        except Exception:
            score = 0.01

        score   = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_THRESHOLD

    except Exception:
        score = 0.01
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    tasks= ["easy", "medium", "hard"]
    all_scores = []
    for task in tasks:
        score = await run_task(task)
        all_scores.append(score)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        print("[START] task=boot env=contract-risk-env model=boot", flush=True)
        import traceback
        traceback.print_exc()
        log_end(success=False, steps=0, score=0.01, rewards=[])
