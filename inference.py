from typing import Optional,List
import os
import asyncio
from openai import OpenAI
import requests
import json
import sys
import textwrap

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("HF_TOKEN or API_KEY not set,please set when running docker file")

HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "https://iamDipanshuK04-contract-risk-env.hf.space")

BENCHMARK  = os.getenv("CONTRACT_RISK_BENCHMARK") or "contract-risk-env"
MAX_STEPS  = 25
SUCCESS_THRESHOLD = 0.5   
TEMPERATURE       = 0.0    # deterministic 
MAX_TOKENS        = 200

TASK_ALLOWED_ACTIONS = {
    "easy" : ["flag_missing", "approve", "submit_report"],  
    "medium": ["flag_missing", "flag_risky", "approve", "submit_report"],
    "hard":   ["flag_risky", "submit_report"]
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are a contract risk analyst AI agent.
    You will be given contract data and must identify risks.
    
    For each step, respond with ONLY a valid JSON object — no markdown, no explanation,with exact keywords:
    {
      "action_type": "flag_missing",
      "target_clause": "ip_ownership",
      "severity": "high",
      "reason": "IP ownership clause is absent"
    }
    As per the task_id which denotes the difficulty level [low,medium,high] , choose the valid actions type only from the available options
    action_type options:
    - flag_missing  → clause is False/missing in the contract
    - flag_risky    → a value is dangerously low or conflicting
    - approve       → clause is present and value looks fine
    - submit_report → you are done analyzing, end the episode
""").strip()

# def build_user_prompt(obs:dict,task_id: str,step: int,last_reward:float,history: List[str])->str:
#     already_acted = [a["target_clause"] for a in obs.get("actions_taken", [])]
#     history_block = "\n".join(history[-5:]) if history else "None"
#     return textwrap.dedent(
#         f"""
#         TASK: {task_id},
#         CONTRACT ID: {obs['contract_id']}
#         CONTRACT TYPE: {obs['contract_type']}


#         STEP: {step}
#         STEPS REMAINING: {obs['remaining_budget']}
#         LAST REWARD: {last_reward:.2f}
#         ALREADY ACTED ON (do NOT repeat these): {already_acted}

#         PREVIOUS STEPS:
#         {history_block}

#         Choose your next action from the available actions for task_id. 
#         ALLOWED ACTIONS: {TASK_ALLOWED_ACTIONS[task_id]}

#         Remember:
#         - flag_missing → only for clauses that are False
#         - flag_risky   → only for values that are dangerously low/conflicting
#         - submit_report → when you have flagged everything or steps are low
#         - NEVER act on a clause already in ALREADY ACTED ON list
#         """
#     ).strip()

# def build_user_prompt(
#     obs: dict,
#     task_id: str,
#     step: int,
#     last_reward: float,
#     history: List[str]
# ) -> str:
#     already_acted = [a.get("target_clause", "") for a in obs.get("actions_taken", [])]
    
#     already_acted = [a for a in already_acted if a]

#     history_block = "\n".join(history[-5:]) if history else "None"

#     # ── Extract exact keys from contract ──────────────────────
#     clauses = obs.get("clauses", {})
#     values  = obs.get("values",  {})

#     # Tell LLM exactly which are missing vs present
#     missing_clauses = [k for k, v in clauses.items() if v == False]
#     present_clauses = [k for k, v in clauses.items() if v == True]
#     value_keys      = list(values.keys())

#     remaining_missing = [c for c in missing_clauses if c not in already_acted]

#     return textwrap.dedent(f"""
#         TASK: {task_id}
#         CONTRACT ID: {obs['contract_id']}
#         CONTRACT TYPE: {obs['contract_type']}

#         ── CLAUSES (true=present, false=missing) ──
#         {json.dumps(clauses, indent=2)}

#         ── VALUES ──
#         {json.dumps(values, indent=2)}

#         ── WHAT TO FLAG ──
#         Missing clauses (flag_missing targets): {missing_clauses}
#         Risky value keys (flag_risky targets):  {value_keys}
#         Present clauses (approve targets):      {present_clauses}

#         ── EPISODE STATE ──
#         STEP: {step}
#         STEPS REMAINING: {obs['remaining_budget']}
#         LAST REWARD: {last_reward:.2f}
#         ALREADY ACTED ON — never repeat: {already_acted}
#         ALLOWED ACTIONS: {TASK_ALLOWED_ACTIONS[task_id]}

#         ── HISTORY ──
#         {history_block}

#         ── RULES ──
#         1. target_clause MUST be an exact key from CLAUSES or VALUES above
#         2. flag_missing  → pick from: {missing_clauses}
#         3. flag_risky    → pick from: {value_keys}
#         4. approve       → pick from: {present_clauses}
#         5. submit_report → use when all issues flagged or steps low
#         6. NEVER invent clause names
#         7. NEVER repeat a clause from ALREADY ACTED ON

#         Respond ONLY with valid JSON, no markdown, no explanation:
#         {{
#           "action_type": "flag_missing",
#           "target_clause": "exact_key_from_above",
#           "severity": "high",
#           "reason": "brief reason"
#         }}
#     """).strip()
def build_user_prompt(
    obs: dict,
    task_id: str,
    step: int,
    last_reward: float,
    history: List[str]
) -> str:

    # ── Already acted — safe extraction ───────────────────────
    # obs["actions_taken"] is a list of dicts like:
    # [{"action_type": "flag_missing", "target_clause": "ip_ownership", "reward": 1.0}]
    # Use .get() to avoid KeyError if structure is unexpected
    already_acted = [
        a.get("target_clause", "")
        for a in obs.get("actions_taken", [])
    ]
    # Remove empty strings just in case
    already_acted = [a for a in already_acted if a]

    history_block = "\n".join(history[-5:]) if history else "None"

    clauses = obs.get("clauses", {})
    values  = obs.get("values",  {})

    # ══════════════════════════════════════════════════════════
    # HARD TASK — special case, handle separately
    # clauses = {"contract_a": {...}, "contract_b": {...}}
    # values  = {"contract_a": {...}, "contract_b": {...}}
    # LLM must find fields that DIFFER between A and B
    # ══════════════════════════════════════════════════════════
    if task_id == "hard":

        # Extract A and B values
        a_vals = values.get("contract_a", {})
        b_vals = values.get("contract_b", {})
        a_clauses = clauses.get("contract_a", {})
        b_clauses = clauses.get("contract_b", {})

        # Find ALL fields that appear in either contract
        all_fields = set(a_vals.keys()) | set(b_vals.keys())

        # Build field-by-field comparison
        # Only show fields where values DIFFER = conflicts
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

        # What's still left to flag (not yet acted on)
        remaining_conflicts = [
            f for f in conflicts_found
            if f not in already_acted
        ]

        # Should agent stop?
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


    # Which clauses are missing (False) vs present (True)
    missing_clauses = [k for k, v in clauses.items() if v is False]
    present_clauses = [k for k, v in clauses.items() if v is True]
    value_keys      = list(values.keys())
    # In build_user_prompt, for easy/medium:
    # Add logic to identify suspicious values
    suspicious_values = {}
    for k, v in values.items():
        if isinstance(v, (int, float)):
            # Flag suspiciously low numbers
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
    # What's still remaining based on task
    if task_id == "easy":
        # Easy: only flag missing clauses, ignore values
        remaining_missing = [c for c in missing_clauses if c not in already_acted]
        remaining_risky   = []    # not graded in easy
        should_submit     = len(remaining_missing) == 0

    elif task_id == "medium":
        # Medium: flag both missing clauses AND risky values
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
        7. {'⚠️ NOTHING LEFT TO FLAG → use submit_report NOW' if should_submit else f'Flag remaining items above'}

        f"── VALUES WITH RISK ANALYSIS ──\n{json.dumps(suspicious_values, indent=2)}"
        
        Respond ONLY with valid JSON, no markdown:
        {{
          "action_type": "flag_missing",
          "target_clause": "exact_key_from_above",
          "severity": "high",
          "reason": "brief reason"
        }}
    """).strip()

def get_llm_action(
        client: OpenAI,
        obs: dict,
        task_id: str,
        step: int,
        last_reward: float,
        history: List[str]
    ) -> dict:

    if obs.get("remaining_budget", 0) <= 1:
        return {
            "action_type": "submit_report",
            "target_clause": "none",
            "severity": None,
            "reason": "budget exhausted"
        }
    user_prompt = build_user_prompt(obs, task_id, step, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)

        # if action.get("action_type") == "submit_report":
        #     action["target_clause"] = "none"
        if action.get("action_type") == "submit_report":
            action["target_clause"] = "none"

        if action.get("severity") in [None, "null", "none", ""]:
            action["severity"] = None
        clean_action = {
            "action_type":   action.get("action_type", "submit_report"),
            "target_clause": action.get("target_clause", "none"),
            "severity":      action.get("severity", None),
            "reason":        action.get("reason", "")
        }
        return clean_action
        # return action

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {
            "action_type": "submit_report",
            "target_clause": "none",
            "severity": None,
            "reason": f"llm_error: {str(e)}"
        }


    
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )
def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run_task(client: OpenAI, task_id: str)->float:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()
        last_reward = 0.0
        history: List[str] = []
        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            # LLM decides action
            action = get_llm_action(client, obs, task_id, step, last_reward, history)
            action_str = f"{action['action_type']}('{action['target_clause']}')"

            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json=action,
                    timeout=30
                )
                step_resp.raise_for_status()
                result    = step_resp.json()
                obs       = result["observation"]
                reward    = float(result["reward"])
                done      = bool(result["done"])
                error_msg = None
            except Exception as e:
                reward    = 0.0
                done      = True
                error_msg = str(e)
                print(f"[DEBUG] Step error: {e}", flush=True)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg
            )

            history.append(
                f"Step {step}: {action_str} → reward={reward:+.2f}"
            )

            if done:
                break

        try:
            grade_resp = requests.post(f"{ENV_URL}/grade", timeout=30)
            grade_resp.raise_for_status()
            score = float(grade_resp.json()["score"])
        except Exception as e:
            print(f"[DEBUG] Grade error: {e}", flush=True)
            score = 0.0

        score   = min(max(score, 0.0), 1.0)    #  [0, 1]
        success = score >= SUCCESS_THRESHOLD


    except Exception as e:
        print(f"[DEBUG] Task {task_id} outer error: {e}", flush=True)
        score   = 0.0
        success = False
    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )
    return score

async def main():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
    all_scores = {}

    for task_id in ["easy", "medium", "hard"]:
        score = await run_task(client, task_id)
        all_scores[task_id] = score
        print()  

    # Final summary
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"[SUMMARY] easy={all_scores['easy']:.3f} medium={all_scores['medium']:.3f} hard={all_scores['hard']:.3f} avg={avg:.3f}", flush=True)


    
    

if __name__ == "__main__":
    asyncio.run(main())

