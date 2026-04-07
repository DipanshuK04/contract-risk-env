from environment import ContractRiskEnvironment
from models import Contract_Actions
from tasks import TASK_CONFIG
from graders import run_grader
import random

contract_files = "data/contracts"
env = ContractRiskEnvironment(data_path=contract_files)

for task_id in ["easy","medium","hard"]:
    env.reset(task_id=task_id)
    env.step(Contract_Actions(action_type=random.choice(TASK_CONFIG[task_id]["allowed_actions"]), target_clause="ip_ownership", reason="missing"))
    score = run_grader(task_id, env)
    print(f"Grader score: {score}")   

# import sys
# sys.path.insert(0, ".")   # so imports work from env/ folder

# from environment import ContractRiskEnvironment
# from models import Contract_Actions
# from graders import run_grader

# contract_files = "data/contracts"
# env = ContractRiskEnvironment(data_path=contract_files)

# # ── EASY TEST ─────────────────────────────────────────────────
# print("\n=== EASY TASK ===")
# obs = env.reset(task_id="easy")
# print(f"Contract: {obs.contract_id}")
# print(f"Clauses: {obs.clauses}")
# print(f"Ground truth: {env.current_contract['ground_truth']}")

# # Read actual missing clauses and flag them correctly
# missing = env.current_contract["ground_truth"]["missing_clauses"]
# for clause in missing:
#     obs, reward, done, info = env.step(
#         Contract_Actions(action_type="flag_missing", target_clause=clause, reason="not present")
#     )
#     print(f"  flagged '{clause}' → reward: {reward} | {info['reason']}")

# env.step(Contract_Actions(action_type="submit_report", target_clause="none", reason="done"))
# score = run_grader("easy", env)
# print(f"Easy grader score: {score}")  # should be 1.0


# # ── MEDIUM TEST ───────────────────────────────────────────────
# print("\n=== MEDIUM TASK ===")
# obs = env.reset(task_id="medium")
# print(f"Contract: {obs.contract_id}")
# print(f"Ground truth: {env.current_contract['ground_truth']}")

# gt = env.current_contract["ground_truth"]
# for clause in gt["missing_clauses"]:
#     obs, reward, done, info = env.step(
#         Contract_Actions(action_type="flag_missing", target_clause=clause, reason="missing")
#     )
#     print(f"  flag_missing '{clause}' → reward: {reward}")

# for clause in gt["risky_values"]:
#     obs, reward, done, info = env.step(
#         Contract_Actions(action_type="flag_risky", target_clause=clause, reason="risky")
#     )
#     print(f"  flag_risky '{clause}' → reward: {reward}")

# env.step(Contract_Actions(action_type="submit_report", target_clause="none", reason="done"))
# score = run_grader("medium", env)
# print(f"Medium grader score: {score}")  # should be 1.0


# # ── HARD TEST ─────────────────────────────────────────────────
# print("\n=== HARD TASK ===")
# obs = env.reset(task_id="hard")
# print(f"Contract: {obs.contract_id}")
# print(f"Ground truth: {env.current_contract['ground_truth']}")

# conflicts = env.current_contract["ground_truth"]["conflicts"]
# for conflict in conflicts:
#     field = conflict["field"]   # use actual conflict field names
#     obs, reward, done, info = env.step(
#         Contract_Actions(action_type="flag_risky", target_clause=field, reason="conflict detected")
#     )
#     print(f"  flag_risky '{field}' → reward: {reward}")

# env.step(Contract_Actions(action_type="submit_report", target_clause="none", reason="done"))
# score = run_grader("hard", env)
# print(f"Hard grader score: {score}")  # should be 1.0