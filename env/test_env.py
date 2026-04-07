from environment import ContractRiskEnvironment
from models import Contract_Actions

env = ContractRiskEnvironment("data/contracts")
obs = env.reset(task_id="easy")
print("Contract loaded:", obs.contract_id)
print("Clauses:", obs.clauses)

action = Contract_Actions(action_type="flag_missing", target_clause="ip_ownership", reason="not present")
obs, reward, done, info = env.step(action)
print("Reward:", reward, "| Info:", info , "\nobs",obs)