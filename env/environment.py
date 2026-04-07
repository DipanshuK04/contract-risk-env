import json
from models import Contract_Observation,Contract_Actions,Contract_Reward,ContractState
from tasks import TASK_CONFIG
from pathlib import Path
import random

class ContractRiskEnvironment:
    def __init__(self,data_path):
        self.data_path = data_path
        self.contracts = self._load_all_contracts(self.data_path)
        self.current_contract = None
        self.step_count = 0
        self.done = False
        self.actions_taken = []
        self.total_reward = 0.0    
        self.task_id = None

    def _load_all_contracts(self,data_path):
        contracts = {}
        for file in Path(data_path).glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f)
                contracts[data["contract_id"]] = data 
        return contracts

    def reset(self,task_id:str = "easy"):

        self._task_configuration = TASK_CONFIG[task_id]
        self.task_id = task_id
        self.max_steps = self._task_configuration["max_steps"]
        prefix = self._task_configuration["contract_prefix"]
        count  = self._task_configuration["contract_count"]
        random_num = random.randint(1, count)
        contract_id = f"{prefix}{random_num}"
        self.current_contract = self.contracts[contract_id]

        self.step_count = 0
        self.done = False
        self.actions_taken = []
        self.total_reward = 0.0
        
        return self._get_observation(reward=0.0)
    
    def step(self,action:Contract_Actions):
        if self.done:
            raise ValueError("Episode finished. Call reset() first.")
        reward, reason = self._compute_reward(action)
        self.total_reward += reward

        self.step_count += 1
        action_dict = action.dict()
        action_dict["reward"] = reward
        self.actions_taken.append(action_dict)

        if self.step_count >= self.max_steps or action.action_type == "submit_report":
            self.done = True
        info = {'reason':reason, "step": self.step_count}
        obs = self._get_observation(reward=reward)
        return obs,reward,self.done,info
    
    def state(self):
        return ContractState(
            episode_id   = str(id(self)),  
            step_count   = self.step_count,
            task_id      = self.task_id or "unknown",
            contract_id  = self.current_contract["contract_id"] if self.current_contract else "none",
            actions_taken= [a["action_type"] for a in self.actions_taken], 
            total_reward = self.total_reward,
            done         = self.done
        )

    def _compute_reward(self, action: Contract_Actions):
        gt = self.current_contract["ground_truth"]
        clause = action.target_clause
        already_seen = [a["target_clause"] for a in self.actions_taken]
        if clause in already_seen:
            return -0.2, f"Repeated action on {clause}"

        # HARD TASK 
        if self.current_contract["contract_type"] == "CONFLICT_DETECTION":
            conflict_fields = [c["field"] for c in gt["conflicts"]]

            if action.action_type == "flag_risky":   
                if clause in conflict_fields:
                    return 1.0, f"Correct conflict found: {clause}"
                else:
                    return -0.3, f"Wrong: {clause} is not a conflict"

            elif action.action_type == "submit_report":
                caught = {a["target_clause"] for a in self.actions_taken}
                missed = set(conflict_fields) - caught
                penalty = len(missed) * 0.5
                return -penalty, f"Missed conflicts: {list(missed)}"

            return 0.0, "No reward for this action in hard task"
        
        # EASY 
        if self.task_id == "easy":
            if action.action_type == "flag_missing":
                if clause in gt["missing_clauses"]:
                    return 1.0, f"Correct: {clause} is missing"
                else:
                    return -0.3, f"Wrong: {clause} is not missing"
            elif action.action_type == "approve":
                if clause not in gt["missing_clauses"]:
                    return 0.5, f"Correct approval: {clause}"
                else:
                    return -0.3, f"Approved a missing clause: {clause}"
            elif action.action_type == "flag_risky":
                return -0.1, "flag_risky not scored in easy task"   # ← key line
            elif action.action_type == "submit_report":
                caught = {a["target_clause"] for a in self.actions_taken
                        if a["action_type"] == "flag_missing"}
                missed = set(gt["missing_clauses"]) - caught
                penalty = len(missed) * 0.5
                return -penalty, f"Missed missing clauses: {list(missed)}"
        
        # MEDIUM
        if self.task_id == "medium":
            if action.action_type == "flag_missing":
                if clause in gt["missing_clauses"]:
                    return 1.0, f"Correct: {clause} is missing"
                else:
                    return -0.3, f"Wrong: {clause} is not missing"
            elif action.action_type == "flag_risky":
                if clause in gt["risky_values"]:
                    return 1.0, f"Correct: {clause} is risky"
                else:
                    return -0.3, f"Wrong: {clause} is not risky"
            elif action.action_type == "approve":
                all_bad = gt["missing_clauses"] + gt["risky_values"]
                if clause not in all_bad:
                    return 0.5, f"Correct approval: {clause}"
                else:
                    return -0.3, f"Approved a bad clause: {clause}"
            elif action.action_type == "submit_report":
                caught = {a["target_clause"] for a in self.actions_taken}
                all_issues = set(gt["missing_clauses"] + gt["risky_values"])
                missed = all_issues - caught
                penalty = len(missed) * 0.5
                return -penalty, f"Missed: {list(missed)}"
        
        return 0.0,"Unknown Action type"

        
    def _get_observation(self,reward):
        c = self.current_contract
        if c["contract_type"] == "CONFLICT_DETECTION":
            clauses = {
                    "contract_a": c["contract_a"]["clauses"],
                    "contract_b": c["contract_b"]["clauses"]
                }
            values = {
                "contract_a": c["contract_a"]["values"],
                "contract_b": c["contract_b"]["values"]
                }
        else:
            clauses = c["clauses"]
            values  = c['values']

        return Contract_Observation(
            contract_id = c['contract_id'] ,
            contract_type = c['contract_type'],
            clauses = clauses,
            values = values,
            step_number = self.step_count,
            actions_taken=  self.actions_taken,
            remaining_budget=  self.max_steps - self.step_count,
            # task_id: str
            done = self.done,
            reward = reward
        )
    
# print(ContractRiskEnvironment("e:\contract_review\contract-risk-env\data\contracts").reset("easy"))