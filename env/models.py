from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal

class Contract_Observation(BaseModel):
    contract_id:str
    contract_type: str
    clauses:dict = Field(description='Dictionary of main observations fields in bool')
    values:dict = Field(description='Dictionary of other main observations fields in floats')
    step_number: int
    actions_taken: List[dict]   
    remaining_budget: int  = Field(description='Remaing steps ')
    # task_id: str
    done: bool
    reward: float            

class Contract_Actions(BaseModel): 
    action_type: Literal["flag_missing", "flag_risky", "approve", "submit_report"]
    target_clause: str = Field(description='which target like liability_cap_usd,or termination_notice_days etc')
    severity: Optional[Literal["low", "medium", "high"]] = None
    reason: str = ""

class Contract_Reward(BaseModel):
    score: float    
    feedback: str = Field(description='Why this reward')

class ContractState(BaseModel):
    episode_id: str
    step_count: int
    task_id: str
    contract_id: str
    actions_taken: List[str]  = Field(description='Actiosn taken so far')
    total_reward: float
    done: bool 
    