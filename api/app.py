import sys
import os

# project_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"env")
# sys.path.insert(0, project_root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path     = os.path.join(project_root, "env")
sys.path.insert(0, env_path)
sys.path.insert(0, project_root)

DATA_PATH = os.path.join(project_root, "data", "contracts")

print(project_root)

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from typing import Optional
from environment import ContractRiskEnvironment
from graders import run_grader
from tasks import TASK_CONFIG 
from models import Contract_Actions

app = FastAPI(
    title="Contract Risk Analysis Environment",
    description="AI contract risk analysis",
    version="1.0.0"
)

env = ContractRiskEnvironment(DATA_PATH)

class ResetRequest(BaseModel):
    task_id : str = Field(default="easy",description = "Difficulty level of task")

@app.post('/reset')
def reset(request:ResetRequest):
    if request.task_id not in TASK_CONFIG:
        raise HTTPException(status_code=400, detail=f"Invalid task_id. Choose from: {list(TASK_CONFIG.keys())}")
    obs = env.reset(task_id=request.task_id)
    return obs.dict()

class StepRequest(BaseModel):
    action_type: str
    target_clause: str = Field(description='which target like liability_cap_usd,or termination_notice_days etc')
    severity: Optional[str] = None
    reason: str = ""
    class Config:
        extra = "allow"

@app.post("/step")
def step(request:StepRequest):
    if env.current_contract is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    if env.done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset to start new episode")
    VALID_ACTIONS = ["flag_missing", "flag_risky", "approve", "submit_report"]
    if request.action_type not in VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type '{request.action_type}'.Please see allowed actions as per difficulty.All Allowed actions: {VALID_ACTIONS}"
        )
    VALID_SEVERITY = ["low", "medium", "high"]
    if request.action_type in ["flag_missing", "flag_risky"]:
        if request.severity not in VALID_SEVERITY:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid severity '{request.severity}'. Select from: {VALID_SEVERITY}"
            )
    action = Contract_Actions(
        action_type=request.action_type,
        target_clause=request.target_clause,
        severity=request.severity,
        reason=request.reason
    )
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }




@app.get('/health')
def health():
    return {"status":"ok","environment":"contract-risk-env"}

@app.get("/state")
def state():
    if env.current_contract is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state().dict()

@app.get('/tasks')
def tasks():
    return {
        task_id : {
            "description": value["description"],
            "max_steps": value["max_steps"],
            "difficulty": value["difficulty"],    
            "allowed_actions": value["allowed_actions"],    
            "graded_on": value["graded_on"]      
        }
        for task_id,value in TASK_CONFIG.items()
    }

@app.post('/grade')
def grade():
    if env.current_contract is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    score = run_grader(env.task_id, env)
    return {
        "task_id": env.task_id,
        "score": score,
        "total_reward": env.total_reward,
        # "steps_taken": env.step_count,
        "actions_taken": env.actions_taken
    }

@app.post("/close")
def close():
    return {"status": "closed"}
