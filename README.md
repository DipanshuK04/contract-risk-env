# Contract Risk Analysis Environment

An OpenEnv-compatible environment where an AI agent analyzes contracts
to detect missing clauses, flag risky values and identify conflicts.

## Setup

### Clone and install
```bash
git clone https://github.com/DipanshuK04/contract-risk-env.git
cd contract-risk-env
pip install -r requirements.txt
```

### Start server
```bash
uvicorn api.app:app --host 0.0.0.0 --port 7860
```
### Set Env variables
```bash
# set environment variables in terminal 
set ENV_URL=http://localhost:7860 \ 
set API_BASE_URL=https://router.huggingface.co/v1 \
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \ 
set HF_TOKEN=your_hf_token_here \
# or in powershell(windows use $env:API_KEY:"your_key")
```

```bash
python inference.py
```

## Environment Description

The agent reads structured JSON contract data and takes actions to analyze it.
Each contract has clauses (present/absent) and values (numeric fields).
The agent must identify problems and submit a final report.

## Action Space

| action_type    | When to use                              |
|---------------|------------------------------------------|
| flag_missing  | A clause is present/absent                 |
| flag_risky    | A value is dangerously low/conflicting   |
| approve       | A clause is present and looks fine       |
| submit_report | Done analyzing, end the episode          |
```json
{
  "action_type": "flag_missing",
  "target_clause": "ip_ownership",
  "severity": "high",
  "reason": "IP ownership clause is absent"
}
```

## Observation Space
```json
{
  "contract_id": "nda1",
  "contract_type": "NDA",
  "clauses": {"liability_cap": true, "ip_ownership": false},
  "values": {"liability_cap_usd": 500, "termination_notice_days": 1},
  "step_number": 1,
  "actions_taken": [],
  "remaining_budget": 9,
  "done": false,
  "reward": 0.0
}
```

## Reward Structure

| Event             | Reward |
|------------------|--------|
| Correct flag     | +1.0   |
| Correct approve  | +0.5   |
| Wrong flag       | -0.3   |
| Repeat action    | -0.2   |
| Missed clause    | -0.5   |

## Tasks

### Easy - Clause Detection
- Contract: Simple NDA (clauses)
- Goal: Find missing clauses
- Max steps: 10
- Grader: F1 score on missing clause detection

### Medium - Risk Flagging
- Contract: Vendor agreement (clauses + values)
- Goal: Find missing clauses AND flag risky values
- Max steps: 20
- Grader: Combined F1 on both skills

### Hard - Conflict Detection
- Contract: Two contracts side by side
- Goal: Find fields that contradict between contract A and B
- Max steps: 30
- Grader: F1 score on conflict set

## API Endpoints

| Method | Endpoint | Description              |
|--------|----------|--------------------------|
| GET    | /health  | Health check             |
| POST   | /reset   | Start new episode        |
| POST   | /step    | Send one action          |
| GET    | /state   | Get internal state       |
| GET    | /tasks   | List all tasks           |
| POST   | /grade   | Get episode score 0-1    |
| POST   | /close   | Close the environment    |

## Baseline SCores

| Task   | Score |
|--------|-------|
| Easy   | 1.000 |
| Medium | 1.000 |
| Hard   | 1.000 |
| Avg    | 1.000 |

Model: `meta-llama/Llama-3.3-70B-Instruct`

## Setup
```bash
# Docker
docker build -t contract-risk-env .
docker run -p 7860:7860 contract-risk-env

# Environment variables
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
HF_TOKEN=your_token
ENV_URL=http://localhost:7860
```
