# VLM World Model

This repository contains code for a VLM-based agent that uses a world model to evaluate plans.

## Overview

The VLM World Model agent:
1. Uses a vision-language model (VLM) like GPT-4o, Claude, or Gemini to propose multiple multi-step action plans.
2. Evaluates each plan using an offline world model to predict cumulative rewards.
3. Selects the plan with the best predicted reward and executes its first action.
4. Stores validated plans in an action buffer for future use.

## Action Buffer

The action buffer feature stores validated plans with high scores. When the buffer is non-empty, actions are taken from the buffer rather than generating new plans from the VLM. This provides several benefits:

- **Efficiency**: Reduces the number of VLM calls needed
- **Consistency**: Uses plans that have already been validated by the world model
- **Performance**: Maintains high-quality action sequences once discovered

### Configuration

The action buffer can be configured with these parameters:

- `use_action_buffer`: Enable/disable the action buffer (default: True)
- `action_buffer_size`: Maximum number of plans to store (default: 10)
- `action_buffer_threshold`: Minimum score required for a plan to be stored (default: 0.0)

## Usage

```python
from vlm_world_model.src.vlm_wm import VLMWorldModelAgent
from vlm_world_model.src.action_buffer import ActionBuffer

# Create an action buffer
action_buffer = ActionBuffer(max_buffer_size=10, score_threshold=0.5)

# Create the agent with the action buffer
agent = VLMWorldModelAgent(
    model_name="gpt-4o-2024-11-20",
    model="gpt4o",
    system_message="You are an agent playing an Atari game.",
    env=env,
    tokenizer=tokenizer,
    world_model=world_model,
    plan_horizon=5,
    num_plans=3,
    device="cuda:0",
    use_world_model=True,
    action_buffer=action_buffer
)

# Generate actions
action, response = agent.generate_response("./output_path")
```

## Running Experiments

Use the `run_experiments_wm.py` script to run experiments with the VLM World Model agent:

```bash
python run_experiments_wm.py --env "ALE/Breakout-v5" --model "gpt4o" --use_action_buffer True --action_buffer_size 10
```
