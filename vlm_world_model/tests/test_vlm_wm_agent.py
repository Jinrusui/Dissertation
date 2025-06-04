#!/usr/bin/env python3
"""
VLMWorldModelAgent Test Script
=============================
This script tests the VLMWorldModelAgent's world model functionality
by sampling random observations and actions from an environment and
checking the agent's ability to predict rewards and outcomes.
"""

import argparse
import logging
import sys
import os
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from iris.src.models.tokenizer.tokenizer import Tokenizer
from iris.src.models.world_model import WorldModel
from iris.src.utils import extract_state_dict
from src.vlm_wm import VLMWorldModelAgent

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("vlm_wm_test")

def create_mock_plans(num_plans, plan_horizon, action_space_size):
    """Create mock plans for testing the world model."""
    plans = []
    for i in range(num_plans):
        actions = [np.random.randint(0, action_space_size) for _ in range(plan_horizon)]
        plan = {
            "actions": actions,
            "explanation": f"Test plan {i+1}",
            "expected_outcome": "Testing world model predictions"
        }
        plans.append(plan)
    return plans

def test_plan_evaluation(agent, env, num_tests=5, num_plans=3, plan_horizon=5, output_dir="./test_results"):
    """Test the agent's ability to evaluate plans using the world model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for test_idx in range(num_tests):
        logger.info(f"Test {test_idx+1}/{num_tests}")
        
        # Reset environment and get initial observation
        observation, _ = env.reset()
        
        # Resize observation to match tokenizer's expected input size (64x64)
        observation = cv2.resize(observation, (64, 64))
        logger.info(f"Resized observation shape: {observation.shape}")
        
        # Create random plans
        plans = create_mock_plans(num_plans, plan_horizon, env.action_space.n)
        
        # Debug: Print the plans
        logger.info(f"Generated plans for test {test_idx+1}:")
        for i, plan in enumerate(plans):
            logger.info(f"  Plan {i+1}: actions={plan['actions']}")
        
        # Test the _pick_best_action method
        try:
            best_action = agent._pick_best_action(observation, plans)
            logger.info(f"Best action selected: {best_action}")
            
            # Find which plan this action came from
            for i, plan in enumerate(plans):
                if plan['actions'][0] == best_action:
                    logger.info(f"Action came from plan {i+1}")
                    break
            
            # Test each plan individually to see scores
            plan_scores = []
            for i, plan in enumerate(plans):
                # Convert observation to tensor
                obs_tensor = agent._rgb_to_bchw(observation, agent.device)
                
                # Encode observation
                with torch.no_grad():
                    enc = agent.tokenizer.encode(obs_tensor, should_preprocess=True)
                    init_tokens = enc.tokens
                
                try:
                    # Simulate the plan
                    score = agent._simulate(init_tokens, plan['actions'])
                    plan_scores.append(score)
                    logger.info(f"  Plan {i+1} score: {score}")
                except Exception as e:
                    logger.error(f"Error simulating plan {i+1}: {str(e)}")
                    plan_scores.append(float('-inf'))  # Use negative infinity for failed plans
            
            # Record results
            test_result = {
                "test_idx": test_idx,
                "plans": [{"actions": p["actions"], "score": float(s)} for p, s in zip(plans, plan_scores)],
                "best_action": int(best_action),
                "best_plan_idx": int(np.argmax(plan_scores)) if plan_scores else -1
            }
            results.append(test_result)
            
            # Save visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(num_plans), plan_scores)
            ax.set_xlabel("Plan Index")
            ax.set_ylabel("Predicted Cumulative Reward")
            ax.set_title(f"Test {test_idx+1}: Plan Scores")
            ax.set_xticks(range(num_plans))
            ax.set_xticklabels([f"Plan {i+1}\n{p['actions']}" for i, p in enumerate(plans)])
            plt.tight_layout()
            plt.savefig(output_dir / f"plan_scores_{test_idx}.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error in test {test_idx+1}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save all results to JSON
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All test results saved to {output_dir}")

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test VLMWorldModelAgent")
    parser.add_argument("--env", type=str, default=cfg.environment.name, help="Gym environment name")
    parser.add_argument("--model-path", type=str, default=os.path.join(cfg.paths.iris_path, "checkpoints", "last.pt"), help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./vlm_wm_test_results", help="Output directory")
    parser.add_argument("--device", type=str, default=cfg.common.device, help="Device to run on (cuda or cpu)")
    parser.add_argument("--num-tests", type=int, default=5, help="Number of tests to run")
    parser.add_argument("--num-plans", type=int, default=cfg.vlm.num_plans, help="Number of plans per test")
    parser.add_argument("--plan-horizon", type=int, default=cfg.vlm.plan_horizon, help="Number of steps per plan")
    
    # Parse only known args to allow Hydra to handle its own args
    args, _ = parser.parse_known_args()
    
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    print(f"Using environment: {args.env}")
    print(f"Using model path: {args.model_path}")
    print(f"Using device: {args.device}")
    
    # Create environment
    env = gym.make(args.env, render_mode="rgb_array")
    logger.info(f"Created environment: {args.env}")
    
    # Load tokenizer and world model
    device = torch.device(args.device)
    logger.info(f"Loading models on device: {device}")
    
    # Initialize tokenizer from config
    tokenizer = instantiate(cfg.tokenizer)
    
    # Load the world model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Initialize world model
    world_model = WorldModel(
        obs_vocab_size=tokenizer.vocab_size,
        act_vocab_size=env.action_space.n,
        config=instantiate(cfg.world_model)
    )
    
    # Load world model state dict
    world_model_state_dict = extract_state_dict(checkpoint, 'world_model')
    world_model.load_state_dict(world_model_state_dict)
    
    # Create agent
    agent = VLMWorldModelAgent(
        model_name="test",
        model="test",
        system_message="Testing only",
        env=env,
        tokenizer=tokenizer,
        world_model=world_model,
        plan_horizon=args.plan_horizon,
        num_plans=args.num_plans,
        device=args.device
    )
    
    # Monkey patch the _query_vlm_for_plans method to avoid actual VLM calls
    def mock_query_vlm(*args, **kwargs):
        return {"plans": create_mock_plans(args.num_plans, args.plan_horizon, env.action_space.n)}
    
    agent._query_vlm_for_plans = mock_query_vlm
    
    # Run tests
    test_plan_evaluation(
        agent=agent,
        env=env,
        num_tests=args.num_tests,
        num_plans=args.num_plans,
        plan_horizon=args.plan_horizon,
        output_dir=args.output_dir
    )
    
    env.close()

if __name__ == "__main__":
    main() 