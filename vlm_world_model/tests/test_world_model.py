#!/usr/bin/env python3
"""
World Model Test Script
======================
This script tests the world model's ability to predict rollouts and rewards
by randomly sampling images and actions from an environment.
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
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from iris.src.models.tokenizer.tokenizer import Tokenizer
from iris.src.models.world_model import WorldModel
from iris.src.utils import extract_state_dict

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("wm_test")

def _rgb_to_bchw(rgb, device):
    """Convert RGB image to batch-channel-height-width tensor."""
    if isinstance(rgb, str):
        raise TypeError(f"Expected image data, got string: {rgb[:30]}...")
        
    if hasattr(rgb, 'convert'):  # Check if it's a PIL Image
        rgb = np.array(rgb)
        
    if hasattr(rgb, 'dtype') and rgb.dtype != np.uint8:
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
    
    # Handle different channel arrangements
    if len(rgb.shape) == 3 and rgb.shape[2] == 3:
        # Standard HWC format
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    elif len(rgb.shape) == 3 and rgb.shape[0] == 3:
        # Already in CHW format
        tensor = torch.from_numpy(rgb).float().div(255.0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image shape: {rgb.shape}")
    
    return tensor.to(device)

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test world model predictions")
    parser.add_argument("--env", type=str, default=cfg.environment.name, help="Gym environment name")
    parser.add_argument("--model-path", type=str, default=os.path.join(cfg.paths.iris_path, "checkpoints", "last.pt"), help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./world_model_test_results", help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default=cfg.common.device, help="Device to run on (cuda or cpu)")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to test")
    parser.add_argument("--rollout-length", type=int, default=cfg.vlm.plan_horizon, help="Length of each rollout")
    
    # Parse only known args to allow Hydra to handle its own args
    args, _ = parser.parse_known_args()
    
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    print(f"Using environment: {args.env}")
    print(f"Using model path: {args.model_path}")
    print(f"Using device: {args.device}")
    
    run_world_model_test(
        env_name=args.env,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device_str=args.device,
        num_samples=args.samples,
        rollout_length=args.rollout_length,
        cfg=cfg
    )

def run_world_model_test(env_name, model_path, output_dir, device_str="cpu", num_samples=5, rollout_length=5, cfg=None):
    """Run tests on the world model by sampling random states and actions."""
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and world model
    logger.info(f"Loading world model from {model_path}")
    
    # Initialize tokenizer from config
    tokenizer = instantiate(cfg.tokenizer)
    logger.info("Tokenizer loaded")
    
    # Load the world model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create environment to get action space size
    env = gym.make(env_name, render_mode="rgb_array")
    action_space_size = env.action_space.n
    logger.info(f"Environment action space size: {action_space_size}")
    
    # Initialize world model with correct action space size
    world_model = WorldModel(
        obs_vocab_size=tokenizer.vocab_size,
        act_vocab_size=action_space_size,
        config=instantiate(cfg.world_model)
    )
    
    # Load world model state dict
    world_model_state_dict = extract_state_dict(checkpoint, 'world_model')
    world_model.load_state_dict(world_model_state_dict)
    logger.info("World model loaded successfully")
    
    tokenizer = tokenizer.to(device).eval()
    world_model = world_model.to(device).eval()
    
    # Get max token limit from config
    max_tokens = cfg.world_model.max_blocks * cfg.world_model.tokens_per_block
    logger.info(f"World model max tokens: {max_tokens}")
    logger.info(f"Tokens per block: {cfg.world_model.tokens_per_block}")
    
    for sample_idx in range(num_samples):
        logger.info(f"Sample {sample_idx+1}/{num_samples}")
        
        # Reset environment and get initial observation
        observation, _ = env.reset()
        
        # Resize observation to match expected input size (64x64 for tokenizer)
        observation = cv2.resize(observation, (64, 64))
        logger.info(f"Resized observation shape: {observation.shape}")
        
        # Convert observation to tensor
        obs_tensor = _rgb_to_bchw(observation, device)
        
        # Encode observation
        with torch.no_grad():
            enc = tokenizer.encode(obs_tensor, should_preprocess=True)
            init_tokens = enc.tokens  # [1,K]
            
        # Calculate how many action steps we can take based on token limit
        num_obs_tokens = init_tokens.shape[1]
        max_actions = max(1, min(rollout_length, max_tokens - num_obs_tokens - 1))
        logger.info(f"Observation tokens: {num_obs_tokens}, Max actions: {max_actions}")
        
        # Random actions for this rollout
        actions = [env.action_space.sample() for _ in range(max_actions)]
        
        # Store predicted rewards and dones
        rewards = []
        dones = []
        
        # Run world model rollout
        tokens = init_tokens.clone()
        
        for action_idx, action in enumerate(actions):
            # Check if we've reached the token limit
            if tokens.shape[1] >= max_tokens - 1:
                logger.warning(f"Reached token limit at step {action_idx+1}/{len(actions)}")
                break
                
            act_tok = torch.tensor([[action]], device=device, dtype=torch.long)
            inp = torch.cat([tokens, act_tok], dim=1)
            
            try:
                with torch.no_grad():
                    out = world_model(inp)
                
                # Extract reward and done predictions
                if out.logits_rewards.size(1) > 0:
                    reward = out.logits_rewards.argmax(dim=-1).item() - 1  # Convert from logits class to {-1,0,1}
                    rewards.append(reward)
                else:
                    logger.warning(f"Empty rewards tensor at step {action_idx}")
                    rewards.append(0)
                
                if out.logits_ends.size(1) > 0:
                    done = out.logits_ends.argmax(dim=-1).item() == 1
                    dones.append(done)
                else:
                    logger.warning(f"Empty ends tensor at step {action_idx}")
                    dones.append(False)
                    done = False
                
                # Get next observation tokens
                tokens = out.logits_observations.argmax(dim=-1)
                
                if done:
                    logger.info(f"World model predicted episode termination at step {len(dones)}")
                    break
            except Exception as e:
                logger.error(f"Error at step {action_idx+1}: {str(e)}")
                break
        
        # Log cumulative reward
        cumulative_reward = sum(rewards)
        logger.info(f"Rollout {sample_idx+1} - Actions: {actions[:len(rewards)]}")
        logger.info(f"Rollout {sample_idx+1} - Rewards: {rewards}")
        logger.info(f"Rollout {sample_idx+1} - Dones: {dones}")
        logger.info(f"Rollout {sample_idx+1} - Cumulative predicted reward: {cumulative_reward:.2f}")
        
        # Save results to a text file
        with open(output_dir / f"rollout_{sample_idx+1}_results.txt", "w") as f:
            f.write(f"Actions: {actions[:len(rewards)]}\n")
            f.write(f"Rewards: {rewards}\n")
            f.write(f"Dones: {dones}\n")
            f.write(f"Cumulative reward: {cumulative_reward}\n")
    
    env.close()
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main() 