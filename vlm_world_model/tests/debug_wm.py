#!/usr/bin/env python3
"""
World Model Debug Script
=======================
A simple script to debug the world model without requiring Hydra configuration.
This script directly inspects the checkpoint structure and visualizes observations.
"""

import sys
import os
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import argparse
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("wm_debug")

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

def inspect_checkpoint(checkpoint_path, output_dir):
    """Inspect the structure of a checkpoint file."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Get top-level keys
        top_keys = list(checkpoint.keys())
        logger.info(f"Top-level keys: {top_keys}")
        
        # Save top-level keys to file
        with open(output_dir / "checkpoint_structure.txt", "w") as f:
            f.write(f"Checkpoint path: {checkpoint_path}\n")
            f.write(f"Top-level keys: {top_keys}\n\n")
            
            # Explore each top-level key
            for key in top_keys:
                value = checkpoint[key]
                f.write(f"Key: {key}\n")
                
                if isinstance(value, dict):
                    f.write(f"  Type: dict with {len(value)} items\n")
                    f.write(f"  Sub-keys: {list(value.keys())}\n")
                elif isinstance(value, torch.Tensor):
                    f.write(f"  Type: Tensor of shape {value.shape} and dtype {value.dtype}\n")
                elif hasattr(value, '__len__'):
                    f.write(f"  Type: {type(value)} with length {len(value)}\n")
                else:
                    f.write(f"  Type: {type(value)}\n")
                f.write("\n")
                
                # If it's a state dict, explore its structure
                if key.endswith('_state_dict') or key in ['tokenizer', 'world_model', 'actor_critic']:
                    if isinstance(value, dict):
                        f.write(f"  State dict structure for {key}:\n")
                        for param_name, param in value.items():
                            if isinstance(param, torch.Tensor):
                                f.write(f"    {param_name}: Tensor of shape {param.shape} and dtype {param.dtype}\n")
                            else:
                                f.write(f"    {param_name}: {type(param)}\n")
                        f.write("\n")
        
        logger.info(f"Checkpoint structure saved to {output_dir / 'checkpoint_structure.txt'}")
        return True
    
    except Exception as e:
        logger.error(f"Error inspecting checkpoint: {e}")
        return False

def visualize_env(env_name, output_dir, num_frames=5):
    """Capture and visualize frames from an environment."""
    logger.info(f"Visualizing environment: {env_name}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create environment
        env = gym.make(env_name, render_mode="rgb_array")
        logger.info(f"Environment created with action space: {env.action_space}")
        
        # Reset environment
        observation, _ = env.reset()
        logger.info(f"Initial observation shape: {observation.shape}, dtype: {observation.dtype}")
        
        # Save original observation
        plt.figure(figsize=(5, 5))
        plt.imshow(observation)
        plt.title("Original Observation")
        plt.axis('off')
        plt.savefig(output_dir / "original_observation.png")
        plt.close()
        
        # Resize observation to 64x64 for tokenizer
        resized_observation = cv2.resize(observation, (64, 64))
        logger.info(f"Resized observation shape: {resized_observation.shape}")
        
        # Save resized observation
        plt.figure(figsize=(5, 5))
        plt.imshow(resized_observation)
        plt.title("Resized Observation (64x64)")
        plt.axis('off')
        plt.savefig(output_dir / "resized_observation.png")
        plt.close()
        
        # Create a figure to show multiple frames
        fig, axes = plt.subplots(1, num_frames, figsize=(4*num_frames, 4))
        
        # Take random actions and save observations
        for i in range(num_frames):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Display frame
            if num_frames > 1:
                ax = axes[i]
            else:
                ax = axes
            
            ax.imshow(observation)
            ax.set_title(f"Frame {i+1}\nAction: {action}, Reward: {reward}")
            ax.axis('off')
            
            # Save individual frame
            plt.figure(figsize=(5, 5))
            plt.imshow(observation)
            plt.title(f"Frame {i+1} - Action: {action}, Reward: {reward}")
            plt.axis('off')
            plt.savefig(output_dir / f"frame_{i+1}.png")
            plt.close()
            
            # Save resized frame
            resized_frame = cv2.resize(observation, (64, 64))
            plt.figure(figsize=(5, 5))
            plt.imshow(resized_frame)
            plt.title(f"Resized Frame {i+1} - Action: {action}, Reward: {reward}")
            plt.axis('off')
            plt.savefig(output_dir / f"resized_frame_{i+1}.png")
            plt.close()
            
            if terminated or truncated:
                logger.info(f"Episode ended after {i+1} steps")
                break
        
        plt.tight_layout()
        plt.savefig(output_dir / "frames_sequence.png")
        plt.close()
        
        env.close()
        logger.info(f"Environment frames saved to {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error visualizing environment: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Debug World Model")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to inspect")
    parser.add_argument("--env", type=str, default="Breakout-v4", help="Gym environment to visualize")
    parser.add_argument("--output-dir", type=str, default="./wm_debug_results", help="Output directory")
    parser.add_argument("--inspect-checkpoint", action="store_true", help="Inspect checkpoint structure")
    parser.add_argument("--visualize-env", action="store_true", help="Visualize environment frames")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to capture")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run requested operations
    if args.inspect_checkpoint and args.checkpoint:
        inspect_checkpoint(args.checkpoint, output_dir / "checkpoint")
    
    if args.visualize_env:
        visualize_env(args.env, output_dir / "env", args.num_frames)
    
    # If no operation specified, run both
    if not (args.inspect_checkpoint or args.visualize_env):
        if args.checkpoint:
            inspect_checkpoint(args.checkpoint, output_dir / "checkpoint")
        visualize_env(args.env, output_dir / "env", args.num_frames)
    
    logger.info("Debug operations completed")

if __name__ == "__main__":
    main() 