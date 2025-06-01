#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directories to path to allow imports
project_dir = Path(__file__).parent.parent
parent_dir = project_dir.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(parent_dir))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

# Import base modules
from vlm_world_model.run_experiments_wm import RunWithWorldModel
from iris.src.models.tokenizer.tokenizer import Tokenizer
from iris.src.models.world_model import WorldModel

def setup_logging(log_path="evaluation_wm.log", log_level="INFO"):
    """Configure logging for the evaluation script."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    # Suppress verbose logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger("vlm_wm_eval")

def load_world_model(cfg: DictConfig, iris_path: str, device: str = "cuda"):
    """Load tokenizer and world model from checkpoints using Hydra config."""
    logger = logging.getLogger("vlm_wm_eval")
    
    try:
        # Initialize tokenizer from config
        tokenizer = instantiate(cfg.tokenizer)
        
        # Load the world model checkpoint
        checkpoint_path = os.path.join(iris_path, "checkpoints", "last.pt")
        logger.info(f"Loading world model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Initialize world model from config
        world_model = WorldModel(
            obs_vocab_size=tokenizer.vocab_size,
            act_vocab_size=18,  # Default for Atari, will be updated per env
            config=instantiate(cfg.world_model.config)
        )
        world_model.load_state_dict(checkpoint["state_dict"])
        
        # Move to device
        device = torch.device(device)
        tokenizer = tokenizer.to(device)
        world_model = world_model.to(device)
        
        # Set to evaluation mode
        tokenizer.eval()
        world_model.eval()
        
        return tokenizer, world_model
    
    except Exception as e:
        logger.error(f"Error loading world model: {e}")
        raise

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Set up logging
    log_dir = cfg.system.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(
        os.path.join(log_dir, 'evaluation_wm.log'),
        cfg.system.log_level
    )
    
    logger.info("Starting evaluation")
    
    # Determine output directory
    output_dir = os.path.join(cfg.paths.results_dir, 'evaluations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed if specified
    if 'seed' in cfg.system:
        import random
        import numpy as np
        seed = cfg.system.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        logger.info(f"Set random seed to {seed}")
    
    # Load world model
    try:
        iris_path = cfg.paths.iris_path
        device = cfg.common.device
        
        logger.info(f"Loading world model from {iris_path} on {device}")
        tokenizer, world_model = load_world_model(cfg, iris_path, device)
        logger.info("World model loaded successfully")
        
        # Get plan parameters from config
        plan_horizon = cfg.vlm.plan_horizon
        num_plans = cfg.vlm.num_plans
        
    except Exception as e:
        logger.error(f"Error loading world model: {str(e)}")
        return
    
    # Determine which models to evaluate
    models = cfg.get('evaluation', {}).get('models', [cfg.vlm.model_key])
    
    # If 'all' is specified, use a predefined list
    if 'all' in models:
        models = ['gpt4', 'gpt4o', 'gemini', 'claude', 'qwen']
    
    # Get environment name
    env_name = cfg.environment.name
    
    # Run evaluations for each model
    logger.info(f"Starting evaluation for game: {env_name}")
    
    for model_key in models:
        try:
            logger.info(f"Running test with model: {model_key}")
            
            # Create system message with properly formatted placeholders
            system_message = cfg.vlm.system_message.format(
                num_plans=num_plans,
                plan_horizon=plan_horizon,
                action_placeholders=cfg.vlm.action_placeholders,
                plan_placeholders=cfg.vlm.plan_placeholders
            )
            
            # Run the evaluation with world model
            results = RunWithWorldModel(
                env_name=env_name,
                prompt=system_message,
                model=model_key,
                output_dir=output_dir,
                tokenizer=tokenizer,
                world_model=world_model,
                plan_horizon=plan_horizon,
                num_plans=num_plans,
                device=device
            )
            
            logger.info(f"Completed test for model: {model_key}")
        except Exception as e:
            logger.error(f"Error running test for {env_name} with model {model_key}: {str(e)}")
            continue
    
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()