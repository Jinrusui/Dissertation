#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path

# Add parent directories to path to allow imports
project_dir = Path(__file__).parent.parent
parent_dir = project_dir.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(parent_dir))

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

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_environments(envs_file):
    """Load environment configurations from JSON file."""
    try:
        with open(envs_file) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Environment config file '{envs_file}' not found")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing '{envs_file}'. Please check the file format.")

def load_world_model(iris_path, checkpoint_path=None, device="cuda"):
    """Load tokenizer and world model from checkpoints."""
    logger = logging.getLogger("vlm_wm_eval")
    
    try:
        # Load the tokenizer
        tokenizer_path = os.path.join(iris_path, "checkpoints", "tokenizer.pt")
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.load(tokenizer_path)
        
        # Load the world model
        if checkpoint_path is None:
            checkpoint_path = os.path.join(iris_path, "checkpoints", "last.pt")
        logger.info(f"Loading world model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        world_model = WorldModel(
            obs_vocab_size=tokenizer.vocab_size, 
            act_vocab_size=18,  # Default for Atari, will be updated per env
            config=checkpoint.get("config", {})
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

def main():
    parser = argparse.ArgumentParser(description='Run VLM+WorldModel evaluations using config')
    parser.add_argument('--config', default='config/config.yaml', 
                        help='Path to configuration YAML file')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Override models to evaluate (default: from config)')
    parser.add_argument('--games', nargs='+', default=None,
                        help='Specific games to evaluate (default: all games)')
    parser.add_argument('--output_dir', default=None,
                        help='Override output directory (default: from config)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_logging(
        os.path.join(config['system'].get('log_dir', 'logs/'), 'evaluation_wm.log'),
        config['system'].get('log_level', 'INFO')
    )
    
    logger.info(f"Starting evaluation with config: {args.config}")
    
    # Determine output directory
    output_dir = args.output_dir or os.path.join(config['paths']['results_dir'], 'evaluations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load environment configurations
    envs_file = config['paths'].get('envs_file', 'envs.json')
    if not os.path.isabs(envs_file):
        # Make path relative to atari_gpt directory
        envs_file = os.path.join(config['paths']['atari_gpt_path'], envs_file)
    
    environments = load_environments(envs_file)
    
    # Filter games if specified
    if args.games:
        environments = {k: v for k, v in environments.items() if k in args.games}
        if not environments:
            logger.error(f"None of the specified games {args.games} found in {envs_file}")
            return
    
    # Determine which models to evaluate
    models = args.models or config.get('evaluation', {}).get('models', [config['vlm']['model_key']])
    
    # If 'all' is specified, use a predefined list
    if 'all' in models:
        models = ['gpt4', 'gpt4o', 'gemini', 'claude', 'qwen']
    
    # Load tokenizer and world model
    try:
        iris_path = config['paths']['iris_path']
        checkpoint_path = os.path.join(iris_path, "checkpoints", "last.pt")
        device = config['world_model']['device']
        
        logger.info(f"Loading world model from {iris_path} on {device}")
        tokenizer, world_model = load_world_model(iris_path, checkpoint_path, device)
        logger.info("World model loaded successfully")
        
        # Get plan parameters from config
        plan_horizon = config['vlm']['plan_horizon']
        num_plans = config['vlm']['num_plans']
        
    except Exception as e:
        logger.error(f"Error loading world model: {str(e)}")
        return
    
    # Run evaluations for each model and game
    for model_key in models:
        logger.info(f"Starting evaluation for model: {model_key}")
        
        for game, prompt in environments.items():
            try:
                logger.info(f"Running test for: {game}")
                
                # Run the evaluation with world model
                results = RunWithWorldModel(
                    env_name=game,
                    prompt=prompt,
                    model=model_key,
                    output_dir=output_dir,
                    tokenizer=tokenizer,
                    world_model=world_model,
                    plan_horizon=plan_horizon,
                    num_plans=num_plans,
                    device=device
                )
                
                logger.info(f"Completed test for: {game}")
            except Exception as e:
                logger.error(f"Error running test for {game} with model {model_key}: {str(e)}")
                continue
                
        logger.info(f"Completed evaluation for model: {model_key}")

if __name__ == "__main__":
    main()